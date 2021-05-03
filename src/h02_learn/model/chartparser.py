import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import constants
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .base import BertParser
from ..algorithm.transition_parsers import ShiftReduceParser
from .modules import StackCell, SoftmaxActions, PendingRNN, Agenda, Chart, Item,ItemMH4, ItemW
from .modules import Biaffine, Bilinear, LabelSmoothingLoss
from .hypergraph import ArcStandard, ArcEager, Hybrid, MH4
from collections import defaultdict

# loool
from termcolor import colored


def print_red(s):
    print(colored(s, "red"))


def print_blue(s):
    print(colored(s, "blue"))


def print_green(s):
    print(colored(s, "green"))


def print_yellow(s):
    print(colored(s, "yellow"))


class ChartParser(BertParser):
    def __init__(self,vocabs, embedding_size, rel_embedding_size, batch_size,hidden_size=100, hypergraph=ArcStandard,language="en",
                 dropout=0.33, eos_token_id=28996,mode="agenda-std",transition_system=None):
        super().__init__(language, vocabs,
                         embedding_size=embedding_size, rel_embedding_size=rel_embedding_size,
                         batch_size=batch_size, dropout=dropout)
        self.eos_token_id = eos_token_id
        self.hidden_size = hidden_size
        self.hypergraph = hypergraph
        self.dropout = nn.Dropout(dropout)
        self.mode = mode
        if mode == "agenda-mh4":
            self.parse_step_chart = self.parse_step_mh4
        else:
            self.parse_step_chart = self.parse_step_std_hyb
        self.linear_tree = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # self.linear_label = nn.Linear(hidden_size * 2, self.rel_embedding_size)
        self.linear_dep = nn.Linear(self.hidden_size, 200).to(device=constants.device)

        self.linear_head = nn.Linear(self.hidden_size, 200).to(device=constants.device)
        # self.biaffine = Biaffine(200, 200)
        # self.biaffine_h = Biaffine(200, 200)
        self.bilinear_item = Bilinear(200, 200, 1)

        linear_items1 = nn.Linear(self.hidden_size * 6, self.hidden_size * 4).to(device=constants.device)
        linear_items2 = nn.Linear(self.hidden_size * 4, self.hidden_size * 2).to(device=constants.device)
        linear_items3 = nn.Linear(self.hidden_size * 2, 1).to(device=constants.device)
        layers = [linear_items1, nn.ReLU(), nn.Dropout(dropout), linear_items2, nn.ReLU(), nn.Dropout(dropout),
                  linear_items3]

        #linear_items1mh4 = nn.Linear(self.hidden_size * 4, self.hidden_size * 2).to(device=constants.device)
        #linear_items2mh4 = nn.Linear(self.hidden_size * 2, 1).to(device=constants.device)
        #layers_mh4 = [linear_items1mh4, nn.ReLU(), nn.Dropout(dropout), linear_items2mh4]
        self.mlp = nn.Sequential(*layers)
        #self.mlp_mh4 = nn.Sequential(*layers_mh4)

        #self.linear_items22 = nn.Linear(self.hidden_size * 2, 200).to(device=constants.device)

        #self.linear_ij = nn.Linear(self.hidden_size * 2, self.hidden_size).to(device=constants.device)
        #self.linear_h = nn.Linear(self.hidden_size * 2, self.hidden_size).to(device=constants.device)
        #self.linear_1 = nn.Linear(self.hidden_size * 2, self.hidden_size).to(device=constants.device)

        #self.biaffine_item = Biaffine(self.hidden_size, self.hidden_size, self.hidden_size * 4)

        #self.ln1 = nn.LayerNorm(self.hidden_size).to(device=constants.device)
        #self.ln2 = nn.LayerNorm(self.hidden_size * 2).to(device=constants.device)
        # self.biaffineChart = BiaffineChart(200, 200)

        self.linear_labels_dep = nn.Linear(self.hidden_size, 200).to(device=constants.device)
        self.linear_labels_head = nn.Linear(self.hidden_size, 200).to(device=constants.device)
        self.bilinear_label = Bilinear(200, 200, self.num_rels)

        self.lstm = nn.LSTM(868, self.hidden_size, 2, batch_first=True, bidirectional=True).to(device=constants.device)
        self.lstm_tree = nn.LSTM(self.hidden_size, self.hidden_size, 1, batch_first=False, bidirectional=False).to(
            device=constants.device)

        #encoder_layer = nn.TransformerEncoderLayer(d_model=868, nhead=4)
        #self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        #self.lstm_tree_left = nn.LSTM(self.hidden_size, self.hidden_size, 1, batch_first=False, bidirectional=False).to(
        #    device=constants.device)
        #self.lstm_tree_right = nn.LSTM(self.hidden_size, self.hidden_size, 1, batch_first=False,
        #                               bidirectional=False).to(
        #    device=constants.device)



    def init_pending(self, n, hypergraph):
        pending = {}
        for i in range(n):
            item = hypergraph.axiom(i)
            pending[item.key] = item
        return pending

    def run_lstm(self, x, sent_lens):
        lstm_in = pack_padded_sequence(x, sent_lens.to(device=torch.device("cpu")), batch_first=True,
                                       enforce_sorted=False)
        # ###print(lstm_in)
        lstm_out, _ = self.lstm(lstm_in)
        h_t, _ = pad_packed_sequence(lstm_out, batch_first=True)
        h_t = self.dropout(h_t).contiguous()
        forward_rep = h_t[:, :, :self.hidden_size]
        backward_rep = h_t[:, :, self.hidden_size:]
        return forward_rep, backward_rep

    def tree_representation(self, head, modifier, label):
        reprs = torch.cat([head, modifier, label],
                          dim=-1)

        c = nn.Tanh()(self.linear_tree(reprs))
        return c

    def span_rep(self, words, words_back, i, j, n):
        j = min(j,len(words_back)-1)
        sij = words[j, :] - words[max(i - 1, 0), :]
        sijb = words_back[min(j + 1, n - 1), :] - words_back[i, :]

        sij = torch.cat([sij, sijb], dim=-1).to(device=constants.device)
        return sij

    def init_arc_list(self, tensor_list, oracle_agenda):
        item_list = {}
        for t in tensor_list:
            item = oracle_agenda[(t[0].item(), t[1].item(), t[2].item())]
            item_list[(item.i, item.j, item.h)] = item
        return item_list

    def gold_arc_set(self,ordered_arcs):
        arcs = []
        for t in ordered_arcs:
            ga = (t[0].item(), t[1].item())
            arcs.append(ga)
        return arcs

    def possible_arcs(self, pending, hypergraph):
        arcs = []
        all_items = {}

        pending = hypergraph.merge_pending(pending)
        # print_blue(pending)
        for item in pending.values():
            possible_arcs, possible_items, _ = hypergraph.iterate_spans(item, pending, merge=False, prev_arc=arcs)
            arcs = arcs + possible_arcs
            all_items = {**all_items, **possible_items}

        return arcs, all_items

    def possible_arcs_mh4(self, pending, hypergraph,prev_arcs):
        arcs = []
        all_items = []
        #for item in list(pending.values()):
        #    possible_arcs,possible_items, _ = hypergraph.iterate_spans(item, pending, merge=False, prev_arc=arcs)
        #    arcs = arcs + possible_arcs
        #    all_items = all_items+possible_items#{**all_items, **possible_items}
        #if len(pending) > 1:
        #    pending = hypergraph.merge_pending(pending)
        arcs_new, all_items_new, _ = hypergraph.iterate_spans(None, pending, merge=False, prev_arc=prev_arcs)
        for (pa, pi) in zip(arcs_new, all_items_new):
            if pa not in prev_arcs:
                arcs.append(pa)
                all_items.append(pi)


        return arcs, all_items

    def score_arcs_mh4(self, possible_arcs, gold_arc, possible_items, words_f, words_b):
        gold_index = None
        gold_key = None
        n = len(words_b)
        scores = []

        ga = (gold_arc[0].item(), gold_arc[1].item())
        index2key = {}

        for iter, ((u, v), item) in enumerate(zip(possible_arcs, possible_items)):
            if (u, v) == ga:
                gold_index = torch.tensor([iter], dtype=torch.long).to(device=constants.device)
                gold_key = item.key
        for iter, ((u, v), item) in enumerate(zip(possible_arcs, possible_items)):
            #(h1, h2, h3, h4) = item.key
            if len(item.heads) == 2:
                i = item.heads[0]
                j = item.heads[1]
                span = self.span_rep(words_f, words_b, i, j, n).unsqueeze(0)
            elif len(item.heads) == 3:
                i = item.heads[0]
                mid = item.heads[1]
                j = item.heads[2]
                span_1 = self.span_rep(words_f, words_b, i, mid, n).unsqueeze(0)
                span_2 = self.span_rep(words_f, words_b, mid, j, n).unsqueeze(0)
                span = span_1-span_2
            elif len(item.heads)==4:
                i1 = item.heads[0]
                j1 = item.heads[1]
                i2 = item.heads[2]
                j2 = item.heads[3]
                span_1 = self.span_rep(words_f, words_b, i1, j1, n).unsqueeze(0)
                span_2 = self.span_rep(words_f, words_b, i2, j2, n).unsqueeze(0)
                span = span_1-span_2
            else:
                # len == 1:
                i = item.heads[0]
                span = torch.cat([words_f[i,:], words_b[i,:]], dim=-1).to(device=constants.device).unsqueeze(0)


            #if gold_index is None:
            #    if (u, v) in gold_arc_set:
            #        gold_index = torch.tensor([iter], dtype=torch.long).to(device=constants.device)
            #        gold_key = item.key
            index2key[iter] = item.key
            #if h1 !=-1:
            #    rh1 = torch.cat([words_f[h1,:].unsqueeze(0),words_b[h1,:].unsqueeze(0)],dim=-1)
            #    rh1 = self.dropout(F.relu(self.linear_1(rh1)))
            #else:
            #    rh1 = torch.zeros_like(words_b[0,:].unsqueeze(0)).to(device=constants.device)
            #if h2 !=-1:
            #    rh2 = torch.cat([words_f[h2,:].unsqueeze(0),words_b[h2,:].unsqueeze(0)],dim=-1)
            #    rh2 = self.dropout(F.relu(self.linear_1(rh2)))
            #else:
            #    rh2 = torch.zeros_like(words_b[0,:].unsqueeze(0)).to(device=constants.device)
            #if h3 !=-1:
            #    rh3 = torch.cat([words_f[h3,:].unsqueeze(0),words_b[h3,:].unsqueeze(0)],dim=-1)
            #    rh3 = self.dropout(F.relu(self.linear_1(rh3)))
            #else:
            #    rh3 = torch.zeros_like(words_b[0,:].unsqueeze(0)).to(device=constants.device)
            #if h4 !=-1:
            #    rh4 = torch.cat([words_f[h4,:].unsqueeze(0),words_b[h4,:].unsqueeze(0)],dim=-1)
            #    rh4 = self.dropout(F.relu(self.linear_1(rh4)))
            #else:
            #    rh4 = torch.zeros_like(words_b[0,:].unsqueeze(0)).to(device=constants.device)
            #if h5 !=-1:
            #    rh5 = torch.cat([words_f[h5,:].unsqueeze(0),words_b[h5,:].unsqueeze(0)],dim=-1)
            #    rh5 = self.dropout(F.relu(self.linear_1(rh5)))
            #else:
            #    rh5 = torch.zeros_like(words_b[0,:].unsqueeze(0)).to(device=constants.device)
            #rep = torch.cat([rh1,rh2,rh3,rh4],dim=-1)
            #s = self.mlp_mh4(rep)
            #scores.append(s)
            fwd_rep = torch.cat([words_f[u, :], words_f[v, :]], dim=-1).unsqueeze(0)
            bckw_rep = torch.cat([words_b[u, :], words_b[v, :]], dim=-1).unsqueeze(0)
            rep = torch.cat([span, fwd_rep, bckw_rep], dim=-1)
            s = self.mlp(rep)
            scores.append(s)
        scores = torch.stack(scores, dim=-1).squeeze(0)
        if not self.training or gold_index is None:
            #print_green(scores)
            gold_index = torch.argmax(scores, dim=-1)
            #print_red(gold_index)
            gold_key = index2key[gold_index.item()]
        return scores, gold_index, gold_key


    def score_arcs(self, possible_arcs, gold_arc, possible_items, words_f, words_b):
        gold_index = None
        gold_key = None
        n = len(words_b)
        scores = []
        ga = (gold_arc[0].item(), gold_arc[1].item())
        index2key = {}

        for iter, ((u, v), item) in enumerate(zip(possible_arcs, possible_items.values())):
            i, j, h = item.i, item.j, item.h
            if (u, v) == ga:
                gold_index = torch.tensor([iter], dtype=torch.long).to(device=constants.device)
                gold_key = item.key
            index2key[iter] = item.key
            span = self.span_rep(words_f, words_b, i, j, n).unsqueeze(0)
            fwd_rep = torch.cat([words_f[u, :], words_f[v, :]], dim=-1).unsqueeze(0)
            bckw_rep = torch.cat([words_b[u, :], words_b[v, :]], dim=-1).unsqueeze(0)
            rep = torch.cat([span, fwd_rep, bckw_rep], dim=-1)
            s = self.mlp(rep)
            scores.append(s)
        scores = torch.stack(scores, dim=-1).squeeze(0)
        if not self.training:
            gold_index = torch.argmax(scores, dim=-1)
            gold_key = index2key[gold_index.item()]
        return scores, gold_index, gold_key

    def parse_step_std_hyb(self,pending, hypergraph,arcs,gold_arc,words_f,words_b):
        #print(pending)
        possible_arcs, items = self.possible_arcs(pending, hypergraph)
        #print(possible_arcs)
        scores, gold_index, gold_key = self.score_arcs(possible_arcs,
                                                           gold_arc, items, words_f, words_b)
        gind = gold_index.item()
        made_arc = possible_arcs[gind]
        #print_green(made_arc)
        #print_red(gold_arc)
        made_item = items[gold_key]
        if made_item.l.key in pending.keys():
            del pending[made_item.l.key]
        if made_item.r.key in pending.keys():
            del pending[made_item.r.key]
        pending[made_item.key] = made_item

        h = made_arc[0]
        m = made_arc[1]
        #gold_arc_set.remove((h,m))
        #print_yellow(gold_arc_set)
        hypergraph = hypergraph.set_head(m)
        #hypergraph.made_arcs.append(made_arc)
        arcs.append(made_arc)
        return scores, gold_index, h, m, arcs, hypergraph, pending

    def parse_step_mh4(self,pending, hypergraph,arcs,gold_arc,words_f,words_b):
        #pending = hypergraph.calculate_pending()

        possible_arcs, items = self.possible_arcs_mh4(pending, hypergraph, arcs)
        scores, gold_index, gold_key = self.score_arcs_mh4(possible_arcs,
                                                           gold_arc, items, words_f, words_b)
        gind = gold_index.item()

        made_arc = possible_arcs[gind]
        #if self.training:
        #    gold_arc_set.remove(made_arc)
        h = made_arc[0]
        m = made_arc[1]
        h = min(h,hypergraph.n-1)
        m = min(m,hypergraph.n-1)
        pending = hypergraph.calculate_pending(pending, m)
        hypergraph = hypergraph.set_head(m)
        #hypergraph.made_arcs.append(made_arc)
        arcs.append(made_arc)
        return scores, gold_index, h, m, arcs, hypergraph, pending

    def forward(self, x, transitions, relations, map, heads, rels):
        x_ = x[0][:, 1:]
        # average of last 4 hidden layers


        with torch.no_grad():
            out = self.bert(x_.to(device=constants.device))[2]
            x_emb = torch.stack(out[-8:]).mean(0)
        heads_batch = torch.ones((x_emb.shape[0], heads.shape[1])).to(device=constants.device)  # * -1
        prob_sum = 0
        batch_loss = 0
        x_mapped = torch.zeros((x_emb.shape[0], heads.shape[1] + 1, x_emb.shape[2] + 100)).to(device=constants.device)
        eos_emb = x_emb[0, -1, :].unsqueeze(0).to(device=constants.device)
        eos_emb = torch.cat([eos_emb, torch.zeros((1, 100)).to(device=constants.device)], dim=-1).to(
            device=constants.device)
        for i, sentence in enumerate(x_emb):
            # initialize a parser
            mapping = map[i, map[i] != -1]
            tag = self.tag_embeddings(x[1][i][x[1][i] != 0].to(device=constants.device))
            sentence = sentence[:-1, :]
            s = self.get_bert_embeddings(mapping, sentence, tag)
            s = torch.cat([s, eos_emb], dim=0)
            curr_sentence_length = s.shape[0]
            x_mapped[i, :curr_sentence_length, :] = s
        sent_lens = (x_mapped[:, :, 0] != 0).sum(-1).to(device=constants.device)
        h_t, bh_t = self.run_lstm(x_mapped, sent_lens)
        h_t_noeos = torch.zeros((h_t.shape[0], heads.shape[1], h_t.shape[2])).to(device=constants.device)
        for i in range(h_t.shape[0]):

            n = int(sent_lens[i] - 1)
            ordered_arcs = transitions[i]
            mask = (ordered_arcs.sum(dim=1) != -2)
            ordered_arcs = ordered_arcs[mask, :]

            s = h_t[i, :n + 1, :]
            s_b = bh_t[i, :n + 1, :]

            # trees = torch.exp(curr_init_weights)
            arcs = []
            history = defaultdict(lambda: 0)
            loss = 0
            popped = []

            right_children = {i: [i] for i in range(n)}
            left_children = {i: [i] for i in range(n)}
            words_f = s  # .clone()
            words_b = s_b  # .clone()
            #gold_arc_set = self.gold_arc_set(ordered_arcs)
            hypergraph = self.hypergraph(n)
            #hypergraph = self.hypergraph(n, gold_arc_set)
            if self.mode == "agenda-mh4":
                pending = hypergraph.calculate_pending_init()
            else:
                pending = self.init_pending(n, hypergraph)
            #pending = self.init_pending(n, hypergraph)
            for iter, gold_arc in enumerate(ordered_arcs):

                scores, gold_index, h, m, arcs, hypergraph,pending = self.parse_step_chart(pending,
                                                                                            hypergraph,
                                                                                            arcs,
                                                                                            gold_arc,
                                                                                            words_f,
                                                                                            words_b)

                if self.training:
                    loss += nn.CrossEntropyLoss(reduction='sum')(scores, gold_index)
                if h < m:
                    # m is a right child
                    right_children[h].append(m)
                else:
                    # m is a left child
                    left_children[h].append(m)
                h_rep = self.tree_lstm(words_f, left_children[h], right_children[h])
                words_f = words_f.clone()
                words_f[h, :] = h_rep

                h_rep = self.tree_lstm(words_b, left_children[h], right_children[h])
                words_b = words_b.clone()
                words_b[h, :] = h_rep

            loss /= len(ordered_arcs)
            pred_heads = self.heads_from_arcs(arcs, n)
            heads_batch[i, :n] = pred_heads

            h_t_noeos[i, :n, :] = h_t[i, :n, :]
            batch_loss += loss
        batch_loss /= x_emb.shape[0]
        heads = heads_batch
        l_logits = self.get_label_logits(h_t_noeos, heads)
        rels_batch = torch.argmax(l_logits, dim=-1)
        batch_loss += self.loss(batch_loss, l_logits, rels)
        return batch_loss, heads_batch, rels_batch

    def get_head_logits(self, h_t, sent_lens):
        h_dep = self.dropout(F.relu(self.linear_dep(h_t)))
        h_arc = self.dropout(F.relu(self.linear_head(h_t)))

        h_logits = self.biaffine(h_arc, h_dep)

        # Zero logits for items after sentence length
        for i, sent_len in enumerate(sent_lens):
            h_logits[i, sent_len:, :] = 0
            h_logits[i, :, sent_len:] = 0

        return h_logits

    def get_label_logits(self, h_t, head):
        l_dep = self.dropout(F.relu(self.linear_labels_dep(h_t)))
        l_head = self.dropout(F.relu(self.linear_labels_head(h_t)))
        # head_int = torch.zeros_like(head,dtype=torch.int64)
        head_int = head.clone().type(torch.int64)
        if self.training:
            assert head is not None, 'During training head should not be None'
        l_head = l_head.gather(dim=1, index=head_int.unsqueeze(2).expand(l_head.size()))

        l_logits = self.bilinear_label(l_dep, l_head)
        return l_logits

    def init_agenda_oracle(self, oracle_hypergraph, rels):
        pending = defaultdict(lambda: 0)
        for item_tree in oracle_hypergraph:
            left = item_tree[0]
            right = item_tree[1]
            derived = item_tree[2]
            if left[0] == left[2] and left[0] + 1 == left[1]:
                # is axiom
                left_item = Item(left[0].item(), left[1].item(), left[2].item(),
                                 left[0].item(), left[0].item())
                pending[(left[0].item(), left[1].item(), left[2].item())] = left_item
            else:
                left_item = pending[(left[0].item(), left[1].item(), left[2].item())]
            if right[0] == right[2] and right[0] + 1 == right[1]:
                # is axiom
                right_item = Item(right[0].item(), right[1].item(), right[2].item(),
                                  right[0].item(), right[0].item())
                pending[(right[0].item(), right[1].item(), right[2].item())] = right_item
            else:
                right_item = pending[(right[0].item(), right[1].item(), right[2].item())]
            arc_made = (derived[2].item(), left_item.h if left_item.h != derived[2].item() else right_item.h)
            m = arc_made[1] - 1
            rel_made = rels[m]

            item = Item(derived[0].item(),
                        derived[1].item(),
                        derived[2].item(),
                        left_item, right_item)
            item = item.add_rel(rel_made)
            pending[(derived[0].item(), derived[1].item(), derived[2].item())] = item
        return pending

    def heads_from_arcs(self, arcs, sent_len):
        heads = [0] * sent_len
        for (u, v) in arcs:
            heads[min(v,sent_len-1)] = u  # .item()
        return torch.tensor(heads).to(device=constants.device)

    def loss(self, batch_loss, l_logits, rels):
        criterion_l = nn.CrossEntropyLoss(ignore_index=0).to(device=constants.device)
        loss = criterion_l(l_logits.reshape(-1, l_logits.shape[-1]), rels.reshape(-1))
        return loss + batch_loss

    def tree_lstm(self, x, left_children, right_children):

        left_reps = x[list(left_children), :].unsqueeze(1).to(device=constants.device)
        right_reps = x[list(right_children), :].to(device=constants.device)
        right_reps = torch.flip(right_reps, dims=[0, 1]).unsqueeze(1).to(device=constants.device)

        _, (lh, _) = self.lstm_tree(left_reps)
        _, (rh, _) = self.lstm_tree(right_reps)

        c = torch.cat([lh, rh], dim=-1).to(device=constants.device)
        c = nn.Tanh()(self.linear_tree(c))
        return c

    def predict_next_prn(self, words, words_back, items, hypergraph, oracle_item, prune=True):
        scores = []
        n = len(words)
        gold_index = None
        next_item = None
        winner_item = None
        ij_set = []
        h_set = []
        keys_to_delete = []
        all_embedding = self.item_lstm.embedding()  # .squeeze(0)

        for iter, item in enumerate(items.values()):
            i, j, h = item.i, item.j, item.h
            if prune:
                if item.l in hypergraph.bucket or item.r in hypergraph.bucket:
                    continue
            ij_set.append((i, j))
            h_set.append(h)
        ij_set = set(ij_set)
        h_set = set(h_set)
        unique_ij = len(ij_set)
        unique_h = len(h_set)
        ij_tens = torch.zeros((unique_ij, self.hidden_size * 2)).to(device=constants.device)
        h_tens = torch.zeros((unique_h, self.hidden_size * 4)).to(device=constants.device)

        index_matrix = torch.ones((unique_ij, unique_h), dtype=torch.int64).to(device=constants.device) * -1
        ij_counts = {(i, j): 0 for (i, j) in list(ij_set)}
        h_counts = {h: 0 for h in list(h_set)}
        ij_rows = {}
        h_col = {}
        ind_ij = 0
        ind_h = 0
        # prev_scores = []
        # for k in keys_to_delete:
        #    del items[k]
        for iter, item in enumerate(items.values()):
            i, j, h = item.i, item.j, item.h
            if prune:
                if item.l in hypergraph.bucket or item.r in hypergraph.bucket:
                    continue
                if i == oracle_item[0] and j == oracle_item[1] and h == oracle_item[2]:
                    gold_index = torch.tensor([iter], dtype=torch.long).to(device=constants.device)
            ij_counts[(i, j)] += 1
            h_counts[h] += 1
            # prev_scores.append(item.score)
            if ij_counts[(i, j)] <= 1:
                ij_rows[(i, j)] = ind_ij
                sij = self.span_rep(words, words_back, i, j, n)  # torch.cat([sij,sijb],dim=-1)
                # w_ij = words[i:j + 1, :].unsqueeze(1).to(device=constants.device)
                # _, (unrootedtree_ij, _) = self.lstm_tree(w_ij)
                # print_yellow(unrootedtree_ij.squeeze(0).shape)
                # print_blue(words[i,:].shape)
                # print_red(all_embedding.shape)
                # rep = torch.cat([unrootedtree_ij.squeeze(0), words[i, :].unsqueeze(0), words[j, :].unsqueeze(0)],
                #                dim=-1)
                ij_tens[ind_ij, :] = sij  # rep  # unrootedtree_ij.squeeze(0)
                ind_ij += 1
            if h_counts[h] <= 1:
                h_col[h] = ind_h
                rep = torch.cat([words[h, :].unsqueeze(0), all_embedding], dim=-1)
                h_tens[ind_h, :] = rep  # words[h, :].unsqueeze(0).to(device=constants.device)
                ind_h += 1

            index_matrix[ij_rows[(i, j)], h_col[h]] = iter
        tmp = self.linear_items1(ij_tens)
        tmp2 = self.linear_items2(h_tens)
        h_ij = self.dropout(self.linear_items11(self.dropout(F.relu(self.ln1(tmp))))).unsqueeze(0)
        h_h = self.dropout(self.linear_items22(self.dropout(F.relu(self.ln2(tmp2))))).unsqueeze(0)
        # h_h = self.dropout(self.linear_items22(self.dropout(F.relu(self.ln2(tmp2))))).unsqueeze(0)
        # h_h = self.dropout(F.relu(self.linear_items2(h_tens))).unsqueeze(0)
        item_logits = self.biaffine_item(h_ij, h_h).squeeze(0)
        # prev_scores = torch.stack(prev_scores, dim=-1)
        scores = item_logits[index_matrix != -1].unsqueeze(0)  # + prev_scores
        # ind = 0
        # for iter, item in enumerate(items.values()):
        #    if prune:
        #        if item.l in hypergraph.bucket or item.r in hypergraph.bucket:
        #            continue
        #    item.update_score(scores[:, ind])
        #    ind += 1

        winner = torch.argmax(scores, dim=-1)
        """
        score all spans at once, 
        do dp 
        calculate loss etc

        """
        if prune:
            winner_item = list(items.values())[winner]
            if gold_index is not None:
                next_item = list(items.values())[gold_index]
        else:
            if self.training:
                gold_index, next_item = hypergraph.return_gold_next(items)
            else:
                gold_index = None
                next_item = list(items.values())[winner]

        return scores, winner_item, gold_index, hypergraph, next_item, items

    def smooth_one_hot(self, true_labels, classes, smoothing=0.0):
        """
        if smoothing == 0, it's one-hot method
        if 0 < smoothing < 1, it's smooth method

        """
        classes = max(2, classes)
        assert 0 <= smoothing < 1
        confidence = 1.0 - smoothing
        label_shape = torch.Size((true_labels.size(0), classes))
        with torch.no_grad():
            true_dist = torch.empty(size=label_shape, device=true_labels.device)
            true_dist.fill_(smoothing / (classes - 1))
            true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
        return true_dist

    def take_step(self, x, x_b, gold_next_item, hypergraph, oracle_agenda, pred_item, pending):
        if self.training:

            key = (gold_next_item.i, gold_next_item.j, gold_next_item.h)
            di = gold_next_item
        else:
            di = pred_item
            key = (pred_item.i, pred_item.j, pred_item.h)

        # rep = torch.cat([x[di.i, :], x[di.j, :], x[di.h, :]], dim=-1).unsqueeze(0).to(device=constants.device)
        spanij = self.span_rep(x, x_b, di.i, di.j, len(x))
        rep = torch.cat([spanij, x[di.h, :]], dim=-1).unsqueeze(0)
        self.item_lstm.push(rep)
        made_arc = None
        del pending[key]
        if isinstance(di.l, Item):
            # h = di.h
            # m = di.l.h if di.l.h != h else di.r.h
            # (h, m)
            made_arc, _ = hypergraph.make_arc(di)
            (h, m) = made_arc
            if h < m:
                # m is a right child
                hypergraph = hypergraph.add_right_child(h, m)
            else:
                # m is a left child
                hypergraph = hypergraph.add_left_child(h, m)

        if di.l in hypergraph.bucket or di.r in hypergraph.bucket:
            # h = di.h
            # m = di.l.h if di.l.h != h else di.r.h
            # made_arc = (h, m)
            made_arc, _ = hypergraph.make_arc(di)
            (h, m) = made_arc
            if h < m:
                # m is a right child
                hypergraph = hypergraph.add_right_child(h, m)
            else:
                # m is a left child
                hypergraph = hypergraph.add_left_child(h, m)
            scores, gold_index, rel_loss = None, None, 0
        else:
            hypergraph = hypergraph.update_chart(di)
            hypergraph = hypergraph.add_bucket(di)
            possible_items = hypergraph.outgoing(di)

            if len(possible_items) > 0:

                scores, winner_item, gold_index, hypergraph, new_item, _ = self.predict_next_prn(x, x_b, possible_items,
                                                                                                 hypergraph, None,
                                                                                                 False)
                if new_item is not None:
                    pending[(new_item.i, new_item.j, new_item.h)] = new_item

            else:
                scores, gold_index = None, None
                pass

        return hypergraph, pending, di, made_arc, scores, gold_index


    def score_arcs_biaffine(self, possible_arcs, gold_arc, possible_items, words_f, words_b):
        n = len(words_f)
        gold_index = None
        gold_key = None
        ij_set = []
        h_set = []

        for iter, item in enumerate(possible_items.values()):
            i, j, h = item.i, item.j, item.h
            ij_set.append((i, j))
            h_set.append(h)
        ij_set = set(ij_set)
        h_set = set(h_set)
        unique_ij = len(ij_set)
        unique_h = len(h_set)
        ij_tens = torch.zeros((unique_ij, self.hidden_size * 2)).to(device=constants.device)
        h_tens = torch.zeros((unique_h, self.hidden_size * 2)).to(device=constants.device)

        index_matrix = torch.ones((unique_ij, unique_h), dtype=torch.int64).to(device=constants.device) * -1
        ij_counts = {(i, j): 0 for (i, j) in list(ij_set)}
        h_counts = {h: 0 for h in list(h_set)}
        ij_rows = {}
        h_col = {}
        ind_ij = 0
        ind_h = 0
        ga = (gold_arc[0].item(), gold_arc[1].item())
        index2key = {}
        for iter, ((u, v), item) in enumerate(zip(possible_arcs, possible_items.values())):
            i, j, h = item.i, item.j, item.h
            if (u, v) == ga:
                gold_index = torch.tensor([iter], dtype=torch.long).to(device=constants.device)
                gold_key = (i, j, h)
            index2key[iter] = (i, j, h)
            ij_counts[(i, j)] += 1
            h_counts[h] += 1
            if ij_counts[(i, j)] <= 1:
                ij_rows[(i, j)] = ind_ij
                sij = self.span_rep(words_f, words_b, i, j, n)
                ij_tens[ind_ij, :] = sij
                ind_ij += 1
            if h_counts[h] <= 1:
                h_col[h] = ind_h
                rep = torch.cat([words_f[h, :].unsqueeze(0), words_b[h, :].unsqueeze(0)], dim=-1)
                h_tens[ind_h, :] = rep
                ind_h += 1
            index_matrix[ij_rows[(i, j)], h_col[h]] = iter

        history_rep = self.item_lstm.embedding()
        h_ij = self.dropout(F.relu(self.linear_ij(ij_tens))).unsqueeze(0)
        h_h = self.dropout(F.relu(self.linear_h(h_tens))).unsqueeze(0)
        item_logits = self.biaffine_item(h_ij, h_h, history_rep).squeeze(0)
        scores = item_logits[index_matrix != -1].unsqueeze(0)

        if not self.training:
            gold_index = torch.argmax(scores, dim=-1)
            gold_key = index2key[gold_index.item()]
        return scores, gold_index, gold_key

    def get_args(self):
        return {
            'language':self.language,
            'hidden_size':self.hidden_size,
            'hypergraph':self.hypergraph,
            'vocabs': self.vocabs,
            'embedding_size': self.embedding_size,
            'rel_embedding_size': self.rel_embedding_size,
            'dropout': self.dropout_prob,
            'batch_size': self.batch_size,
        }