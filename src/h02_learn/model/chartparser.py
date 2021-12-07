import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import constants
import itertools
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
        self.hidden_size = 868
        self.hypergraph = hypergraph
        self.dropout = nn.Dropout(dropout)
        self.mode = mode
        if mode == "agenda-mh4":
            self.parse_step_chart = self.parse_step_mh4
        else:
            self.parse_step_chart = self.parse_step_std_hyb
        #self.linear_tree = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.linear_tree = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # self.linear_label = nn.Linear(hidden_size * 2, self.rel_embedding_size)
        self.linear_dep = nn.Linear(self.hidden_size, 200).to(device=constants.device)

        self.linear_head = nn.Linear(self.hidden_size, 200).to(device=constants.device)
        # self.biaffine = Biaffine(200, 200)
        # self.biaffine_h = Biaffine(200, 200)
        self.bilinear_item = Bilinear(200, 200, 1)


        linear_items12 = nn.Linear(int(self.hidden_size),int(self.hidden_size/2)).to(device=constants.device)
        linear_items22 = nn.Linear(int(self.hidden_size/2), int(self.hidden_size/4)).to(device=constants.device)
        linear_items32 = nn.Linear(int(self.hidden_size/4), 1).to(device=constants.device)

        layers = [linear_items12, nn.ReLU(), nn.Dropout(dropout), linear_items22, nn.ReLU(), nn.Dropout(dropout),
                  linear_items32]

        self.mlp = nn.Sequential(*layers)

        self.linear_arc_head = nn.Linear(self.hidden_size, 400).to(device=constants.device)
        self.linear_arc_dep = nn.Linear(self.hidden_size, 400).to(device=constants.device)
        self.biaffine = Biaffine(400, 400)

        self.linear_labels_dep = nn.Linear(self.hidden_size, 200).to(device=constants.device)
        self.linear_labels_head = nn.Linear(self.hidden_size, 200).to(device=constants.device)
        self.bilinear_label = Bilinear(200, 200, self.num_rels)

        #self.lstm = nn.LSTM(868, self.hidden_size, 2, batch_first=True, bidirectional=True).to(device=constants.device)
        #self.lstm_tree = nn.LSTM(self.hidden_size, self.hidden_size, 1, batch_first=False, bidirectional=False).to(
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

    def get_max_index(self, scores):
        d = torch.tensor(scores.shape[0]).to(device=constants.device)
        n1 = torch.tensor(1).to(device=constants.device)
        x = scores.reshape(1, 1, scores.shape[0], scores.shape[1])
        m = x.view(n1, -1).argmax(1)
        indices = torch.cat(((m // d).view(-1, 1), (m % d).view(-1, 1)), dim=1)
        return indices

    def score_arc_labels_biliniear(self, words):
        # after head is chosen, pick that row against every label?
        l_dep = self.dropout(F.relu(self.linear_label_dep(words)))
        l_head = self.dropout(F.relu(self.linear_label_head(words)))
        # if self.training:
        #    assert head is not None, 'During training head should not be None'
        # l_head = l_head.gather(dim=1, index=head.unsqueeze(2).expand(l_head.size()))
        l_logits = self.bilinear_label(l_dep, l_head)
        #print_green(l_logits.shape)
        return l_logits

    def score_arcs_biaffine(self, possible_arcs, gold_arc, words, all_possible_arcs):

        h_dep = self.dropout(F.relu(self.linear_arc_dep(words)))
        h_arc = self.dropout(F.relu(self.linear_arc_head(words)))

        scores = self.biaffine(h_arc, h_dep)
        gold_index = None
        # zero logits not possible due to tansition system
        #to_be_zeroed_out = all_possible_arcs.difference(set(possible_arcs))
        if not self.training:
            made_arc = self.get_max_index(scores)[0]
        else:
            made_arc = gold_arc



        for iter, (i, j) in enumerate(possible_arcs):
            if (i, j) == (gold_arc[0].item(), gold_arc[1].item()):
                gold_index = torch.tensor(iter).to(device=constants.device)
        if self.training and gold_index is not None:
            keep_inds = torch.zeros((scores.shape[0] * scores.shape[1])).to(device=constants.device)
            for iter, (i,j) in enumerate(possible_arcs):
                keep_inds[i*scores.shape[0]+j] = 1
            mask_ = keep_inds.eq(1)
            flat_scores_only_valid = torch.masked_select(torch.flatten(scores),mask_)
            scores = nn.Softmax(dim=-1)(flat_scores_only_valid)
        else:
            gold_index = torch.tensor(made_arc[0].item() * scores.shape[0] + made_arc[1].item()).to(
                device=constants.device)
            scores = torch.flatten(scores)
            scores = nn.Softmax(dim=-1)(scores)

        #scores = nn.Softmax()(scores)
        #for (i, j) in to_be_zeroed_out:
        #    scores[i, j] = 0
        #    scores[j, i] = 0


        return scores, made_arc, gold_index  # , gold_key

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

    def get_item_representation(self, item, words):
        n = len(words)
        if len(item.heads) == 2:
            i = item.heads[0]
            j = item.heads[1]
            j = min(j, n - 1)
            i = max(i-1,0)
            item_r = torch.mean(words[i:j,:],dim=0)

        elif len(item.heads) == 3:
            i = item.heads[0]
            mid = item.heads[1]
            j = item.heads[2]

            item_r1 = torch.mean(words[max(i - 1, 0):min(mid, n - 1), :],dim=0)
            item_r2 = torch.mean(words[max(mid - 1, 0):min(j, n - 1), :],dim=0)

            item_r = torch.stack([item_r1,item_r2],dim=0).sum(dim=0).div(2)

            # span = span_1-span_2
        elif len(item.heads) == 4:
            i1 = item.heads[0]
            j1 = item.heads[1]
            i2 = item.heads[2]
            j2 = item.heads[3]

            item_r1 = torch.mean(words[max(i1 - 1, 0):min(j1, n - 1), :],dim=0)
            item_r2 = torch.mean(words[max(i2 - 1, 0):min(j2, n - 1), :],dim=0)

            item_r = torch.stack([item_r1,item_r2],dim=0).sum(dim=0).div(2)

            # span = span_1-span_2
        else:
            # len == 1:
            i = item.heads[0]
            item_r = words[i, :]

        return item_r

    def score_arcs_mh4(self, possible_arcs, gold_arc, possible_items, words):
        gold_index = None
        gold_key = None
        n = len(words)
        scores = []

        ga = (gold_arc[0].item(), gold_arc[1].item())
        index2key = {}

        for iter, ((u, v), item) in enumerate(zip(possible_arcs, possible_items)):
            if (u, v) == ga:
                gold_index = torch.tensor([iter], dtype=torch.long).to(device=constants.device)
                gold_key = item.key
        all_spans = []
        for iter, ((u, v), item) in enumerate(zip(possible_arcs, possible_items)):
            span = self.get_item_representation(item, words)
            all_spans.append(span)
            index2key[iter] = item.key

        scores = self.mlp(torch.stack(all_spans,dim=0)).permute(1,0)#.squeeze(1)

        if not self.training or gold_index is None:
            gold_index = torch.argmax(scores, dim=-1)
            gold_key = index2key[gold_index.item()]

        return scores, gold_index, gold_key



    def parse_step_mh4(self,pending, hypergraph,arcs,gold_arc,words, all_possible_arcs):
        pending = hypergraph.calculate_pending()

        possible_arcs, items = self.possible_arcs_mh4(pending, hypergraph, arcs)
        #scores, gold_index, gold_key = self.score_arcs_mh4(possible_arcs,
        #                                                   gold_arc, items, words)
        scores, made_arc, gold_index = self.score_arcs_biaffine(possible_arcs,
                                                                gold_arc, words, all_possible_arcs)
        #gind = gold_index.item()
        # print_green(made_arc)
        #made_arc = possible_arcs[gind]
        #if self.training:
        #    gold_arc_set.remove(made_arc)
        h = made_arc[0]
        m = made_arc[1]
        if self.training:
            h = h.item()
            m = m.item()
        h = min(h,hypergraph.n-1)
        m = min(m,hypergraph.n-1)
        #pending = hypergraph.calculate_pending(pending, m)
        hypergraph = hypergraph.set_head(m)
        #hypergraph.made_arcs.append(made_arc)
        #print_green(h)
        #print_blue(m)
        arcs.append((h,m))

        return scores, gold_index, h, m, arcs, hypergraph, pending

    def forward(self, x, transitions, relations, map, heads, rels):
        x_ = x[0][:, 1:]
        # average of last 4 hidden layers

        #with torch.no_grad():
        out = self.bert(x_.to(device=constants.device))[2]
        # take the average of all the levels
        x_emb = torch.stack(out).mean(0)

        heads_batch = torch.ones((x_emb.shape[0], heads.shape[1])).to(device=constants.device)  # * -1
        prob_sum = 0
        batch_loss = 0
        x_mapped = torch.zeros((x_emb.shape[0], heads.shape[1] + 1, x_emb.shape[2] + 100)).to(device=constants.device)
        eos_emb = x_emb[0, -1, :].unsqueeze(0).to(device=constants.device)
        eos_emb = torch.cat([eos_emb, torch.zeros((1, 100)).to(device=constants.device)], dim=-1).to(
            device=constants.device)
        # think about vectorizing this to improve speed
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
        #h_t, _ = self.run_lstm(x_mapped, sent_lens)
        #print_yellow(h_t.shape)
        #h_t_noeos = torch.zeros((h_t.shape[0], heads.shape[1], h_t.shape[2])).to(device=constants.device)
        # print_blue(h_t_noeos.shape)
        # print_green(h_t_noeos)
        #for i in range(h_t.shape[0]):
        for i in range(x_mapped.shape[0]):

            n = int(sent_lens[i] - 1)
            ordered_arcs = transitions[i]
            mask = (ordered_arcs.sum(dim=1) != -2)
            ordered_arcs = ordered_arcs[mask, :]
            all_possible_arcs = set(itertools.product(range(n), range(n)))

            #s = h_t[i, :n + 1, :]
            #s_b = bh_t[i, :n + 1, :]

            s = x_mapped[i, :n+1, :]
            # trees = torch.exp(curr_init_weights)
            arcs = []
            history = defaultdict(lambda: 0)
            loss = 0
            popped = []

            right_children = {i: [i] for i in range(n)}
            left_children = {i: [i] for i in range(n)}
            #words_f = s  # .clone()
            #words_b = s_b  # .clone()
            #gold_arc_set = self.gold_arc_set(ordered_arcs)
            hypergraph = self.hypergraph(n)
            #hypergraph = self.hypergraph(n, gold_arc_set)
            #if self.mode == "agenda-mh4":
            #    pending = hypergraph.calculate_pending_init()
            #else:
            #    pending = self.init_pending(n, hypergraph)
            pending = self.init_pending(n, hypergraph)
            for iter, gold_arc in enumerate(ordered_arcs):

                #scores, gold_index, h, m, arcs, hypergraph,pending = self.parse_step_chart(pending,
                #                                                                            hypergraph,
                #                                                                            arcs,
                #                                                                            gold_arc,
                #                                                                            words_f,
                #                                                                            words_b)

                scores, gold_index, h, m, arcs, hypergraph, pending = self.parse_step_chart(pending,
                                                                                            hypergraph,
                                                                                            arcs,
                                                                                            gold_arc,
                                                                                            s, all_possible_arcs)
                if self.training:
                    loss += nn.CrossEntropyLoss(reduction='sum')(scores.unsqueeze(0), gold_index.reshape(1))
                s_new = s.clone().detach()
                reprs = torch.cat([s[h,:], s[m,:]],
                                  dim=-1)
                #print_blue(self.training)

                s_new[h, :] = nn.Tanh()(self.linear_tree(reprs))
                s = s_new.clone()
            loss /= len(ordered_arcs)
            pred_heads = self.heads_from_arcs(arcs, n)
            heads_batch[i, :n] = pred_heads
            #h_t_noeos[i, :n, :] = h_t[i, :n, :]
            batch_loss += loss
        batch_loss /= x_emb.shape[0]
        heads = heads_batch
        l_logits = self.get_label_logits(x_mapped[:,:-1,:], heads)
        rels_batch = torch.argmax(l_logits, dim=-1)
        batch_loss += self.loss(batch_loss, l_logits, rels)
        return batch_loss, heads_batch, rels_batch

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