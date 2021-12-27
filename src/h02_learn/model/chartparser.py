import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import constants
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .base import BertParser
from .modules import Item
from .modules import Bilinear
from .hypergraph import MH4
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
    def __init__(self, lang, num_rels, batch_size, hypergraph=MH4,
                 dropout=0.33,is_easy_first=True):
        super().__init__(lang, num_rels,
                         batch_size=batch_size, dropout=dropout)
        #self.eos_token_id = eos_token_id
        self.hypergraph = hypergraph
        self.is_easy_first = is_easy_first
        self.dropout = nn.Dropout(dropout)
        self.parse_step_chart = self.parse_step_mh4
        print(self.num_rels)
        bert_hidden_size = 768#1024 #768
        self.hidden_size = bert_hidden_size

        linear_items1 = nn.Linear(bert_hidden_size * 3, bert_hidden_size * 2).to(device=constants.device)
        linear_items2 = nn.Linear(bert_hidden_size * 2, bert_hidden_size).to(device=constants.device)
        linear_items3 = nn.Linear(bert_hidden_size, 500).to(device=constants.device)
        linear_items4 = nn.Linear(500, 1).to(device=constants.device)

        #layers = [linear_items1, nn.ReLU(), nn.Dropout(dropout), linear_items2, nn.ReLU(), nn.Dropout(dropout),
        #          linear_items3, nn.ReLU(), nn.Dropout(dropout), linear_items4]

        layers = [linear_items1, nn.ReLU(), linear_items2, nn.ReLU(),
                  linear_items3, nn.ReLU(), linear_items4]

        self.mlp = nn.Sequential(*layers)

        #self.linear_labels_dep = nn.Linear(bert_hidden_size, 500).to(device=constants.device)
        #self.linear_labels_head = nn.Linear(bert_hidden_size, 500).to(device=constants.device)
        #self.bilinear_label = Bilinear(500, 500, self.num_rels)

        label_linear = nn.Linear(bert_hidden_size, self.num_rels).to(device=constants.device)
        layers_label = [label_linear,nn.Tanh()]
        self.label_predictor = nn.Sequential(*layers_label)

    def init_pending(self, n, hypergraph):
        pending = {}
        for i in range(n):
            item = hypergraph.axiom(i)
            pending[item.key] = item
        return pending

    def reduce_linear(self, x, sent_lens):
        x_in = pack_padded_sequence(x, sent_lens.to(device=torch.device("cpu")), batch_first=True,
                                    enforce_sorted=False)
        x_out = self.squash_packed(x_in, self.linear_reduce)
        h_t = pad_packed_sequence(x_out, batch_first=True)[0]

        return h_t

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

    def span_rep(self, words, i, j, n):
        j = min(j, len(words) - 1)
        sij = words[j, :] - words[max(i - 1, 0), :]
        return sij

    def init_arc_list(self, tensor_list, oracle_agenda):
        item_list = {}
        for t in tensor_list:
            item = oracle_agenda[(t[0].item(), t[1].item(), t[2].item())]
            item_list[(item.i, item.j, item.h)] = item
        return item_list

    def gold_arc_set(self, ordered_arcs):
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

    def possible_arcs_mh4(self, pending, hypergraph, prev_arcs):
        arcs = []
        all_items = []

        arcs_new, all_items_new, _ = hypergraph.iterate_spans(None, pending, merge=False, prev_arc=prev_arcs)
        for (pa, pi) in zip(arcs_new, all_items_new):
            if pa not in prev_arcs:
                arcs.append(pa)
                all_items.append(pi)

        return arcs, all_items

    def score_arcs_mh4(self, possible_arcs, gold_arc, possible_items, words, hypergraph):
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
        for iter, ((u, v), item) in enumerate(zip(possible_arcs, possible_items)):

            if len(item.heads) > 1:
                span = torch.mean(words[item.heads[0]:item.heads[-1], :], 0)
                span = span.reshape(1, self.hidden_size)
            else:
                span = words[item.heads[0], :]
                span = span.reshape(1, self.hidden_size)

            fwd_rep = torch.cat([words[u, :], words[v, :]], dim=-1).unsqueeze(0)
            rep = torch.cat([span, fwd_rep], dim=-1)
            #item.set_rep(rep)
            #hypergraph.add_item(item)
            s = self.mlp(rep)
            scores.append(s)
            index2key[iter] = item.key

        scores = torch.stack(scores, dim=-1).squeeze(0)

        if not self.training or gold_index is None:
            gold_index = torch.argmax(scores, dim=-1)
            gold_key = index2key[gold_index.item()]

        return scores, gold_index, gold_key

    def parse_step_mh4(self, pending, hypergraph, arcs, gold_arc, words):
        pending = hypergraph.calculate_pending()

        possible_arcs, items = self.possible_arcs_mh4(pending, hypergraph, arcs)
        scores, gold_index, gold_key = self.score_arcs_mh4(possible_arcs,
                                                           gold_arc, items, words, hypergraph)
        gind = gold_index.item()

        made_arc = possible_arcs[gind]
        # if self.training:
        #    gold_arc_set.remove(made_arc)
        h = made_arc[0]
        m = made_arc[1]
        h = min(h, hypergraph.n - 1)
        m = min(m, hypergraph.n - 1)
        hypergraph = hypergraph.set_head(m)
        arcs.append(made_arc)
        return scores, gold_index, h, m, arcs, hypergraph, pending

    def forward(self, x, transitions, relations, map, heads, rels):
        x_ = x[0][:, 1:]
        #print("kir")
        #out = self.bert(x_.to(device=constants.device))[2]
        #print("hafti")
        #print(x_)
        out = self.bert(x_.to(device=constants.device)).hidden_states  # .logits
        #print("xatar")
        #print(out)

        # x_emb = torch.stack(out[-4:]).mean(0)
        x_emb = torch.stack(out[-8:]).mean(0)
        heads_batch = torch.ones((x_emb.shape[0], heads.shape[1])).to(device=constants.device)  # * -1
        batch_loss = 0
        x_mapped = torch.zeros((x_emb.shape[0], heads.shape[1] + 1, x_emb.shape[2])).to(device=constants.device)
        eos_emb = x_emb[0, -1, :].unsqueeze(0).to(device=constants.device)

        for i, sentence in enumerate(x_emb):
            # initialize a parser
            mapping = map[i, map[i] != -1]
            sentence = sentence[:-1, :]
            s = self.get_bert_embeddings(mapping, sentence, None)
            s = torch.cat([s, eos_emb], dim=0)
            curr_sentence_length = s.shape[0]
            x_mapped[i, :curr_sentence_length, :] = s

        sent_lens = (x_mapped[:, :, 0] != 0).sum(-1).to(device=constants.device)

        h_t_noeos = torch.zeros((x_mapped.shape[0], heads.shape[1], x_mapped.shape[2])).to(device=constants.device)
        for i in range(x_mapped.shape[0]):
            n = int(sent_lens[i] - 1)
            ordered_arcs = transitions[i]
            mask = (ordered_arcs.sum(dim=1) != -2)
            ordered_arcs = ordered_arcs[mask, :]

            s = x_mapped[i, :n + 1, :]

            arcs = []
            loss = 0

            words = s  # .clone()

            #hypergraph = self.hypergraph(n)
            hypergraph = self.hypergraph(n, self.is_easy_first)

            pending = self.init_pending(n, hypergraph)
            for iter, gold_arc in enumerate(ordered_arcs):

                scores, gold_index, h, m, arcs, hypergraph, pending = self.parse_step_chart(pending,
                                                                                            hypergraph,
                                                                                            arcs,
                                                                                            gold_arc,
                                                                                            words)

                if self.training:
                     loss += nn.CrossEntropyLoss(reduction='sum')(scores, gold_index)
                     #print(loss)
            loss /= len(ordered_arcs)
            pred_heads = self.heads_from_arcs(arcs, n)
            heads_batch[i, :n] = pred_heads
            h_t_noeos[i, :n, :] = x_mapped[i, :n, :].clone()
            batch_loss += loss

        batch_loss /= x_emb.shape[0]
        #heads = heads_batch
        l_logits = nn.Softmax(dim=-1)(self.label_predictor(h_t_noeos))
        #print(l_logits)
        #print(l_logits.shape)
        #l_logits = self.get_label_logits(h_t_noeos, heads)
        rels_batch = torch.argmax(l_logits, dim=-1)
        batch_loss = self.loss(batch_loss, l_logits, rels)
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
            heads[min(v, sent_len - 1)] = u  # .item()
        return torch.tensor(heads).to(device=constants.device)

    def loss(self, batch_loss, l_logits, rels):
        criterion_l = nn.CrossEntropyLoss().to(device=constants.device)
        #print(l_logits)
        #print(rels)
        #print(l_logits.shape)
        #print(rels.shape)
        l_logits = l_logits[rels != -1]
        rels = rels[rels != -1]
        #print(l_logits)
        #print(rels)
        #print(l_logits.shape)
        #print(rels.shape)
        #print("(///(((((((//(Z/(/(/(/(/(/(/")
        #print(rels)
        #print(batch_loss)
        #print("(///(((((((//(Z/(/(/(/(/(/(/")
        loss = criterion_l(l_logits.reshape(-1, l_logits.shape[-1]), rels.reshape(-1))

        return loss + batch_loss

    def get_args(self):
        return {
            'hidden_size': self.hidden_size,
            'hypergraph': self.hypergraph,
            'dropout': self.dropout_prob,
            'batch_size': self.batch_size,
        }
