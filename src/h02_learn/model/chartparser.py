import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import constants
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .base import BertParser
from ..algorithm.transition_parsers import ShiftReduceParser
from .modules import StackCell, SoftmaxActions, PendingRNN, Agenda, Chart, Item, ItemMH4, ItemW
from .modules import Biaffine, Bilinear, LabelSmoothingLoss, TreeLayer
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
    def __init__(self, vocabs, embedding_size, rel_embedding_size, batch_size, hidden_size=100, hypergraph=ArcStandard,
                 language="en",
                 dropout=0.33, eos_token_id=28996, mode="agenda-std", transition_system=None):
        super().__init__(language, vocabs,
                         embedding_size=embedding_size, rel_embedding_size=rel_embedding_size,
                         batch_size=batch_size, dropout=dropout)
        self.eos_token_id = eos_token_id
        self.hidden_size = 400  # hidden_size
        self.hypergraph = hypergraph
        self.dropout = nn.Dropout(dropout)
        self.mode = mode
        self.parse_step_chart = self.parse_step_mh4

        self.linear_tree = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.linear_dep = nn.Linear(self.hidden_size, 200).to(device=constants.device)

        self.linear_head = nn.Linear(self.hidden_size, 200).to(device=constants.device)

        self.bilinear_item = Bilinear(200, 200, 1)

        linear_items1 = nn.Linear(400 * 4, 400 * 2).to(device=constants.device)
        linear_items2 = nn.Linear(400 * 2, 400).to(device=constants.device)
        linear_items3 = nn.Linear(400, 1).to(device=constants.device)

        layers = [linear_items1, nn.ReLU(), nn.Dropout(dropout), linear_items2, nn.ReLU(), nn.Dropout(dropout),
                  linear_items3]

        self.mlp = nn.Sequential(*layers)

        self.linear_labels_dep = nn.Linear(self.hidden_size, 200).to(device=constants.device)
        self.linear_labels_head = nn.Linear(self.hidden_size, 200).to(device=constants.device)

        self.linear_arc_dep = nn.Linear(400, 200).to(device=constants.device)
        self.linear_arc_head = nn.Linear(400, 200).to(device=constants.device)
        self.biaffine = Biaffine(200, 200)
        self.bilinear_label = Bilinear(200, 200, self.num_rels)
        self.linear_reduce = nn.Linear(868, 400).to(device=constants.device)

        self.tree_layer = TreeLayer(400)

    def init_pending(self, n, hypergraph):
        pending = {}
        for i in range(n):
            item = hypergraph.axiom(i)
            pending[item.key] = item
        return pending

    def squash_packed(self, x, fn=torch.tanh):
        return torch.nn.utils.rnn.PackedSequence(fn(x.data), x.batch_sizes,
                                                 x.sorted_indices, x.unsorted_indices)

    def reduce_linear(self, x, sent_lens):
        x_in = pack_padded_sequence(x, sent_lens.to(device=torch.device("cpu")), batch_first=True,
                                    enforce_sorted=False)
        x_out = self.squash_packed(x_in, self.linear_reduce)
        h_t = pad_packed_sequence(x_out, batch_first=True)[0]

        return h_t


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


    def possible_arcs_mh4(self, pending, hypergraph, prev_arcs):
        arcs = []
        arcs_new = hypergraph.iterate_spans(None, pending, merge=False, prev_arc=prev_arcs)
        for pa in arcs_new:
            if pa not in prev_arcs:
                arcs.append(pa)

        return arcs

    def score_arcs_mh4_no_items(self, possible_arcs, gold_arc, words):

        n = len(words)

        ga = (gold_arc[0].item(), gold_arc[1].item())

        h_dep = self.dropout(F.relu(self.linear_arc_dep(words)))
        h_arc = self.dropout(F.relu(self.linear_arc_head(words)))

        gold_index = None

        flat_ind_to_arc = {}
        keep_inds = torch.zeros((n, n)).to(device=constants.device)
        for (i, j) in possible_arcs:
            if (i, j) == ga:
                gold_index = torch.tensor([i*n+j]).to(device=constants.device).reshape(1)
            keep_inds[i,j] = 1
            flat_ind_to_arc[i * n + j] = (i, j)

        mask_ = ~keep_inds.eq(1)
        scores = self.biaffine(h_arc, h_dep, mask_)
        scores = nn.Softmax(dim=-1)(torch.flatten(scores)).unsqueeze(0)

        if not self.training or gold_index is None:
            gold_index = torch.argmax(scores, dim=-1).reshape(1)
            made_arc = flat_ind_to_arc[gold_index.item()]

        else:
            made_arc = gold_arc

        return scores, made_arc, gold_index

    def parse_step_mh4(self, hypergraph, arcs, gold_arc, words):
        pending = hypergraph.calculate_pending()

        possible_arcs = self.possible_arcs_mh4(pending, hypergraph, arcs)
        scores, made_arc, gold_index = self.score_arcs_mh4_no_items(possible_arcs,
                                                           gold_arc, words)

        h = made_arc[0]
        m = made_arc[1]
        h = min(h, hypergraph.n - 1)
        m = min(m, hypergraph.n - 1)
        hypergraph = hypergraph.set_head(m)
        arcs.append(made_arc)
        return scores, gold_index, h, m, arcs, hypergraph

    def forward(self, x, transitions, relations, map, heads, rels):
        x_ = x[0][:, 1:]
        # average of last 4 hidden layers
        #if not self.training:
        #    self.bert.eval()
        #else:
        #    self.bert.train()
        #with torch.no_grad():
        #    out = self.bert(x_.to(device=constants.device))[2]
        #    x_emb = torch.stack(out[-8:]).mean(0)
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
        # h_t, bh_t = self.run_lstm(x_mapped, sent_lens)
        h_t = self.reduce_linear(x_mapped, sent_lens)

        h_t_noeos = torch.zeros((h_t.shape[0], heads.shape[1], h_t.shape[2])).to(device=constants.device)
        for i in range(h_t.shape[0]):

            n = int(sent_lens[i] - 1)
            ordered_arcs = transitions[i]
            mask = (ordered_arcs.sum(dim=1) != -2)
            ordered_arcs = ordered_arcs[mask, :]

            s = h_t[i, :n + 1, :]

            arcs = []
            loss = 0

            words = s
            hypergraph = self.hypergraph(n)

            for iter, gold_arc in enumerate(ordered_arcs):
                scores, gold_index, h, m, arcs, hypergraph = self.parse_step_chart(hypergraph,
                                                                                   arcs,
                                                                                   gold_arc,
                                                                                   words)

                if self.training:
                    loss += nn.CrossEntropyLoss(reduction='sum')(scores, gold_index)
                words = self.tree_layer(words, h, m)
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
        criterion_l = nn.CrossEntropyLoss(ignore_index=0).to(device=constants.device)
        loss = criterion_l(l_logits.reshape(-1, l_logits.shape[-1]), rels.reshape(-1))
        return loss + batch_loss

    def get_args(self):
        return {
            'language': self.language,
            'hidden_size': self.hidden_size,
            'hypergraph': self.hypergraph,
            'vocabs': self.vocabs,
            'embedding_size': self.embedding_size,
            'rel_embedding_size': self.rel_embedding_size,
            'dropout': self.dropout_prob,
            'batch_size': self.batch_size,
        }
