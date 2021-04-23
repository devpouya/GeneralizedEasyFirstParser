import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import constants
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .base import BertParser
from ..algorithm.transition_parsers import ShiftReduceParser
from .modules import StackCell, SoftmaxActions, PendingRNN, Agenda, Chart, Item
from .modules import Biaffine, Bilinear, BiaffineChart
from .hypergraph import LazyArcStandard, LazyArcEager, LazyHybrid, LazyMH4
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
    def __init__(self, vocabs, embedding_size, rel_embedding_size, batch_size, hypergraph,
                 dropout=0.33, beam_size=10, max_sent_len=190, easy_first=False):
        super().__init__(vocabs, embedding_size, rel_embedding_size, batch_size, dropout=dropout,
                         beam_size=beam_size)
        hidden_size = 200
        self.hypergraph = hypergraph
        weight_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=8)
        self.weight_encoder = nn.TransformerEncoder(weight_encoder_layer, num_layers=2)
        self.prune = True  # easy_first
        self.dropout = nn.Dropout(dropout)
        layers = []
        # self.mlp = nn.Linear(500 * 3, 1)
        l1 = nn.Linear(hidden_size * 6, hidden_size)
        l11 = nn.Linear(hidden_size * 6, hidden_size)
        l2 = nn.Linear(hidden_size, 1)
        l22 = nn.Linear(hidden_size, 1)
        layers = [l1, nn.ReLU(), l2, nn.Sigmoid()]
        layers2 = [l11, nn.ReLU(), l22, nn.Sigmoid()]
        self.mlp = nn.Sequential(*layers)
        self.mlp2 = nn.Sequential(*layers2)

        self.linear_tree = nn.Linear(hidden_size * 2, hidden_size)
        self.linear_label = nn.Linear(hidden_size * 2, self.rel_embedding_size)
        self.max_size = max_sent_len
        self.linear_dep = nn.Linear(hidden_size, 100).to(device=constants.device)
        self.linear_h11 = nn.Linear(hidden_size, 100).to(device=constants.device)
        self.linear_h12 = nn.Linear(hidden_size, 100).to(device=constants.device)
        self.linear_h21 = nn.Linear(hidden_size, 100).to(device=constants.device)
        self.linear_h22 = nn.Linear(hidden_size, 100).to(device=constants.device)
        self.linear_head = nn.Linear(hidden_size, 100).to(device=constants.device)
        self.biaffine_item = Biaffine(100, 100)
        self.biaffine = Biaffine(100, 100)
        self.biaffine_h = Biaffine(100, 100)
        self.biaffineChart = BiaffineChart(100, 100)

        self.linear_labels_dep = nn.Linear(hidden_size, 100).to(device=constants.device)
        self.linear_labels_head = nn.Linear(hidden_size, 100).to(device=constants.device)
        self.bilinear_label = Bilinear(100, 100, self.num_rels)

        self.weight_matrix = nn.MultiheadAttention(868, num_heads=1, dropout=dropout).to(device=constants.device)
        self.root_selector = nn.LSTM(
            868, 1, 1, dropout=(dropout if 1 > 1 else 0),
            batch_first=True, bidirectional=False).to(device=constants.device)

        self.lstm = nn.LSTM(868, hidden_size, 2, batch_first=True, bidirectional=False).to(device=constants.device)
        self.lstm_tree = nn.LSTM(hidden_size, hidden_size, 1, batch_first=False, bidirectional=False).to(
            device=constants.device)
        self.compressor = nn.LSTM(hidden_size, 1, 1, batch_first=False, bidirectional=False).to(
            device=constants.device)

        input_init = torch.zeros((1, hidden_size*3)).to(
            device=constants.device)
        hidden_init = torch.zeros((1, hidden_size*3)).to(
            device=constants.device)
        self.empty_initial = nn.Parameter(torch.zeros(1, hidden_size*3)).to(device=constants.device)

        self.lstm_init_state = (nn.init.xavier_uniform_(input_init), nn.init.xavier_uniform_(hidden_init))
        self.stack_lstm = nn.LSTMCell(hidden_size*3, hidden_size*3).to(device=constants.device)

        self.item_lstm = StackCell(self.stack_lstm, self.lstm_init_state, self.lstm_init_state, self.dropout,
                           self.empty_initial)

    def init_pending(self, n):
        pending = {}
        for i in range(n):
            k, j, h = i, i + 1, i
            pending[(k, j, h)] = Item(k, j, h, k, k)
            # pending.append(Item(k, j, h, k, k))
        return pending

    def run_lstm(self, x, sent_lens):
        lstm_in = pack_padded_sequence(x, sent_lens.to(device=torch.device("cpu")), batch_first=True,
                                       enforce_sorted=False)
        # ###print(lstm_in)
        lstm_out, _ = self.lstm(lstm_in)
        h_t, _ = pad_packed_sequence(lstm_out, batch_first=True)
        h_t = self.dropout(h_t).contiguous()
        return h_t

    def tree_representation(self, head, modifier, label):
        reprs = torch.cat([head, modifier, label],
                          dim=-1)

        c = nn.Tanh()(self.linear_tree(reprs))
        return c

    def window(self, i, n):
        if i >= n:
            return [n - 1, None, None, None]
        elif i - 1 >= 0 and i + 1 < n and i + 2 < n:
            return [i - 1, i, i + 1, i + 2]
        elif i - 1 < 0 and i + 1 < n and i + 2 < n:
            return [None, i, i + 1, i + 2]
        elif i - 1 >= 0 and i + 1 < n < i + 2:
            return [i - 1, i, i + 1, None]
        elif i - 1 >= 0 and i + 1 > n:
            return [i - 1, i, None, None]
        else:
            return [None, i, None, None]

    def window2(self, i, n):

        if i >= n:
            i = n - 2
        if i - 1 >= 0 and i + 2 < n:
            return [i - 1, i, i + 1, i + 2]
        elif i - 1 < 0 and i + 2 < n:
            return [None, i, i + 1, i + 2]
        elif i - 1 >= 0 and i + 2 > n:
            return [i - 1, i, i + 1, None]
        else:
            return [None, i, i + 1, None]

    def possible_arcs_simple(self, words, remaining, oracle_arc):
        scores = []
        z = torch.zeros_like(words[0, :]).to(device=constants.device)
        n = len(remaining)
        arcs = []
        oracle_ind = 0
        left = oracle_arc[0]
        right = oracle_arc[1]
        derived = oracle_arc[2]
        # print(colored("oracle {}".format(di),"yellow"))
        if derived[2] != right[2]:
            # right is child and is popped
            gold = (derived[2], right[2])
        else:
            gold = (derived[2], left[2])
        for i in range(n - 1):
            window = self.window(i, n)
            rep = torch.cat([words[i, :] if i is not None else z for i in window], dim=-1).to(device=constants.device)
            pair1 = (remaining[i], remaining[i + 1])
            pair2 = (remaining[i + 1], remaining[i])
            if pair1 == gold:
                oracle_ind = i
            if pair2 == gold:
                oracle_ind = i + 1
            score = self.mlp2(rep)
            # print_red(score)
            scores.append(score[0])
            scores.append(score[1])
            arcs.append((remaining[i], remaining[i + 1]))
            arcs.append((remaining[i + 1], remaining[i]))
        # scores = torch.stack(scores,dim=-1)
        scores = torch.tensor(scores).to(device=constants.device)
        best_score_ind = torch.argmax(scores, dim=0)
        return scores, torch.tensor([oracle_ind], dtype=torch.long).to(device=constants.device), arcs[best_score_ind]

    def take_action_simple(self, predicted_arc, oracle_arc, remaining):
        if self.training:
            left = oracle_arc[0]
            right = oracle_arc[1]
            derived = oracle_arc[2]
            # print(colored("oracle {}".format(di),"yellow"))
            if derived[2] != right[2]:
                # right is child and is popped
                made_arc = (derived[2], right[2])
            else:
                made_arc = (derived[2], left[2])

            remaining.remove(made_arc[1])

            return remaining, made_arc
        else:
            remaining.remove(predicted_arc[1])
            return remaining, (torch.tensor(predicted_arc[0]), torch.tensor(predicted_arc[1]))

    def pick_next(self, words, pending, hypergraph, oracle_item):
        scores = []
        gold_index = None
        all_embedding = self.item_lstm.embedding().squeeze(0)
        for iter, item in enumerate(pending.values()):
            if item.l in hypergraph.bucket or item.r in hypergraph.bucket:
                # print_blue("PRUNE")
                continue
            i, j, h = item.i, item.j, item.h
            if i == oracle_item[0] and j == oracle_item[1] and h == oracle_item[2]:
                gold_index = torch.tensor([iter], dtype=torch.long).to(device=constants.device)
            if item in hypergraph.scored_items:
                score = item.score
            else:
                if j >= len(words):
                    j = len(words) - 1

                features_derived = torch.cat([words[i, :], words[j, :], words[h, :]], dim=-1).to(device=constants.device)
                features_derived = torch.cat([features_derived, all_embedding], dim=-1)
                score = self.mlp(features_derived)
                item.update_score(score)
                hypergraph.score_item(item)
            scores.append(score)
        scores = torch.stack(scores).permute(1, 0)
        # scores = torch.tensor(scores).to(device=constants.device).unsqueeze(0)
        winner = torch.argmax(scores, dim=-1)
        winner_item = list(pending.values())[winner]
        # print_green(winner_item)
        # print_red(list(pending.values())[gold_index])
        gold_next_item = None
        if gold_index is not None:
            gold_next_item = list(pending.values())[gold_index]
        return scores, winner_item, gold_index, hypergraph, gold_next_item

    def predict_next(self, x, possible_items, hypergraph, oracle_agenda, list_possible_next):
        n = len(x)
        z = torch.zeros_like(x[0, :]).to(device=constants.device)
        scores = []
        all_embedding = self.item_lstm.embedding().squeeze(0)
        for item in possible_items:
            i, j, h = item.i, item.j, item.h
            if item in hypergraph.scored_items:
                score = item.score
            else:
                if j >= len(x):
                    j = len(x) - 1
                features_derived = torch.cat([x[i, :], x[j, :], x[h, :]], dim=-1).to(device=constants.device)
                features_derived = torch.cat([features_derived,all_embedding],dim=-1)
                score = self.mlp(features_derived)  # *l_score*r_score
                item.update_score(score)
                hypergraph.score_item(item)
            scores.append(score)
        scores = torch.stack(scores).permute(1, 0)

        winner = torch.argmax(scores, dim=-1)
        if self.training:
            next_item = None
            gold_index = None
            for i, item in enumerate(possible_items):
                if (item.i, item.j, item.h) in list_possible_next.keys():
                    gold_index = torch.tensor([i], dtype=torch.long).to(device=constants.device)
                    next_item = list_possible_next[(item.i, item.j, item.h)]
                    break
        else:
            gold_index = None
            next_item = possible_items[winner]
        return next_item, hypergraph, scores, gold_index

    def take_step(self, x, transitions, gold_next_item, hypergraph, oracle_agenda, pred_item, pending):
        if self.training:

            key = (gold_next_item.i, gold_next_item.j, gold_next_item.h)
            di = gold_next_item
            # oracle_agenda[(derived[0].item(), derived[1].item(), derived[2].item())]

        else:
            di = pred_item
            key = (pred_item.i, pred_item.j, pred_item.h)

        made_arc = None
        del pending[key]

        if isinstance(di.l, Item):
            h = di.h
            m = di.l.h if di.l.h != h else di.r.h
            made_arc = (h, m)
        if di.l in hypergraph.bucket or di.r in hypergraph.bucket:
            h = di.h
            m = di.l.h if di.l.h != h else di.r.h
            made_arc = (h, m)
            scores, gold_index = None, None
        else:
            hypergraph = hypergraph.update_chart(di)
            hypergraph = hypergraph.add_bucket(di)
            possible_items = hypergraph.outgoing(di)
            if di.j >= len(x):
                j = len(x)-1
            else:
                j = di.j
            rep = torch.cat([x[di.i,:],x[j,:],x[di.h,:]],dim=-1).unsqueeze(0).to(device=constants.device)

            self.item_lstm.push(rep)
            if len(possible_items) > 0:

                new_item, hypergraph, scores, gold_index = self.predict_next(x, possible_items, hypergraph,
                                                                             oracle_agenda, transitions)
                if new_item is not None:
                    pending[(new_item.i, new_item.j, new_item.h)] = new_item
            else:
                scores, gold_index = None, None
                pass

        return hypergraph, pending, di, made_arc, scores, gold_index

    def init_arc_list(self, tensor_list, oracle_agenda):
        item_list = {}
        for t in tensor_list:
            item = oracle_agenda[(t[0].item(), t[1].item(), t[2].item())]
            item_list[(item.i, item.j, item.h)] = item
        return item_list

    def forward(self, x, transitions, relations, map, heads, rels):

        # average of last 4 hidden layers
        with torch.no_grad():
            out = self.bert(x[0].to(device=constants.device))[2]
            x_emb = torch.stack(out[-8:]).mean(0)

        heads_batch = torch.ones((x_emb.shape[0], heads.shape[1])).to(device=constants.device)  # * -1

        prob_sum = 0
        batch_loss = 0
        x_mapped = torch.zeros((x_emb.shape[0], heads.shape[1], x_emb.shape[2] + 100)).to(device=constants.device)
        for i, sentence in enumerate(x_emb):
            # initialize a parser
            mapping = map[i, map[i] != -1]
            tag = self.tag_embeddings(x[1][i][x[1][i] != 0].to(device=constants.device))

            s = self.get_bert_embeddings(mapping, sentence, tag)
            curr_sentence_length = s.shape[0]
            x_mapped[i, :curr_sentence_length, :] = s

        sent_lens = (x_mapped[:, :, 0] != 0).sum(-1).to(device=constants.device)
        max_len = torch.max(sent_lens)
        h_t = self.run_lstm(x_mapped, sent_lens)
        # initial_weights_logits = self.get_head_logits(h_t, sent_lens)

        tree_loss = 0
        for i in range(h_t.shape[0]):

            curr_sentence_length = sent_lens[i]

            # curr_init_weights = initial_weights_logits[i]
            # curr_init_weights = curr_init_weights[:curr_sentence_length + 1, :curr_sentence_length + 1]
            # curr_init_weights = torch.exp(curr_init_weights)

            oracle_hypergraph = transitions[i]
            oracle_hypergraph = oracle_hypergraph[oracle_hypergraph[:, 0, 0] != -1, :, :]
            oracle_agenda = self.init_agenda_oracle(oracle_hypergraph)

            s = h_t[i, :curr_sentence_length, :]
            s_ind = list(range(curr_sentence_length))
            chart = Chart()
            pending = self.init_pending(curr_sentence_length)
            hypergraph = self.hypergraph(curr_sentence_length, chart)

            # trees = torch.exp(curr_init_weights)
            arcs = []
            history = defaultdict(lambda: 0)
            loss = 0
            popped = []

            # 1. compute tree
            # gold_tree = self.compute_tree(s, heads[i, :curr_sentence_length], rels[i, :curr_sentence_length])
            # s_wrong = s.clone().detach()

            right_children = {i: [i] for i in range(curr_sentence_length)}
            left_children = {i: [i] for i in range(curr_sentence_length)}
            remaining = list(range(curr_sentence_length))
            current_representations = s.clone()

            oracle_hypergraph_picks = oracle_hypergraph[:, -1, :].clone()
            list_oracle_hypergraph_picks = [t for t in oracle_hypergraph_picks]
            # print_red(list_oracle_hypergraph_picks)
            """
                convert this to a list of actual item types
                use that to do thing
            """
            # oracle_hypergraph_picks[:,-2,:] = torch.zeros_like(oracle_hypergraph_picks[:,0,:]).to(device=constants.device)
            oracle_transition_picks = oracle_hypergraph[:, :-1, :].clone()

            dim1 = int(oracle_transition_picks.shape[0] * oracle_transition_picks.shape[1])
            dim2 = int(oracle_transition_picks.shape[2])

            oracle_transition_picks = oracle_transition_picks.view(dim1, dim2)
            # oracle_hypergraph_picks = oracle_hypergraph_picks.view(dim1,dim2)
            # print_red(oracle_hypergraph_picks)
            # print_green(oracle_transition_picks)
            arc_list = self.init_arc_list(list_oracle_hypergraph_picks, oracle_agenda)
            for step in range(len(oracle_transition_picks)):

                # if arc_index_aux != 2:
                scores, item_to_make, gold_index, hypergraph, gold_next_item = self.pick_next(current_representations,
                                                                                             pending,
                                                                                             hypergraph,
                                                                                             oracle_transition_picks[
                                                                                                 step])
                #scores, item_to_make, gold_index, gold_next_item = self.get_item_logits(current_representations,
                #                                                                        pending,
                #                                                                        hypergraph,
                #                                                                        oracle_transition_picks[step])

                hypergraph, pending, made_item, made_arc, scores_hg, gold_index_hg = self.take_step(
                                                                                                current_representations,
                                                                                                arc_list,
                                                                                                gold_next_item,
                                                                                                hypergraph,
                                                                                                oracle_agenda,
                                                                                                item_to_make,
                                                                                                pending)
                if made_arc is not None:
                    h = made_arc[0]
                    m = made_arc[1]
                    arcs.append(made_arc)
                    if h < m:
                        # m is a right child
                        right_children[h].append(m)
                    else:
                        # m is a left child
                        left_children[h].append(m)
                    h_rep = self.tree_lstm(current_representations, left_children[h], right_children[h])
                    current_representations = current_representations.clone()
                    current_representations[h, :] = h_rep

                max_score = scores[:,torch.argmax(scores,dim=-1)]
                if self.training:
                    loss += 0.5 * nn.CrossEntropyLoss()(scores,
                                                        gold_index) +0.5*nn.ReLU()(1-scores[:,gold_index]+max_score)
                    if gold_index_hg is not None and scores_hg is not None:
                        max_score_hg = scores_hg[:,torch.argmax(scores_hg,dim=-1)]
                        loss += 0.5 * nn.CrossEntropyLoss()(scores_hg,
                                                            gold_index_hg) +0.5*nn.ReLU()(scores_hg[:,gold_index_hg]+max_score_hg)
            pred_heads = self.heads_from_arcs(arcs, curr_sentence_length)
            heads_batch[i, :curr_sentence_length] = pred_heads
            # if self.training:
            #    print_yellow(pred_heads)
            #    print_blue(heads[i])
            loss /= len(oracle_hypergraph)
            batch_loss += loss
            self.item_lstm.back_to_init()

        batch_loss /= x_emb.shape[0]
        # print_yellow(heads_batch)
        heads = heads_batch
        # tree_loss /= x_emb.shape[0]
        l_logits = self.get_label_logits(h_t, heads)
        rels_batch = torch.argmax(l_logits, dim=-1)
        # rels_batch = rels_batch.permute(1, 0)
        batch_loss += self.loss(batch_loss, l_logits, rels)
        # batch_loss += tree_loss
        return batch_loss, heads_batch, rels_batch

    def item_oracle_loss_single_step(self, scores, oracle_item):

        criterion_head = nn.CrossEntropyLoss().to(device=constants.device)
        criterion_mod = nn.CrossEntropyLoss().to(device=constants.device)

        left = oracle_item[0]
        right = oracle_item[1]
        derived = oracle_item[2]

        head = derived[2]
        mod = right[2] if right[2] != head else left[2]
        head_scores_cm = torch.sum(scores, dim=-1).unsqueeze(0).to(device=constants.device)
        # mod_scores_cm = torch.sum(scores,dim=0)
        h = torch.argmax(head_scores_cm)
        mod_scores = scores[:, h].unsqueeze(0).to(device=constants.device)

        mod_t = torch.zeros(1, dtype=torch.long).to(device=constants.device)
        mod_t[0] = mod
        head_t = torch.zeros(1, dtype=torch.long).to(device=constants.device)
        head_t[0] = head

        loss = criterion_head(head_scores_cm, head_t)
        loss += criterion_mod(mod_scores, mod_t)

        return loss
    def predict_next_biaffine(self, x, possible_items, hypergraph, oracle_agenda,list_possible_next):
        n = len(x)
        z = torch.zeros_like(x[0, :]).to(device=constants.device)
        scores = []
        h_11 = self.dropout(F.relu(self.linear_h11(x))).unsqueeze(0)
        #h_12 = self.dropout(F.relu(self.linear_h12(x))).unsqueeze(0)
        h_21 = self.dropout(F.relu(self.linear_h21(x))).unsqueeze(0)
        h,_ = self.compressor(x.unsqueeze(1))
        #print_red(h.shape)
        #h_22 = self.dropout(F.relu(self.linear_h22(x))).unsqueeze(0)
        #h_1 = torch.cat([h_11,h_12],dim=0)
        #h_2 = torch.cat([h_21,h_22],dim=0)
        h_logits = self.biaffine_h(h_11, h_21).squeeze(0)
        change = torch.kron(h_logits,h)
        dim = int(h_logits.shape[0])
        change = change.view(dim,dim,dim)
        mask = torch.zeros_like(change,dtype=torch.bool).to(device=constants.device)

        #print_red(change.shape)
        #print_red(h_logits.shape)
        #jj
        for item in possible_items:
            i, j, h = item.i, item.j, item.h

            if j >= len(x):
                j = len(x) - 1
            mask[i,j,h] = True

        scores = torch.masked_select(h_logits, mask).unsqueeze(0)
        winner = torch.argmax(scores, dim=-1)

        if self.training:
            next_item = None
            gold_index = None
            for i, item in enumerate(possible_items):
                if (item.i, item.j, item.h) in list_possible_next.keys():
                    gold_index = torch.tensor([i], dtype=torch.long).to(device=constants.device)
                    next_item = list_possible_next[(item.i, item.j, item.h)]
                    break
        else:
            gold_index = None
            next_item = possible_items[winner]
        return next_item, hypergraph, scores, gold_index

    def get_item_logits(self, s, pending, hypergraph, oracle_item):
        gold_index = None
        h_dep = self.dropout(F.relu(self.linear_dep(s)))
        h_arc = self.dropout(F.relu(self.linear_head(s)))

        h_logits = self.biaffine_item(h_arc.unsqueeze(0), h_dep.unsqueeze(0)).squeeze(0)
        scores = []
        mask = torch.zeros_like(h_logits,dtype=torch.bool).to(device=constants.device)
        n = h_logits.shape[0]
        for iter, item in enumerate(pending.values()):
            if item.l in hypergraph.bucket or item.r in hypergraph.bucket:
                continue
            i, j, h = item.i, item.j, item.h

            if i == oracle_item[0] and j == oracle_item[1] and h == oracle_item[2]:
                gold_index = torch.tensor([iter], dtype=torch.long).to(device=constants.device)
            if j >= n:
                j = n - 1
            mask[i,j] = True
            #scores.append(h_logits[i, j])
        scores = torch.masked_select(h_logits,mask).unsqueeze(0)
        winner = torch.argmax(scores, dim=-1)
        winner_item = list(pending.values())[winner]

        gold_next_item = None
        if gold_index is not None:
            gold_next_item = list(pending.values())[gold_index]

        return scores, winner_item, gold_index, gold_next_item

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

    def loss(self, batch_loss, l_logits, rels):
        criterion_l = nn.CrossEntropyLoss(ignore_index=0).to(device=constants.device)
        loss = criterion_l(l_logits.reshape(-1, l_logits.shape[-1]), rels.reshape(-1))
        return loss + batch_loss

    def init_agenda_oracle(self, oracle_hypergraph):
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

            pending[(derived[0].item(), derived[1].item(), derived[2].item())] = Item(derived[0].item(),
                                                                                      derived[1].item(),
                                                                                      derived[2].item(),
                                                                                      left_item, right_item)
        # ###print("cherry pies")
        # for item in pending.values():
        #    ###print(item)
        return pending

    def margin_loss_step(self, oracle_action, scores):
        # correct action is the oracle action for now
        left = oracle_action[0]
        right = oracle_action[1]
        derived = oracle_action[2]
        correct_head = derived[2]
        correct_mod = right[2] if right[2] != derived[2] else left[2]
        score_incorrect = torch.max(scores)
        score_correct = scores[correct_head, correct_mod]
        # score_correct = self.mlp(
        #    torch.cat([words[correct_head], words[correct_mod]], dim=-1).to(device=constants.device))

        return nn.ReLU()(1 - score_correct + torch.max(score_incorrect))

    def heads_from_arcs(self, arcs, sent_len):
        heads = [0] * sent_len
        for (u, v) in arcs:
            heads[v] = u  # .item()
        return torch.tensor(heads).to(device=constants.device)

    def post_order(self, root, arcs):
        data = []

        def recurse(node):
            if not node:
                return
            children = [v for (u, v) in arcs if u == node]
            for c in children:
                recurse(c)
            data.append(node)

        recurse(root)
        return data

    def compute_tree(self, x, heads, labels):
        arcs = []
        for i, elem in enumerate(heads):
            arcs.append((elem, i + 1))

        postorder = self.post_order(0, arcs)
        for node in postorder:
            children = [v for (u, v) in arcs if u == node]
            if len(children) == 0:
                continue
            tmp = x[node]
            for c in children:
                tmp = self.tree_representation(tmp, x[c], labels[node])
            x[node] = tmp

        return x

    def tree_lstm(self, x, left_children, right_children):
        # print_blue(left_children)
        # print_blue(right_children)
        left_reps = x[list(left_children), :].unsqueeze(1)
        right_reps = x[list(right_children), :]  # .unsqueeze(1)
        right_reps = torch.flip(right_reps, dims=[0, 1]).unsqueeze(1)
        # print_green(left_reps.shape)
        # print_green(right_reps.shape)
        _, (lh, _) = self.lstm_tree(left_reps)
        _, (rh, _) = self.lstm_tree(right_reps)
        # print_yellow(lh.shape)
        # print_yellow(rh.shape)
        c = torch.cat([lh, rh], dim=-1).to(device=constants.device)
        c = self.linear_tree(c)
        return c
