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
                 dropout=0.33, beam_size=10, max_sent_len=190, easy_first=True, eos_token_id=28996):
        super().__init__(vocabs, embedding_size, rel_embedding_size, batch_size, dropout=dropout,
                         beam_size=beam_size)
        hidden_size = 400
        self.eos_token_id = eos_token_id
        self.hypergraph = hypergraph
        weight_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=8)
        self.weight_encoder = nn.TransformerEncoder(weight_encoder_layer, num_layers=2)
        self.prune = True  # easy_first
        self.dropout = nn.Dropout(dropout)
        layers = []
        # self.mlp = nn.Linear(500 * 3, 1)
        #l1 = nn.Linear(hidden_size * 6 + rel_embedding_size, 3 * hidden_size)
        #l1 = nn.Linear(hidden_size * 6 + rel_embedding_size, 3 * hidden_size)
        #l11 = nn.Linear(hidden_size * 6 + rel_embedding_size, 3 * hidden_size)
        #l11 = nn.Linear(hidden_size * 6 + rel_embedding_size, 3 * hidden_size)
        #l2 = nn.Linear(3 * hidden_size, hidden_size)
        #l2 = nn.Linear(3 * hidden_size, hidden_size)
        #lf = nn.Linear(hidden_size, 1)
        #lf = nn.Linear(hidden_size, 1)
        #l22 = nn.Linear(3 * hidden_size, hidden_size)
        #l22 = nn.Linear(3 * hidden_size, hidden_size)
        #lf2 = nn.Linear(hidden_size, 1)
        #lf2 = nn.Linear(hidden_size, 1)
        #l3 = nn.Linear(hidden_size * 3, hidden_size)
        #l3 = nn.Linear(hidden_size * 3, hidden_size)
        #l33 = nn.Linear(hidden_size, self.num_rels)
        #l33 = nn.Linear(hidden_size, self.num_rels)
        #layers = [l1, nn.ReLU(), l2, nn.ReLU(), lf, nn.Sigmoid()]
        #layers2 = [l11, nn.ReLU(), l22, nn.ReLU(), lf2, nn.Sigmoid()]
        #layers3 = [l3, nn.ReLU(), l33, nn.Softmax(dim=-1)]
        #self.ls = nn.Linear(200,1)
        #layers_small = [ls,nn.Sigmoid()]
        #self.mlp = nn.Sequential(*layers)
        #self.mlp2 = nn.Sequential(*layers2)
        #self.mlp_rel = nn.Sequential(*layers3)
        #self.mlp_small = nn.Sequential(*layers_small)
        self.linear_tree = nn.Linear(hidden_size * 2, hidden_size)
        #self.linear_label = nn.Linear(hidden_size * 2, self.rel_embedding_size)
        self.max_size = max_sent_len
        self.linear_dep = nn.Linear(hidden_size, 200).to(device=constants.device)
        #self.linear_h11 = nn.Linear(hidden_size, 200).to(device=constants.device)
        #self.linear_h12 = nn.Linear(hidden_size, 200).to(device=constants.device)
        #self.linear_h21 = nn.Linear(hidden_size, 200).to(device=constants.device)
        #self.linear_h22 = nn.Linear(hidden_size, 200).to(device=constants.device)
        self.linear_head = nn.Linear(hidden_size, 200).to(device=constants.device)
        self.biaffine_item = Biaffine(200, 200)
        self.biaffine = Biaffine(200, 200)
        #self.biaffine_h = Biaffine(200, 200)
        self.bilinear_item = Bilinear(200,200,1)
        self.linear_items1 = nn.Linear(hidden_size, 200).to(device=constants.device)
        self.linear_items2 = nn.Linear(hidden_size, 200).to(device=constants.device)
        #self.biaffineChart = BiaffineChart(200, 200)

        self.linear_labels_dep = nn.Linear(hidden_size, 200).to(device=constants.device)
        self.linear_labels_head = nn.Linear(hidden_size, 200).to(device=constants.device)
        self.bilinear_label = Bilinear(200, 200, self.num_rels)

        #self.weight_matrix = nn.MultiheadAttention(868, num_heads=1, dropout=dropout).to(device=constants.device)
        #self.root_selector = nn.LSTM(
        #    868, 1, 1, dropout=(dropout if 1 > 1 else 0),
        #    batch_first=True, bidirectional=False).to(device=constants.device)

        self.lstm = nn.LSTM(868, hidden_size, 2, batch_first=True, bidirectional=False).to(device=constants.device)
        #self.lstm_tree = nn.LSTM(hidden_size, hidden_size, 1, batch_first=False, bidirectional=False).to(
        #    device=constants.device)
        self.lstm_tree_left = nn.LSTM(hidden_size, hidden_size, 1, batch_first=False, bidirectional=False).to(
            device=constants.device)
        self.lstm_tree_right = nn.LSTM(hidden_size, hidden_size, 1, batch_first=False, bidirectional=False).to(
            device=constants.device)
        #self.compressor = nn.LSTM(hidden_size, 1, 1, batch_first=False, bidirectional=False).to(
        #    device=constants.device)

        #input_init = torch.zeros((1, hidden_size * 3)).to(
        #    device=constants.device)
        #hidden_init = torch.zeros((1, hidden_size * 3)).to(
        #    device=constants.device)
        #self.empty_initial = nn.Parameter(torch.zeros(1, hidden_size * 3)).to(device=constants.device)

        #self.lstm_init_state = (nn.init.xavier_uniform_(input_init), nn.init.xavier_uniform_(hidden_init))
        #self.stack_lstm = nn.LSTMCell(hidden_size * 3, hidden_size * 3).to(device=constants.device)

        #self.item_lstm = StackCell(self.stack_lstm, self.lstm_init_state, self.lstm_init_state, self.dropout,
        #                           self.empty_initial)

        #self.ij_score = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1)
        #self.ih_score = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1)
        #self.jh_score = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1)

        # with torch.no_grad():
        #    eos_tens = torch.tensor([self.eos_token_id]).unsqueeze(0).to(device=constants.device)
        #    out = self.bert(eos_tens)[2]
        #    eos_emb = torch.stack(out[-8:]).mean(0)
        # print(eos_emb.shape)
        # print(eos_emb)
        # kjsl

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

    def predict_next_prn(self, words, items, hypergraph, oracle_item, prune=True):
        scores = []
        gold_index = None
        next_item = None
        winner_item = None
        all_embedding = self.item_lstm.embedding().squeeze(0)
        rel_0 = torch.tensor([0], dtype=torch.long).to(device=constants.device)
        rel_dummy = self.rel_embeddings(rel_0).to(device=constants.device)
        zrel = torch.zeros_like(rel_dummy).to(device=constants.device)
        rel_loss = 0
        for iter, item in enumerate(items.values()):
            if prune:
                if item.l in hypergraph.bucket or item.r in hypergraph.bucket:
                    continue
            i, j, h = item.i, item.j, item.h
            if prune:
                if i == oracle_item[0] and j == oracle_item[1] and h == oracle_item[2]:
                    gold_index = torch.tensor([iter], dtype=torch.long).to(device=constants.device)
            if item in hypergraph.scored_items:
                score = item.score
            else:
                if j >= len(words):
                    j = len(words) - 1
                if h >= len(words):
                    h = len(words) - 1
                features_derived = torch.cat([words[i, :], words[j, :], words[h, :]], dim=-1).to(
                    device=constants.device)
                rel = item.rel
                if rel is not None:
                    rel_target = torch.tensor([rel], dtype=torch.long).to(device=constants.device)
                    rel_gold = self.rel_embeddings(rel_target).to(device=constants.device)
                    pred_rel = self.mlp_rel(features_derived)
                    pred_rel_ind = torch.argmax(pred_rel, dim=-1)
                    rel_pred = self.rel_embeddings(pred_rel_ind).to(device=constants.device)
                    rel_loss += nn.CrossEntropyLoss()(pred_rel.unsqueeze(0), rel_target)
                    if self.training:
                        rel_embed = rel_gold
                    else:
                        rel_embed = rel_pred
                else:
                    rel_embed = zrel
                    rel_loss += 0

                features_derived = torch.cat([features_derived, all_embedding, rel_embed.squeeze(0)], dim=-1)
                score = self.mlp(features_derived)
                item.update_score(score)
                hypergraph.score_item(item)
            scores.append(score)
            scores = torch.stack(scores).permute(1, 0)
            # scores = torch.tensor(scores).to(device=constants.device).unsqueeze(0)
            winner = torch.argmax(scores, dim=-1)
            if prune:
                winner_item = list(items.values())[winner]
                if gold_index is not None:
                    next_item = list(items.values())[gold_index]
            else:
                if self.training:

                    gold_index, next_item = hypergraph.return_gold_next(items)
                else:
                    gold_index = None
                    next_item = items[winner]

            return scores, winner_item, gold_index, hypergraph, next_item, rel_loss

    def pick_next(self, words,n, pending, hypergraph, oracle_item):
        scores = []

        gold_index = None
        #all_embedding = self.item_lstm.embedding().squeeze(0)
        #rel_0 = torch.tensor([0], dtype=torch.long).to(device=constants.device)
        #rel_dummy = self.rel_embeddings(rel_0).to(device=constants.device)
        #zrel = torch.zeros_like(rel_dummy).to(device=constants.device)
        rel_loss = 0
        items_tensor = torch.zeros((1,400)).to(device=constants.device)
        for iter, item in enumerate(pending.values()):
            if item.l in hypergraph.bucket or item.r in hypergraph.bucket:
                continue
            i, j, h = item.i, item.j, item.h
            if i == oracle_item[0] and j == oracle_item[1] and h == oracle_item[2]:
                gold_index = torch.tensor([iter], dtype=torch.long).to(device=constants.device)
            #if item in hypergraph.scored_items:
            #    score = torch.tensor(item.score).to(device=constants.device)
            #else:
            #if j >= len(words):
            #    j = len(words) - 1
            #features_derived = torch.cat([words[i, :], words[j, :], words[h, :]], dim=-1).to(
            #    device=constants.device)
            #rel = item.rel
            #if rel is not None:
            #    rel_target = torch.tensor([rel], dtype=torch.long).to(device=constants.device)
            #    rel_gold = self.rel_embeddings(rel_target).to(device=constants.device)
            #    pred_rel = self.mlp_rel(features_derived)
            #    pred_rel_ind = torch.argmax(pred_rel, dim=-1)
            #    rel_pred = self.rel_embeddings(pred_rel_ind).to(device=constants.device)
            #    rel_loss += nn.CrossEntropyLoss()(pred_rel.unsqueeze(0), rel_target)
            #    if self.training:
            #        rel_embed = rel_gold
            #    else:
            #        rel_embed = rel_pred
            #else:
            #    rel_embed = zrel
            #    rel_loss += 0
            #features_derived = torch.cat([features_derived, all_embedding, rel_embed.squeeze(0)], dim=-1)
            #features_derived = self.tree_rep(i,j,h,words)
            #score = self.mlp_small(features_derived)
            #if item not in hypergraph.repped_items:
            left_children = hypergraph.get_left_children_from(h, i)  # list(range(i)) + [h]
            right_children = hypergraph.get_right_children_until(h, j)  # [h] + list(range(j, h))
            features_derived = self.tree_lstm(words, left_children, right_children).squeeze(0)
            hypergraph = hypergraph.set_item_vec(features_derived,item)
            #else:
            #    features_derived = item.vector_rep
            if iter == 0:
                items_tensor = features_derived
            else:
                items_tensor = torch.cat([items_tensor,features_derived],dim=0)

            #score = self.mlp_small(features_derived)
            #item.update_score(score)
            #hypergraph.score_item(item)
            #scores.append(score)
        #scores = torch.stack(scores).squeeze(1).permute(1, 0)
        # scores = torch.tensor(scores).to(device=constants.device).unsqueeze(0)
        h1 = self.dropout(F.relu(self.linear_items1(items_tensor))).unsqueeze(0)
        h2 = self.dropout(F.relu(self.linear_items2(items_tensor))).unsqueeze(0)
        scores = self.bilinear_item(h1, h2).squeeze(0).permute(1,0)
        #scores = nn.Softmax(dim=-1)(torch.sum(item_logits,dim=0)).unsqueeze(0)
        #scores = nn.Softmax(dim=-1)(F.relu(self.ls(item_logits)))

        winner = torch.argmax(scores, dim=-1)
        winner_item = list(pending.values())[winner]
        # print_green(winner_item)
        # print_red(list(pending.values())[gold_index])
        gold_next_item = None
        if gold_index is not None:
            gold_next_item = list(pending.values())[gold_index]
        return scores, winner_item, gold_index, hypergraph, gold_next_item, rel_loss

    def attn_score_item(self, x,orig, items, hypergraph):
        n = x.shape[0]
        print_green(orig.shape)
        mask_ij = torch.empty((n, n)).fill_(float('-inf'))
        mask_ih = torch.empty((n, n)).fill_(float('-inf'))
        mask_jh = torch.empty((n, n)).fill_(float('-inf'))
        mask = torch.zeros((n,n,n), dtype=torch.bool).to(device=constants.device)

        for iter, item in enumerate(items.values()):
            if item.l in hypergraph.bucket or item.r in hypergraph.bucket:
                continue
            i, j, h = item.i, item.j, item.h
            mask_ij[i, j] = 0
            mask_ih[i, h] = 0
            mask_jh[j, h] = 0
            mask[i,j,h] = True
        # put original matrix as third input?
        x = x.unsqueeze(1)
        attn_ij = self.ij_score(x, x, x, attn_mask=mask_ij)[1].squeeze(0)
        attn_ih = self.ih_score(x, x, x, attn_mask=mask_ih)[1].squeeze(0)
        attn_jh = self.jh_score(x, x, x, attn_mask=mask_jh)[1].squeeze(0)
        scores = torch.zeros((n,n,n))
        for item in items.values():
            if item.l in hypergraph.bucket or item.r in hypergraph.bucket:
                continue
            i,j,h = item.i,item.j,item.h
            scores[i,j,h] = torch.exp(attn_ij[i,j])*attn_jh[j,h]*attn_ih[i,h]
        scores = torch.masked_select(scores, mask)
        return attn_jh

    def predict_next(self, x, possible_items, hypergraph):
        n = len(x)
        z = torch.zeros_like(x[0, :]).to(device=constants.device)
        items_tensor = torch.zeros((1,400)).to(device=constants.device)

        #all_embedding = self.item_lstm.embedding().squeeze(0)
        #rel_0 = torch.tensor([0], dtype=torch.long).to(device=constants.device)
        #rel_dummy = self.rel_embeddings(rel_0).to(device=constants.device)
        #zrel = torch.zeros_like(rel_dummy).to(device=constants.device)
        rel_loss = 0
        for iter, item in enumerate(possible_items):
            i, j, h = item.i, item.j, item.h
            # print_yellow((i,j,h))
            #if item in hypergraph.scored_items:
            #    score = item.score
            #else:
            #if j >= len(x):
            #    j = len(x) - 1
            #if h >= len(x):
            #    h = len(x) - 1
            #features_derived = torch.cat([x[i, :], x[j, :], x[h, :]], dim=-1).to(device=constants.device)
            #rel = item.rel
            #if rel is not None:
            #    rel_target = torch.tensor([rel], dtype=torch.long).to(device=constants.device)
            #    rel_gold = self.rel_embeddings(rel_target).to(device=constants.device)
            #    pred_rel = self.mlp_rel(features_derived)
            #    pred_rel_ind = torch.argmax(pred_rel, dim=-1)
            #    rel_pred = self.rel_embeddings(pred_rel_ind).to(device=constants.device)
            #    rel_loss += nn.CrossEntropyLoss()(pred_rel.unsqueeze(0), rel_target)
            #    if self.training:
            #        rel_embed = rel_gold
            #    else:
            #        rel_embed = rel_pred
            #else:
            #    rel_embed = zrel
            #    rel_loss += 0
            #features_derived = torch.cat([features_derived, all_embedding, rel_embed.squeeze(0)], dim=-1)
            #score = self.mlp2(features_derived)  # *l_score*r_score
            #item.update_score(score)
            #left_children = list(range(i))+[h]
            #right_children = [h]+list(range(j,h))
            #features_derived = self.tree_lstm(x, left_children, right_children).squeeze(0)
            ##features_derived = self.tree_rep(i, j, h, x)
            #score = self.mlp_small(features_derived)
            #hypergraph.score_item(item)
            #scores.append(score)
            #if item.vector_rep is None:
            #if item not in hypergraph.repped_items:
            left_children = hypergraph.get_left_children_from(h,i)#list(range(i)) + [h]
            right_children = hypergraph.get_right_children_until(h,j)#[h] + list(range(j, h))

            features_derived = self.tree_lstm(x, left_children, right_children).squeeze(0)
            hypergraph = hypergraph.set_item_vec(features_derived, item)
            #else:
            #    features_derived = item.vector_rep
            #else:
            #    features_derived = item.vector_rep
            if iter == 0:
                items_tensor = features_derived
            else:
                items_tensor = torch.cat([items_tensor,features_derived],dim=0)
        #scores = torch.stack(scores).squeeze(1).permute(1,0)
        h1 = self.dropout(F.relu(self.linear_items1(items_tensor))).unsqueeze(0)
        h2 = self.dropout(F.relu(self.linear_items2(items_tensor))).unsqueeze(0)
        scores = self.bilinear_item(h1, h2).squeeze(0).permute(1,0)
        #scores = nn.Softmax(dim=-1)(torch.sum(item_logits, dim=0)).unsqueeze(0)
        winner = torch.argmax(scores, dim=-1)
        if self.training:

            gold_index, next_item = hypergraph.return_gold_next(possible_items)

            # print_blue("ind {} item {}".format(gold_index, next_item))
        else:
            gold_index = None
            next_item = possible_items[winner]

        return next_item, hypergraph, scores, gold_index, rel_loss

    def take_step(self, x, gold_next_item, hypergraph, oracle_agenda, pred_item, pending):
        if self.training:

            key = (gold_next_item.i, gold_next_item.j, gold_next_item.h)
            di = gold_next_item
        else:
            di = pred_item
            key = (pred_item.i, pred_item.j, pred_item.h)

        made_arc = None
        del pending[key]

        if isinstance(di.l, Item):
            # h = di.h
            # m = di.l.h if di.l.h != h else di.r.h
            # (h, m)
            made_arc, _ = hypergraph.make_arc(di)
            (h,m) = made_arc
            if h < m:
                # m is a right child
                hypergraph = hypergraph.add_right_child(h,m)
            else:
                # m is a left child
                hypergraph = hypergraph.add_left_child(h,m)

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
            # print(len(possible_items))
            if di.j >= len(x):
                j = len(x) - 1
            else:
                j = di.j
            if di.h >= len(x):
                h = len(x) - 1
            else:
                h = di.h
            # rep = torch.cat([x[di.i, :], x[j, :], x[h, :]], dim=-1).unsqueeze(0).to(device=constants.device)
            # self.item_lstm.push(rep)

            if len(possible_items) > 0:

                new_item, hypergraph, scores, gold_index, rel_loss = self.predict_next(x, possible_items, hypergraph)
                if new_item is not None:
                    pending[(new_item.i, new_item.j, new_item.h)] = new_item
                    # hypergraph.add_bucket(new_item)
                    # hypergraph.update_chart(new_item)
            else:
                scores, gold_index, rel_loss = None, None, 0
                pass

        return hypergraph, pending, di, made_arc, scores, gold_index, rel_loss

    def init_arc_list(self, tensor_list, oracle_agenda):
        item_list = {}
        for t in tensor_list:
            item = oracle_agenda[(t[0].item(), t[1].item(), t[2].item())]
            item_list[(item.i, item.j, item.h)] = item
        return item_list

    def forward(self, x, transitions, relations, map, heads, rels):
        x_ = x[0][:,1:]
        # average of last 4 hidden layers
        with torch.no_grad():
            out = self.bert(x_.to(device=constants.device))[2]
            x_emb = torch.stack(out[-8:]).mean(0)
        heads_batch = torch.ones((x_emb.shape[0], heads.shape[1])).to(device=constants.device)  # * -1
        prob_sum = 0
        batch_loss = 0
        x_mapped = torch.zeros((x_emb.shape[0], heads.shape[1]+1, x_emb.shape[2] + 100)).to(device=constants.device)
        eos_emb = x_emb[0,-1, :].unsqueeze(0).to(device=constants.device)
        eos_emb = torch.cat([eos_emb,torch.zeros((1,100)).to(device=constants.device)],dim=-1).to(device=constants.device)
        for i, sentence in enumerate(x_emb):
            # initialize a parser
            mapping = map[i, map[i] != -1]
            tag = self.tag_embeddings(x[1][i][x[1][i] != 0].to(device=constants.device))
            sentence = sentence[:-1,:]
            s = self.get_bert_embeddings(mapping, sentence, tag)
            s = torch.cat([s,eos_emb],dim=0)
            curr_sentence_length = s.shape[0]
            x_mapped[i, :curr_sentence_length, :] = s

        sent_lens = (x_mapped[:, :, 0] != 0).sum(-1).to(device=constants.device)
        max_len = torch.max(sent_lens)
        h_t = self.run_lstm(x_mapped, sent_lens)
        # initial_weights_logits = self.get_head_logits(h_t, sent_lens)
        h_t_noeos = torch.zeros((h_t.shape[0],heads.shape[1],h_t.shape[2])).to(device=constants.device)
        tree_loss = 0
        for i in range(h_t.shape[0]):

            curr_sentence_length = sent_lens[i]-1

            # curr_init_weights = initial_weights_logits[i]
            # curr_init_weights = curr_init_weights[:curr_sentence_length + 1, :curr_sentence_length + 1]
            # curr_init_weights = torch.exp(curr_init_weights)

            oracle_hypergraph = transitions[i]
            oracle_hypergraph = oracle_hypergraph[oracle_hypergraph[:, 0, 0] != -1, :, :]
            oracle_agenda = self.init_agenda_oracle(oracle_hypergraph, rels[i])

            s = h_t[i, :curr_sentence_length+1, :]
            chart = Chart()
            pending = self.init_pending(curr_sentence_length)
            hypergraph = self.hypergraph(curr_sentence_length, chart, rels[i])

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
            current_representations = s.clone()

            oracle_hypergraph_picks = oracle_hypergraph[:, -1, :].clone()
            list_oracle_hypergraph_picks = [t for t in oracle_hypergraph_picks]

            # oracle_hypergraph_picks[:,-2,:] = torch.zeros_like(oracle_hypergraph_picks[:,0,:]).to(device=constants.device)
            oracle_transition_picks = oracle_hypergraph[:, :-1, :].clone()

            dim1 = int(oracle_transition_picks.shape[0] * oracle_transition_picks.shape[1])
            dim2 = int(oracle_transition_picks.shape[2])

            oracle_transition_picks = oracle_transition_picks.view(dim1, dim2)
            # oracle_hypergraph_picks = oracle_hypergraph_picks.view(dim1,dim2)
            arc_list = self.init_arc_list(list_oracle_hypergraph_picks, oracle_agenda)
            hypergraph = hypergraph.set_possible_next(arc_list)
            for step in range(len(oracle_transition_picks)):

                # if arc_index_aux != 2:
                #scores = self.attn_score_item(current_representations,s,pending,hypergraph)
                scores, item_to_make, gold_index, hypergraph, gold_next_item, rel_loss = self.pick_next(
                    current_representations,
                    curr_sentence_length,
                    pending,
                    hypergraph,
                    oracle_transition_picks[
                        step])
                hypergraph, pending, made_item, made_arc, scores_hg, gold_index_hg, rel_loss_hg = self.take_step(
                    current_representations,
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

                if self.training:
                    loss += 0.25 * nn.CrossEntropyLoss()(scores,
                                                         gold_index) + 0.25 * rel_loss + 0.25 * rel_loss_hg
                    if gold_index_hg is not None and scores_hg is not None:
                        loss += 0.25 * nn.CrossEntropyLoss()(scores_hg,
                                                             gold_index_hg)
            pred_heads = self.heads_from_arcs(arcs, curr_sentence_length)
            heads_batch[i, :curr_sentence_length] = pred_heads

            loss /= len(oracle_hypergraph)
            h_t_noeos[i,:curr_sentence_length,:] = h_t[i, :curr_sentence_length, :]
            batch_loss += loss
        batch_loss /= x_emb.shape[0]
        heads = heads_batch
        # tree_loss /= x_emb.shape[0]
        l_logits = self.get_label_logits(h_t_noeos, heads)
        rels_batch = torch.argmax(l_logits, dim=-1)
        # rels_batch = rels_batch.permute(1, 0)
        batch_loss += self.loss(batch_loss, l_logits, rels)
        # batch_loss += tree_loss
        return batch_loss, heads_batch, rels_batch

    def predict_next_biaffine(self, x, possible_items, hypergraph, oracle_agenda, list_possible_next):
        n = len(x)
        z = torch.zeros_like(x[0, :]).to(device=constants.device)
        scores = []
        h_11 = self.dropout(F.relu(self.linear_h11(x))).unsqueeze(0)
        h_21 = self.dropout(F.relu(self.linear_h21(x))).unsqueeze(0)
        h, _ = self.compressor(x.unsqueeze(1))

        h_logits = self.biaffine_h(h_11, h_21).squeeze(0)
        change = torch.kron(h_logits, h)
        dim = int(h_logits.shape[0])
        change = change.view(dim, dim, dim)
        mask = torch.zeros_like(change, dtype=torch.bool).to(device=constants.device)

        for item in possible_items:
            i, j, h = item.i, item.j, item.h

            if j >= len(x):
                j = len(x) - 1
            mask[i, j, h] = True

        scores = torch.masked_select(change, mask).unsqueeze(0)
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
        h, _ = self.compressor(s.unsqueeze(1))

        h_logits = self.biaffine_item(h_arc.unsqueeze(0), h_dep.unsqueeze(0)).squeeze(0)
        change = torch.kron(h_logits, h)
        dim = int(h_logits.shape[0])
        change = change.view(dim, dim, dim)
        scores = []
        mask = torch.zeros_like(change, dtype=torch.bool).to(device=constants.device)
        n = h_logits.shape[0]
        for iter, item in enumerate(pending.values()):
            if item.l in hypergraph.bucket or item.r in hypergraph.bucket:
                continue
            i, j, h = item.i, item.j, item.h

            if i == oracle_item[0] and j == oracle_item[1] and h == oracle_item[2]:
                gold_index = torch.tensor([iter], dtype=torch.long).to(device=constants.device)
            if j >= n:
                j = n - 1
            mask[i, j, h] = True
            # scores.append(h_logits[i, j])
        scores = torch.masked_select(change, mask).unsqueeze(0)
        winner = torch.argmax(scores, dim=-1)
        winner_item = list(pending.values())[winner]

        gold_next_item = None
        if gold_index is not None:
            gold_next_item = list(pending.values())[gold_index]

        return scores, winner_item, gold_index, hypergraph, gold_next_item

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
        # ###print("cherry pies")
        # for item in pending.values():
        #    ###print(item)
        return pending

    def heads_from_arcs(self, arcs, sent_len):
        heads = [0] * sent_len
        for (u, v) in arcs:
            heads[v] = u  # .item()
        return torch.tensor(heads).to(device=constants.device)


    def tree_lstm(self, x, left_children, right_children):
        # print_blue(left_children)
        # print_blue(right_children)
        left_reps = x[list(left_children), :].unsqueeze(1).to(device=constants.device)
        right_reps = x[list(right_children), :].to(device=constants.device)
        right_reps = torch.flip(right_reps, dims=[0, 1]).unsqueeze(1).to(device=constants.device)
        # print_green(left_reps.shape)
        # print_green(right_reps.shape)
        _, (lh, _) = self.lstm_tree(left_reps)
        _, (rh, _) = self.lstm_tree(right_reps)
        # print_yellow(lh.shape)
        # print_yellow(rh.shape)
        c = torch.cat([lh, rh], dim=-1).to(device=constants.device)
        c = nn.Tanh()(self.linear_tree(c))
        return c
