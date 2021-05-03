import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

import numpy as np

from utils import constants
from .base import BaseParser
from .modules import Biaffine, Bilinear, StackLSTM
from .word_embedding import WordEmbedding, ActionEmbedding
from ..algorithm.transition_parsers import ShiftReduceParser, arc_standard, arc_eager, hybrid, non_projective, \
    easy_first
def ttt():
    ting = StackLSTM(100 * 2, int(100 / 2),dropout=0.3,
                                 batch_first=True, bidirectional=False)

    ting.push(make_root=True)
    ting.push()
    ting.push()
    ting.pop()
    ting.push()
    ting.push()
    ting.pop()
    ting.push(make_root=True)
    ting.push()
    ting.push()
    ting.push()
    ting.pop()
    ting.pop()
    ting.pop()
    ting.push()
    ting.plot_structure(show=True)


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
    def compute_trees(self, item, label):
        i, j, h = item.i, item.j, item.h
        item_l = item.l
        item_r = item.r
        il, jl, hl = item_l.i, item_l.j, item_l.h
        ir, jr, hr = item_r.i, item_r.j, item_r.h

        tl = self.trees[(il, jl, hl)]
        tr = self.trees[(ir, jr, hr)]
        repr_l = torch.cat([tl, tr, label], dim=-1)
        repr_r = torch.cat([tr, tl, label], dim=-1)
        c_l = nn.Tanh()(self.linear_tree_left(repr_l))
        c_r = nn.Tanh()(self.linear_tree_right(repr_r))
        self.trees[(i, j, hl)] = c_l
        self.trees[(i, j, hr)] = c_r


    def make_legal(self, x, picks):
        scores = torch.ones_like(x)
        scores *= -float('inf')
        for (u, v) in picks:
            scores[u, v] = x[u, v]
            scores[v, u] = x[v, u]
        return scores

    def new_trees(self, item, popped):

        # print(colored("IN hypergraph chart {}".format(len(self.chart.chart)), "red"))

        i, j, h = item.i, item.j, item.h
        # if item.l in self.bucket or item.r in self.bucket:
        #    #print(colored("whole squad","blue"))
        #    return []
        picks_left = []
        picks_right = []
        picks = []
        for k in range(0, i + 1):
            for g in range(k, i):
                if (k, i, g) in self.chart and g not in popped:
                    item_l = self.chart[(k, i, g)]
                    if g not in popped and h not in popped:
                        picks.append((g, h))
                        self.locator[(g, h)] = Item(k, j, g, item_l, item)
                    if h not in popped and g not in popped:
                        picks.append((g, h))
                        self.locator[(h, g)] = Item(k, j, h, item_l, item)
                    # kjg = torch.tensor([[k, i, g], [i, j, h], [k, j, g]], dtype=torch.int).to(device=constants.device)
                    # kjh = torch.tensor([[k, i, g], [i, j, h], [k, j, h]], dtype=torch.int).to(device=constants.device)
                    # picks_left.append(kjg)
                    # picks_right.append(kjh)
        for k in range(j, self.n + 1):
            for g in range(j, k):
                if (j, k, g) in self.chart:
                    item_r = self.chart[(j, k, g)]
                    if h not in popped and g not in popped:
                        picks.append((h, g))
                        self.locator[(h, g)] = Item(i, k, h, item, item_r)
                    if g not in popped and h not in popped:
                        picks.append((h, g))
                        self.locator[(g, h)] = Item(i, k, g, item, item_r)
                    # ikh = torch.tensor([[i, j, h], [j, k, g], [i, k, h]], dtype=torch.int).to(device=constants.device)
                    # ikg = torch.tensor([[i, j, h], [j, k, g], [i, k, g]], dtype=torch.int).to(device=constants.device)
                    # picks_left.append(ikh)
                    # picks_right.append(ikg)
        return picks
    def subtrees_(self, item):
        i, j, h = item.i, item.j, item.h
        for k in range(i, j):
            for r in range(i, j):

                if (i, k, r) in self.chart:
                    item_l = self.chart[(k, i, r)]
                    item.subtrees[item_l] = item_l
                if (j, k, r) in self.chart:
                    item_r = self.chart[(j, k, r)]
                    item.subtrees[item_r] = item_r

        return item

    def arc(self, item):
        i, j, h = item.i, item.j, item.h
        # print((i,j,h))
        if i + 1 == j:
            if i == h:
                item.arcs.append((h, j))
            else:
                item.arcs.append((h, i))
        else:
            if item.l.h == h:
                item.arcs = item.l.arcs + item.r.arcs + [(h, item.r.h)]
            else:
                item.arcs = item.l.arcs + item.r.arcs + [(h, item.l.h)]

        return item

    def traverse_up(self, item_l, item_r, path):
        path.append(item_l)
        path.append(item_r)
        if not self.is_axiom(item_l):
            path = self.traverse_up(item_l.l, item_l.r, path)
        if not self.is_axiom(item_r):
            path = self.traverse_up(item_r.l, item_r.r, path)

        return path

    def easy_first_path(self):
        for item in self.chart:
            i, j, h = item.i, item.j, item.h
        pass

    def check_laplacian(self, L):
        return torch.all(torch.diag(L) >= 0.0) and torch.all(L - torch.diag(L) <= 0.0)

    def partition(self):
        # G = complete graph on self.n
        A = torch.exp(self.W)
        A = A - torch.diag(A)
        col_sum = torch.diag(torch.sum(A, dim=0))
        L = -1 * A + col_sum
        # A = torch.exp(A)
        # A = torch.narrow(A, 0, 0, self.n)
        # A = torch.narrow(A,-1,0,self.n)
        # B = A * (1 - torch.eye(self.n+1, self.n+1))
        # col_sum = torch.diag(torch.sum(B, dim=-1))
        # L = col_sum - A
        indices = list(range(self.n + 1))
        indices.remove(1)
        L = L[indices, :]
        L = L[:, indices]
        print(L)
        # L = torch.narrow(L,0,1,self.n)
        # L = torch.narrow(L,-1,1,self.n)
        print("XUYUYUYUYUYU {}".format(self.check_laplacian(L)))

        # L[0, :] = r[:-1]
        Z = L.det()
        return Z

    def probability(self, heads):
        heads_proper = [0] + heads.tolist()

        sentence_proper = list(range(len(heads_proper)))
        word2head = {w: h for (w, h) in zip(sentence_proper, heads_proper)}
        arcs = []
        for word in word2head:
            arcs.append((word2head[word], word))
        psum = 0
        for (u, v) in arcs:
            if u == 0:
                root = v
                continue
            psum += self.W[u, v]
        root_selection = self.W[root, :]
        psum = torch.exp(torch.sum(torch.tensor(psum).to(device=constants.device)))
        Z = self.partition()
        return psum / Z  # self.partition(root_selection)

    def best_path(self):
        for item in self.chart:
            # print(item)
            if item[0] == 0 and item[1] == self.n:
                goal = (0, self.n, item[2])
                break

        goal_item = self.chart[goal]
        arcs = []
        scores = torch.zeros(self.n)
        path = [goal_item]

        path = self.traverse_up(goal_item.l, goal_item.r, path)
        # print(path)
        arcs = set(goal_item.arcs)
        path_arcs = set([])
        for item in path:
            if item != goal_item:
                path_arcs = path_arcs.union(set(item.arcs))

        # print(path_arcs)
        # print(arcs)

        arcs = arcs.intersection(path_arcs)
        # do in order

        # scores = [self.chart[item].w for item in reversed(path)]
        # print(scores)

        # tot_sum = sum(scores)
        # cumsum = torch.cumsum(torch.tensor(scores), dim=-1)
        # probs = cumsum  # /tot_sum

        arcs = set(arcs)
        heads = torch.zeros(self.n)
        for (u, v) in arcs:
            heads[v - 1] = u
        return heads





    def pick_next(self, words, n, pending, hypergraph, oracle_item):
        scores = []

        gold_index = None
        # all_embedding = self.item_lstm.embedding().squeeze(0)
        # rel_0 = torch.tensor([0], dtype=torch.long).to(device=constants.device)
        # rel_dummy = self.rel_embeddings(rel_0).to(device=constants.device)
        # zrel = torch.zeros_like(rel_dummy).to(device=constants.device)
        rel_loss = 0
        items_tensor = torch.zeros((1, 400)).to(device=constants.device)
        for iter, item in enumerate(pending.values()):
            if item.l in hypergraph.bucket or item.r in hypergraph.bucket:
                continue
            i, j, h = item.i, item.j, item.h
            if i == oracle_item[0] and j == oracle_item[1] and h == oracle_item[2]:
                gold_index = torch.tensor([iter], dtype=torch.long).to(device=constants.device)
            # if item in hypergraph.scored_items:
            #    score = torch.tensor(item.score).to(device=constants.device)
            # else:
            # if j >= len(words):
            #    j = len(words) - 1
            # features_derived = torch.cat([words[i, :], words[j, :], words[h, :]], dim=-1).to(
            #    device=constants.device)
            # rel = item.rel
            # if rel is not None:
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
            # else:
            #    rel_embed = zrel
            #    rel_loss += 0
            # features_derived = torch.cat([features_derived, all_embedding, rel_embed.squeeze(0)], dim=-1)
            # features_derived = self.tree_rep(i,j,h,words)
            # score = self.mlp_small(features_derived)
            # if item not in hypergraph.repped_items:
            left_children = hypergraph.get_left_children_from(h, i)  # list(range(i)) + [h]
            right_children = hypergraph.get_right_children_until(h, j)  # [h] + list(range(j, h))
            features_derived = self.tree_lstm(words, left_children, right_children).squeeze(0)
            hypergraph = hypergraph.set_item_vec(features_derived, item)
            # else:
            #    features_derived = item.vector_rep
            if iter == 0:
                items_tensor = features_derived
            else:
                items_tensor = torch.cat([items_tensor, features_derived], dim=0)

            # score = self.mlp_small(features_derived)
            # item.update_score(score)
            # hypergraph.score_item(item)
            # scores.append(score)
        # scores = torch.stack(scores).squeeze(1).permute(1, 0)
        # scores = torch.tensor(scores).to(device=constants.device).unsqueeze(0)
        h1 = self.dropout(F.relu(self.linear_items1(items_tensor))).unsqueeze(0)
        h2 = self.dropout(F.relu(self.linear_items2(items_tensor))).unsqueeze(0)
        item_logits = self.biaffine_item(h1, h2).squeeze(0)

        #scores = self.bilinear_item(h1, h2).squeeze(0).permute(1, 0)
        scores = nn.Softmax(dim=-1)(torch.sum(item_logits,dim=0)).unsqueeze(0)
        # scores = nn.Softmax(dim=-1)(F.relu(self.ls(item_logits)))

        winner = torch.argmax(scores, dim=-1)
        winner_item = list(pending.values())[winner]
        # print_green(winner_item)
        # print_red(list(pending.values())[gold_index])
        gold_next_item = None
        if gold_index is not None:
            gold_next_item = list(pending.values())[gold_index]
        return scores, winner_item, gold_index, hypergraph, gold_next_item, rel_loss
    def predict_next(self, x, possible_items, hypergraph):
        n = len(x)
        z = torch.zeros_like(x[0, :]).to(device=constants.device)
        items_tensor = torch.zeros((1, 400)).to(device=constants.device)

        # all_embedding = self.item_lstm.embedding().squeeze(0)
        # rel_0 = torch.tensor([0], dtype=torch.long).to(device=constants.device)
        # rel_dummy = self.rel_embeddings(rel_0).to(device=constants.device)
        # zrel = torch.zeros_like(rel_dummy).to(device=constants.device)
        rel_loss = 0
        for iter, item in enumerate(possible_items):
            i, j, h = item.i, item.j, item.h
            # print_yellow((i,j,h))
            # if item in hypergraph.scored_items:
            #    score = item.score
            # else:
            # if j >= len(x):
            #    j = len(x) - 1
            # if h >= len(x):
            #    h = len(x) - 1
            # features_derived = torch.cat([x[i, :], x[j, :], x[h, :]], dim=-1).to(device=constants.device)
            # rel = item.rel
            # if rel is not None:
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
            # else:
            #    rel_embed = zrel
            #    rel_loss += 0
            # features_derived = torch.cat([features_derived, all_embedding, rel_embed.squeeze(0)], dim=-1)
            # score = self.mlp2(features_derived)  # *l_score*r_score
            # item.update_score(score)
            # left_children = list(range(i))+[h]
            # right_children = [h]+list(range(j,h))
            # features_derived = self.tree_lstm(x, left_children, right_children).squeeze(0)
            ##features_derived = self.tree_rep(i, j, h, x)
            # score = self.mlp_small(features_derived)
            # hypergraph.score_item(item)
            # scores.append(score)
            # if item.vector_rep is None:
            # if item not in hypergraph.repped_items:
            left_children = hypergraph.get_left_children_from(h, i)  # list(range(i)) + [h]
            right_children = hypergraph.get_right_children_until(h, j)  # [h] + list(range(j, h))

            features_derived = self.tree_lstm(x, left_children, right_children).squeeze(0)
            hypergraph = hypergraph.set_item_vec(features_derived, item)
            # else:
            #    features_derived = item.vector_rep
            # else:
            #    features_derived = item.vector_rep
            if iter == 0:
                items_tensor = features_derived
            else:
                items_tensor = torch.cat([items_tensor, features_derived], dim=0)
        # scores = torch.stack(scores).squeeze(1).permute(1,0)
        h1 = self.dropout(F.relu(self.linear_items1(items_tensor))).unsqueeze(0)
        h2 = self.dropout(F.relu(self.linear_items2(items_tensor))).unsqueeze(0)
        item_logits = self.biaffine_item(h1, h2).squeeze(0)

        #scores = self.bilinear_item(h1, h2).squeeze(0).permute(1, 0)
        scores = nn.Softmax(dim=-1)(torch.sum(item_logits, dim=0)).unsqueeze(0)
        winner = torch.argmax(scores, dim=-1)
        if self.training:

            gold_index, next_item = hypergraph.return_gold_next(possible_items)

            # print_blue("ind {} item {}".format(gold_index, next_item))
        else:
            gold_index = None
            next_item = possible_items[winner]

        return next_item, hypergraph, scores, gold_index, rel_loss

# if arc_index_aux != 2:
# scores = self.attn_score_item(current_representations,s,pending,hypergraph)
# scores, item_to_make, gold_index, hypergraph, gold_next_item, rel_loss = self.pick_next(
#    current_representations,
#    curr_sentence_length,
#    pending,
#    hypergraph,
#    oracle_transition_picks[
#        step])


 # self.mlp = nn.Linear(500 * 3, 1)
        # l1 = nn.Linear(hidden_size * 6 + rel_embedding_size, 3 * hidden_size)
        # l1 = nn.Linear(hidden_size * 6 + rel_embedding_size, 3 * hidden_size)
        # l11 = nn.Linear(hidden_size * 6 + rel_embedding_size, 3 * hidden_size)
        # l11 = nn.Linear(hidden_size * 6 + rel_embedding_size, 3 * hidden_size)
        # l2 = nn.Linear(3 * hidden_size, hidden_size)
        # l2 = nn.Linear(3 * hidden_size, hidden_size)
        # lf = nn.Linear(hidden_size, 1)
        # lf = nn.Linear(hidden_size, 1)
        # l22 = nn.Linear(3 * hidden_size, hidden_size)
        # l22 = nn.Linear(3 * hidden_size, hidden_size)
        # lf2 = nn.Linear(hidden_size, 1)
        # lf2 = nn.Linear(hidden_size, 1)
        # l3 = nn.Linear(hidden_size * 3, hidden_size)
        # l3 = nn.Linear(hidden_size * 3, hidden_size)
        # l33 = nn.Linear(hidden_size, self.num_rels)
        # l33 = nn.Linear(hidden_size, self.num_rels)
        # layers = [l1, nn.ReLU(), l2, nn.ReLU(), lf, nn.Sigmoid()]
        # layers2 = [l11, nn.ReLU(), l22, nn.ReLU(), lf2, nn.Sigmoid()]
        # layers3 = [l3, nn.ReLU(), l33, nn.Softmax(dim=-1)]
        # self.ls = nn.Linear(200,1)
        # layers_small = [ls,nn.Sigmoid()]
        # self.mlp = nn.Sequential(*layers)
        # self.mlp2 = nn.Sequential(*layers2)
        # self.mlp_rel = nn.Sequential(*layers3)
        # self.mlp_small = nn.Sequential(*layers_small)


        # self.linear_h11 = nn.Linear(hidden_size, 200).to(device=constants.device)
        # self.linear_h12 = nn.Linear(hidden_size, 200).to(device=constants.device)
        # self.linear_h21 = nn.Linear(hidden_size, 200).to(device=constants.device)
        # self.linear_h22 = nn.Linear(hidden_size, 200).to(device=constants.device)



        # self.weight_matrix = nn.MultiheadAttention(868, num_heads=1, dropout=dropout).to(device=constants.device)
        # self.root_selector = nn.LSTM(
        #    868, 1, 1, dropout=(dropout if 1 > 1 else 0),
        #    batch_first=True, bidirectional=False).to(device=constants.device)


        # self.compressor = nn.LSTM(hidden_size, 1, 1, batch_first=False, bidirectional=False).to(
        #    device=constants.device)

        # input_init = torch.zeros((1, hidden_size * 3)).to(
        #    device=constants.device)
        # hidden_init = torch.zeros((1, hidden_size * 3)).to(
        #    device=constants.device)
        # self.empty_initial = nn.Parameter(torch.zeros(1, hidden_size * 3)).to(device=constants.device)

        # self.lstm_init_state = (nn.init.xavier_uniform_(input_init), nn.init.xavier_uniform_(hidden_init))
        # self.stack_lstm = nn.LSTMCell(hidden_size * 3, hidden_size * 3).to(device=constants.device)

        # self.item_lstm = StackCell(self.stack_lstm, self.lstm_init_state, self.lstm_init_state, self.dropout,
        #                           self.empty_initial)

        # self.ij_score = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1)
        # self.ih_score = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1)
        # self.jh_score = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1)

        # with torch.no_grad():
        #    eos_tens = torch.tensor([self.eos_token_id]).unsqueeze(0).to(device=constants.device)
        #    out = self.bert(eos_tens)[2]
        #    eos_emb = torch.stack(out[-8:]).mean(0)
        # print(eos_emb.shape)
        # print(eos_emb)
        # kjsl



    for (u, v) in ordered_arcs:
        derived_items = sorted(derived_items, key=lambda x: x[2])
        item_l, item_r, new_item = None, None, None
        found = False
        for (i1, j1, h1,b1) in derived_items:
            for (i2, j2, h2,b2) in derived_items:
                # lae: j --> k; [i^b,v]|[k^0,u] --> [i^b,j]; i^b = h^b
                if j2 == u and j1 == v and i2 == v and b2 == 0:
                    item_l = (i1,j1,h1,b1)
                    item_r = (i2,j2,h2,b2)
                    found = True
                    new_item = (i1,j2,h1,b1)
                    if new_item not in derived_items:
                        derived_items.append(new_item)

                    break
                if i1 == u and j1 == v:
                    found = True
                    item_l = (i1,j1,h1,b1)
                    item_r = (i1,j1,h1,b1)
                    new_item = (j1,j1+1,j1,1)
                    if new_item not in derived_items:
                        derived_items.append(new_item)
                    break
                if i2 == u and j2 == v:
                    found = True
                    item_l = (i2, j2, h2, b2)
                    item_r = (i2, j2, h2, b2)
                    new_item = (j2, j2 + 1, j2, 1)
                    if new_item not in derived_items:
                        derived_items.append(new_item)

                    break
                if b2 == 1 and j1 == i2:
                    item_l = (i1,j1,h1,b1)
                    item_r = (i2,j2,h2,b2)
                    new_item = (i1,j2,i1,b1)
                    if new_item not in derived_items:
                        derived_items.append(new_item)
                    break
        if found:
            ordered_arcs.remove((u,v))
        b_hypergraph.append((item_l, item_r, new_item))
        if item_l in derived_items:
            derived_items.remove(item_l)
        if item_r in derived_items:
            derived_items.remove(item_r)

    def attn_score_item(self, x, orig, items, hypergraph):
        n = x.shape[0]
        print_green(orig.shape)
        mask_ij = torch.empty((n, n)).fill_(float('-inf'))
        mask_ih = torch.empty((n, n)).fill_(float('-inf'))
        mask_jh = torch.empty((n, n)).fill_(float('-inf'))
        mask = torch.zeros((n, n, n), dtype=torch.bool).to(device=constants.device)

        for iter, item in enumerate(items.values()):
            if item.l in hypergraph.bucket or item.r in hypergraph.bucket:
                continue
            i, j, h = item.i, item.j, item.h
            mask_ij[i, j] = 0
            mask_ih[i, h] = 0
            mask_jh[j, h] = 0
            mask[i, j, h] = True
        # put original matrix as third input?
        x = x.unsqueeze(1)
        attn_ij = self.ij_score(x, x, x, attn_mask=mask_ij)[1].squeeze(0)
        attn_ih = self.ih_score(x, x, x, attn_mask=mask_ih)[1].squeeze(0)
        attn_jh = self.jh_score(x, x, x, attn_mask=mask_jh)[1].squeeze(0)
        scores = torch.zeros((n, n, n))
        for item in items.values():
            if item.l in hypergraph.bucket or item.r in hypergraph.bucket:
                continue
            i, j, h = item.i, item.j, item.h
            scores[i, j, h] = torch.exp(attn_ij[i, j]) * attn_jh[j, h] * attn_ih[i, h]
        scores = torch.masked_select(scores, mask)
        return attn_jh

    # left_children = hypergraph.get_left_children_from(h, i)
    # right_children = hypergraph.get_right_children_until(h, j)
    # left_reps = words[list(left_children), :].unsqueeze(1).to(device=constants.device)
    # right_reps = words[list(right_children), :].to(device=constants.device)
    # right_reps = torch.flip(right_reps, dims=[0, 1]).unsqueeze(1).to(device=constants.device)
    #
    # if not prune:
    #    print_green(left_children)
    #    print_blue(right_children)
    #    print_red((i,j,h))
    # features_derived = self.tree_lstm(words, left_children, right_children).squeeze(0)

    """
           for i in range(h_t.shape[0]):

               curr_sentence_length = sent_lens[i] - 1

               # curr_init_weights = initial_weights_logits[i]
               # curr_init_weights = curr_init_weights[:curr_sentence_length + 1, :curr_sentence_length + 1]
               # curr_init_weights = torch.exp(curr_init_weights)

               oracle_hypergraph = transitions[i]
               oracle_hypergraph = oracle_hypergraph[oracle_hypergraph[:, 0, 0] != -1, :, :]
               oracle_agenda = self.init_agenda_oracle(oracle_hypergraph, rels[i])

               s = h_t[i, :curr_sentence_length + 1, :]
               s_b = bh_t[i, :curr_sentence_length + 1, :]
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
               current_representations = s#.clone()
               current_representations_back = s_b#.clone()

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
               # possible_items = {}
               # want this to hold item reps
               popped = {}
               nc = curr_sentence_length*(curr_sentence_length+1)**2
               loss_f = LabelSmoothingLoss(classes=nc)
               for step in range(len(oracle_transition_picks)):
                   scores, item_to_make, gold_index, \
                   hypergraph, gold_next_item, pending = self.predict_next_prn(current_representations,
                                                                               current_representations_back,
                                                                               pending, hypergraph,
                                                                               oracle_transition_picks[step])
                   hypergraph, pending, made_item, \
                   made_arc, scores_hg, gold_index_hg = self.take_step(current_representations,
                                                                       current_representations_back,
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

                       h_rep = self.tree_lstm(current_representations_back, left_children[h], right_children[h])
                       current_representations_back = current_representations_back.clone()
                       current_representations_back[h, :] = h_rep

                   if self.training:
                       #smooth_label = self.smooth_one_hot(gold_index,len(scores),smoothing=0.33)

                       #loss += 0.5 * nn.CrossEntropyLoss(reduction='sum')(scores, gold_index)
                       loss += 0.5 * loss_f(scores, gold_index)
                       if gold_index_hg is not None and scores_hg is not None:
                           #smooth_label_hg = self.smooth_one_hot(gold_index_hg, len(scores_hg), smoothing=0.33)

                           #loss += 0.5 * nn.CrossEntropyLoss(reduction='sum')(scores_hg, smooth_label_hg)
                           loss += 0.5 * loss_f(scores_hg, gold_index_hg)
               """



    def outgoing(self, item):
        i, j, h = item.i, item.j, item.h

        all_arcs = {}
        for k in range(0, i + 1):
            if (k, i, k) in self.chart:
                if self.chart[(k, i, k)] not in self.bucket:
                    item_l = self.chart[(k, i, k)]
                    item_1 = Item(k, j, k, item_l, item)
                    _, item_1 = self.make_arc(item_1, add_rel=True)
                    # all_arcs.append(item_1)
                    all_arcs[item_1] = item_1
                    if j < self.n:
                        item_2 = Item(k, j, j, item_l, item)
                        _, item_2 = self.make_arc(item_2, add_rel=True)
                        # all_arcs.append(item_2)
                        all_arcs[item_2] = item_2
            if (k, i, i) in self.chart:
                if self.chart[(k, i, i)] not in self.bucket:
                    item_l = self.chart[(k, i, i)]
                    item_1 = Item(k, j, k, item_l, item)
                    _, item_1 = self.make_arc(item_1, add_rel=True)
                    # all_arcs.append(item_1)
                    all_arcs[item_1] = item_1
                    if j < self.n:
                        item_2 = Item(k, j, j, item_l, item)
                        _, item_2 = self.make_arc(item_2, add_rel=True)
                        # all_arcs.append(item_2)
                        all_arcs[item_2] = item_2

        for k in range(j, self.n + 1):
            if (j, k, j) in self.chart:
                if self.chart[(j, k, j)] not in self.bucket:
                    item_r = self.chart[(j, k, j)]
                    item_1 = Item(i, k, i, item, item_r)
                    _, item_1 = self.make_arc(item_1, add_rel=True)
                    # all_arcs.append(item_1)
                    all_arcs[item_1] = item_1
                    if k < self.n:
                        item_2 = Item(i, k, k, item, item_r)
                        _, item_2 = self.make_arc(item_2, add_rel=True)
                        # all_arcs.append(item_2)
                        all_arcs[item_2] = item_2
            if (j, k, k) in self.chart:
                if self.chart[(j, k, k)] not in self.bucket:
                    item_r = self.chart[(j, k, k)]
                    item_1 = Item(i, k, i, item, item_r)
                    _, item_1 = self.make_arc(item_1, add_rel=True)
                    # all_arcs.append(item_1)
                    all_arcs[item_1] = item_1
                    if k < self.n:
                        item_2 = Item(i, k, k, item, item_r)
                        _, item_2 = self.make_arc(item_2, add_rel=True)
                        # all_arcs.append(item_2)
                        all_arcs[item_2] = item_2

        return all_arcs



    def parse_step_easy_first(self, parser, labeled_transitions):
        # item = parser.next_item()

        (target_head, target_mod, target_direction) = labeled_transitions[0]
        rel = labeled_transitions[1]
        if target_direction == 1:
            best_action = target_head * 2
        else:
            best_action = target_head * 2 + 1
        action_target = torch.tensor([best_action], dtype=torch.long).to(device=constants.device)
        rel_target = torch.tensor([rel], dtype=torch.long).to(device=constants.device)
        rel_embed = self.rel_embeddings(rel_target).to(device=constants.device)
        action_probabilities, rel_probabilities, index, dir, self.pending = parser.score_pending(self.mlp_u, self.mlp_l,
                                                                                                 self.pending,
                                                                                                 self.action_embeddings(
                                                                                                     torch.tensor(1,
                                                                                                                  dtype=torch.long).to(
                                                                                                         device=constants.device)),
                                                                                                 self.action_embeddings(
                                                                                                     torch.tensor(0,
                                                                                                                  dtype=torch.long).to(
                                                                                                         device=constants.device)))
        if mode == 'eval':
            rel = torch.argmax(rel_probabilities, dim=-1)
            rel_embed = self.rel_embeddings(rel).to(device=constants.device)
            rel = rel.item()  # +1
            if index == 0:
                if dir == 1:
                    # left
                    index_head = index + 1
                    index_mod = index
                else:
                    index_head = index
                    index_mod = index + 1
            elif index == len(parser.pending) - 1:
                if dir == 1:
                    # left
                    index_head = index
                    index_mod = index - 1
                else:
                    index_head = index - 1
                    index_mod = index
            elif dir == 1:
                index_mod = index
                index_head = index - 1
            elif dir == 0:
                index_mod = index + 1
                index_head = index
            parser.easy_first_action(index_head, index_mod, rel, rel_embed, self.linear_tree)
        else:
            parser.easy_first_action(target_head, target_mod, rel, rel_embed, self.linear_tree)

        return parser, (action_probabilities, rel_probabilities), (action_target, rel_target)


class EasyFirstParser(BertParser):
    def __init__(self, vocabs, embedding_size, rel_embedding_size, batch_size,
                 dropout=0.33, beam_size=10, transition_system=None, is_easy_first=True):
        super().__init__(vocabs, embedding_size, rel_embedding_size, batch_size,
                         dropout=dropout, beam_size=beam_size, transition_system=transition_system)

        self.dropout = nn.Dropout(self.dropout_prob)
        # lstms
        stack_lstm_size = self.embedding_size + self.rel_embedding_size
        self.stack_lstm = nn.LSTMCell(stack_lstm_size, stack_lstm_size).to(device=constants.device)
        self.buffer_lstm = nn.LSTMCell(stack_lstm_size, stack_lstm_size).to(device=constants.device)
        self.action_lstm = nn.LSTMCell(self.action_embeddings_size, self.action_embeddings_size).to(
            device=constants.device)

        input_init = torch.zeros((1, stack_lstm_size)).to(
            device=constants.device)
        hidden_init = torch.zeros((1, stack_lstm_size)).to(
            device=constants.device)

        input_init_act = torch.zeros((1, self.action_embeddings_size)).to(
            device=constants.device)
        hidden_init_act = torch.zeros((1, self.action_embeddings_size)).to(
            device=constants.device)

        self.lstm_init_state = (nn.init.xavier_uniform_(input_init), nn.init.xavier_uniform_(hidden_init))
        self.lstm_init_state_actions = (
            nn.init.xavier_uniform_(input_init_act), nn.init.xavier_uniform_(hidden_init_act))

        self.empty_initial = nn.Parameter(torch.zeros(1, stack_lstm_size)).to(device=constants.device)
        self.empty_initial_act = nn.Parameter(torch.zeros(1, self.action_embeddings_size)).to(device=constants.device)

        self.mlp_u = nn.Sequential(
            nn.Linear(stack_lstm_size * 6 + self.action_embeddings_size, stack_lstm_size * 6),
            nn.ReLU(),
            nn.Linear(stack_lstm_size * 6, stack_lstm_size * 5),
            nn.ReLU(),
            nn.Linear(stack_lstm_size * 5, stack_lstm_size * 4),
            nn.ReLU(),
            nn.Linear(stack_lstm_size * 4, stack_lstm_size * 3),
            nn.ReLU(),
            nn.Linear(stack_lstm_size * 3, stack_lstm_size),
            nn.ReLU(),
            nn.Linear(stack_lstm_size, 1)
        ).to(device=constants.device)

        self.mlp_l = nn.Sequential(
            nn.Linear(stack_lstm_size * 6 + self.action_embeddings_size, stack_lstm_size * 6),
            nn.ReLU(),
            nn.Linear(stack_lstm_size * 6, stack_lstm_size * 5),
            nn.ReLU(),
            nn.Linear(stack_lstm_size * 5, stack_lstm_size * 4),
            nn.ReLU(),
            nn.Linear(stack_lstm_size * 4, stack_lstm_size * 3),
            nn.ReLU(),
            nn.Linear(stack_lstm_size * 3, stack_lstm_size),
            nn.ReLU(),
            nn.Linear(stack_lstm_size, self.num_rels)
        ).to(device=constants.device)

        # self.mlp_u = nn.Linear(stack_lstm_size*6+self.action_embeddings_size,1).to(device=constants.device)
        # self.mlp_l = nn.Linear(stack_lstm_size*6+self.action_embeddings_size,self.num_rels).to(device=constants.device)
        # torch.nn.init.xavier_uniform_(self.mlp_u.weight)
        # torch.nn.init.xavier_uniform_(self.mlp_l.weight)

        self.linear_tree = nn.Linear(self.rel_embedding_size + 2 * stack_lstm_size, stack_lstm_size).to(
            device=constants.device)
        torch.nn.init.xavier_uniform_(self.linear_tree.weight)

        # self.stack = StackCell(self.stack_lstm, self.lstm_init_state, self.lstm_init_state, self.dropout,
        #                      self.empty_initial)
        # self.pending = PendingRNN(self.buffer_lstm, self.lstm_init_state, self.lstm_init_state, self.dropout,
        #                       self.empty_initial)
        self.pending = nn.LSTM(stack_lstm_size, stack_lstm_size)
        # self.action = StackCell(self.action_lstm, self.lstm_init_state_actions, self.lstm_init_state_actions,
        #                       self.dropout,
        #                       self.empty_initial_act)

        # self.transform_weight = nn.GRU(input_size=stack_lstm_size,hidden_size=max_sent_len)

    def easy_first_labeled_transitions(self, transitions, relations):
        labeled_actions = []
        relations = relations.tolist()

        for i in range(0, len(transitions) - 1, 2):
            head = transitions[i]
            mod = transitions[i + 1]
            if head > mod:
                direction = 1
            else:
                direction = 0
            labeled_actions.append(((head, mod, direction), relations.pop(0)))

        return labeled_actions


    def forward(self, x, transitions, relations, map, heads, rels):
        sent_lens = (x[0] != 0).sum(-1).to(device=constants.device)
        max_sent_len = max(sent_lens)
        transit_lens = (transitions != -1).sum(-1).to(device=constants.device)
        # ##print(x[1])
        tags = self.tag_embeddings(x[1].to(device=constants.device))
        # average of last 4 hidden layers
        with torch.no_grad():
            out = self.bert(x[0].to(device=constants.device))[2]
            x_emb = torch.stack(out[-8:]).mean(0)
        num_actions = (max_sent_len - 1) * 2
        probs_action_batch = torch.ones((x_emb.shape[0], transitions.shape[1], num_actions), dtype=torch.float).to(
            device=constants.device)
        probs_rel_batch = torch.ones((x_emb.shape[0], transitions.shape[1], self.num_rels), dtype=torch.float).to(
            device=constants.device)
        targets_action_batch = torch.ones((x_emb.shape[0], transitions.shape[1], 1), dtype=torch.long).to(
            device=constants.device)
        targets_rel_batch = torch.ones((x_emb.shape[0], transitions.shape[1], 1), dtype=torch.long).to(
            device=constants.device)

        heads_batch = torch.ones((x_emb.shape[0], tags.shape[1])).to(device=constants.device)
        rels_batch = torch.ones((x_emb.shape[0], tags.shape[1])).to(device=constants.device)
        heads_batch *= -1
        rels_batch *= -1
        probs_rel_batch *= -1
        probs_action_batch *= -1
        targets_rel_batch *= -1
        targets_action_batch *= -1

        for i, sentence in enumerate(x_emb):
            # initialize a parser
            mapping = map[i, map[i] != -1]
            tag = self.tag_embeddings(x[1][i][x[1][i] != 0].to(device=constants.device))

            s = self.get_bert_embeddings(mapping, sentence, tag)
            curr_sentence_length = s.shape[0]
            curr_transition_length = transit_lens[i]
            s = s[:curr_sentence_length, :]
            # n * embedding_size ---> n*n
            labeled_transitions = self.easy_first_labeled_transitions(transitions[i, :curr_transition_length],
                                                                      relations[i, :curr_sentence_length])
            parser = ShiftReduceParser(s, self.rel_embedding_size, self.transition_system)
            self.pending(s.unsqueeze(1))
            step = 0
            while len(parser.pending) > 1:
                # for step in range(len(labeled_transitions)):
                parser, probs, target = self.parse_step(parser,
                                                        labeled_transitions[step],
                                                        mode)
                step += 1

                (action_probs, rel_probs) = probs
                (action_target, rel_target) = target
                probs_action_batch[i, step, :action_probs.shape[0]] = action_probs.transpose(1, 0)
                probs_rel_batch[i, step, :] = rel_probs
                targets_action_batch[i, step, :] = action_target
                targets_rel_batch[i, step, :] = rel_target

            heads_batch[i, :curr_sentence_length] = parser.heads_from_arcs()[0]
            rels_batch[i, :curr_sentence_length] = parser.heads_from_arcs()[1]
            # self.stack.back_to_init()
            # self.pending.back_to_init()
            # self.action.back_to_init()

        batch_loss = self.loss(probs_action_batch, targets_action_batch, probs_rel_batch, targets_rel_batch)
        return batch_loss, heads_batch, rels_batch
    def score_arcs_eager(self, possible_arcs, gold_arc_set, gold_arc_oracle, possible_items, words_f, words_b):
        gold_index = None
        gold_key = None
        n = len(words_b)
        all_scores = []
        index2key = {}
        ga = (gold_arc_oracle[0].item(), gold_arc_oracle[1].item())
        #for iter_mother , (possible_arcs, possible_items) in enumerate(zip(possible_arcs_all,possible_items_all)):
        scores = []
        for iter, ((u, v), item) in enumerate(zip(possible_arcs, possible_items)):
            if (u, v) == ga:
                gold_index = torch.tensor([iter], dtype=torch.long).to(device=constants.device)
                gold_key = item.key
        for iter, ((u, v), item) in enumerate(zip(possible_arcs, possible_items)):
            i, j, h = item.i, item.j, item.h
            #if gold_index is None:
            #    if (u,v) in gold_arc_set:
            #        gold_index = torch.tensor([iter], dtype=torch.long).to(device=constants.device)
            #        gold_key = item.key
            index2key[iter] = item.key

            span = self.span_rep(words_f, words_b, i, j, n).unsqueeze(0)
            fwd_rep = torch.cat([words_f[u, :], words_f[v, :]], dim=-1).unsqueeze(0)
            bckw_rep = torch.cat([words_b[u, :], words_b[v, :]], dim=-1).unsqueeze(0)
            rep = torch.cat([span, fwd_rep, bckw_rep], dim=-1)
            s = self.mlp(rep)
            scores.append(s)
        scores = torch.stack(scores, dim=-1).squeeze(0)
        all_scores.append(scores)

        #scores = torch.cat([s for s in all_scores],dim=-1)

        if not self.training:
            gold_index = torch.argmax(scores, dim=-1)
            gold_key = index2key[gold_index.item()]
        return scores, gold_index, gold_key
