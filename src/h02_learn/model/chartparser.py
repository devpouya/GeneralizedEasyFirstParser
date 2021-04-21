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
        #self.mlp = nn.Linear(500 * 3, 1)
        l1 = nn.Linear(hidden_size * 3, hidden_size)
        l11 = nn.Linear(hidden_size * 4, hidden_size)
        l2 = nn.Linear(hidden_size,1)
        l22 = nn.Linear(hidden_size,2)
        layers = [l1,nn.ReLU(),l2,nn.ReLU()]
        layers2 = [l11,nn.ReLU(),l22,nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.mlp2 = nn.Sequential(*layers2)

        self.linear_tree = nn.Linear(hidden_size * 2, hidden_size)
        self.linear_label = nn.Linear(hidden_size * 2, self.rel_embedding_size)
        self.max_size = max_sent_len
        self.linear_dep = nn.Linear(hidden_size, 100).to(device=constants.device)
        self.linear_head = nn.Linear(hidden_size, 100).to(device=constants.device)
        self.biaffine = Biaffine(100, 100)
        self.biaffineChart = BiaffineChart(100, 100)

        self.linear_labels_dep = nn.Linear(hidden_size, 100).to(device=constants.device)
        self.linear_labels_head = nn.Linear(hidden_size, 100).to(device=constants.device)
        self.bilinear_label = Bilinear(100, 100, self.num_rels)

        self.weight_matrix = nn.MultiheadAttention(868, num_heads=1, dropout=dropout).to(device=constants.device)
        self.root_selector = nn.LSTM(
            868, 1, 1, dropout=(dropout if 1 > 1 else 0),
            batch_first=True, bidirectional=False).to(device=constants.device)

        self.lstm = nn.LSTM(868, hidden_size, 2, batch_first=True, bidirectional=False).to(device=constants.device)
        self.lstm_tree = nn.LSTM(hidden_size, hidden_size, 1, batch_first=False, bidirectional=False).to(device=constants.device)

    def init_pending(self, n):
        pending = []
        for i in range(n):
            k, j, h = i, i + 1, i
            pending.append(Item(k, j, h, k, k))
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

    def window(self,i,n):
        if i-1 >= 0 and i+2 < n:
            return [i-1,i,i+1,i+2]
        elif i-1 < 0 and i+2 < n:
            return [None,i, i+1, i+2]
        elif i-1 >= 0 and i+2>n:
            return [i-1,i,i+1,None]
        else:
            return [None,i,i+1,None]

    def possible_arcs_simple(self, words,remaining,oracle_arc):
        scores = []
        z = torch.zeros_like(words[0,:]).to(device=constants.device)
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
        for i in range(n-1):
            window = self.window(i, n)
            rep = torch.cat([words[i,:] if i is not None else z for i in window],dim=-1).to(device=constants.device)
            pair1 = (remaining[i],remaining[i+1])
            pair2 = (remaining[i+1],remaining[i])
            if pair1 == gold:
                oracle_ind = i
            if pair2 == gold:
                oracle_ind = i+1
            score = self.mlp2(rep)
            print_red(score)
            scores.append(score[0])
            scores.append(score[1])
            arcs.append((remaining[i],remaining[i+1]))
            arcs.append((remaining[i+1],remaining[i]))
        #scores = torch.stack(scores,dim=-1)
        scores = torch.tensor(scores).to(device=constants.device)
        best_score_ind = torch.argmax(scores,dim=0)
        return scores, torch.tensor([oracle_ind],dtype=torch.long).to(device=constants.device), arcs[best_score_ind]

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
            return remaining, (torch.tensor(predicted_arc[0]),torch.tensor(predicted_arc[1]))


    def possible_arcs(self, words, pending,popped, hypergraph, history, oracle_arc):
        all_options = []
        all_items = []
        arcs = []
        # #print("pending len {}".format(len(pending)))
        item_index2_pending_index = {}

        counter_all_items = 0
        for iter, item in enumerate(pending):
            if item.l in hypergraph.bucket or item.r in hypergraph.bucket:
                continue
            hypergraph = hypergraph.update_chart(item)
            # ###print(colored("Item {} should be added".format(item),"red"))
            # for tang in hypergraph.chart:
            #    ###print(colored(tang,"red"))
            possible_arcs = hypergraph.outgoing(item,popped)
            for tree in possible_arcs:
                item_index2_pending_index[counter_all_items] = iter
                counter_all_items += 1
                #if (tree.i, tree.j, tree.h) in history:
                #    continue
                all_items.append(tree)
                all_options.append(torch.tensor(
                    [[tree.l.i, tree.l.j, tree.l.h], [tree.r.i, tree.r.j, tree.r.h], [tree.i, tree.j, tree.h]]
                ).to(device=constants.device))
                arcs.append((tree.h, tree.r.h if tree.r.h != tree.h else tree.l.h))
            if not hypergraph.axiom(item):
                hypergraph = hypergraph.delete_from_chart(item)
            ###print(colored("Item {} should be deleted".format(item), "blue"))
            # for tang in hypergraph.chart:
            #    ###print(colored(tang, "blue"))
        #print_red(len(arcs))
        #print_blue(len(list(set(arcs))))

        triples = torch.stack(all_options)
        scores = []
        gold_index = 0
        for item in all_items:
            i,j,h = item.i, item.j, item.h
            if isinstance(item.l,Item):
                l_score = item.l.score
            else:
                l_score = 1
            if isinstance(item.r,Item):
                r_score = item.r.score
            else:
                r_score = 1
            if item in hypergraph.scored_items:
                score = item.score
            else:
                if j >=len(words):
                    j = len(words)-1
                features_derived = torch.cat([words[i,:],words[j,:],words[h,:]],dim=-1).to(device=constants.device)
                score = self.mlp(features_derived)*l_score*r_score
                item.update_score(score)
                hypergraph.score_item(item)
            #print_green(item.score)
            scores.append(score)

        for i, (u, v) in enumerate(arcs):
            if (u, v) == oracle_arc:
                gold_index = i
        #    w = torch.cat([words[u], words[v]], dim=-1).to(device=constants.device)
        #    s = self.mlp(w)
        #    scores.append(s)
        scores = torch.tensor(scores).to(device=constants.device)
        winner = torch.argmax(scores)
        # pending.pop(item_index2_pending_index[winner.item()])
        winner_item = all_items[winner]
        # if not self.training:
        #    pending.append(winner_item)
        return triples[winner], winner_item, arcs[winner], scores, pending, torch.tensor([gold_index],dtype=torch.long)\
            .to(device=constants.device), hypergraph

    def take_step(self, transitions, hypergraph, oracle_agenda, pred_item, pending):
        if self.training:
            left = transitions[0]
            right = transitions[1]
            derived = transitions[2]
            di = oracle_agenda[(derived[0].item(), derived[1].item(), derived[2].item())]
            # print(colored("oracle {}".format(di),"yellow"))
            pending.append(di)
            hypergraph = hypergraph.update_chart(di)
            hypergraph = hypergraph.add_bucket(di)
            if derived[2] != right[2]:
                # right is child and is popped
                made_arc = (derived[2], right[2])
            else:
                made_arc = (derived[2], left[2])

        else:

            hypergraph = hypergraph.update_chart(pred_item)
            hypergraph = hypergraph.add_bucket(pred_item)
            pending.append(pred_item)
            if isinstance(pred_item.r, Item):
                right_head = pred_item.r.h
                right_key = (pred_item.r.i, pred_item.r.j, pred_item.r.h)
            else:
                right_head = pred_item.r
                right_key = (pred_item.r, pred_item.r + 1, pred_item.r)
            if isinstance(pred_item.l, Item):
                left_head = pred_item.l.h
                left_key = (pred_item.l.i, pred_item.l.j, pred_item.l.h)
            else:
                left_head = pred_item.l
                left_key = (pred_item.l, pred_item.l + 1, pred_item.l)

            if pred_item.h != right_head:
                made_arc = (torch.tensor(pred_item.h), torch.tensor(right_head))
            else:
                made_arc = (torch.tensor(pred_item.h), torch.tensor(left_head))

        return hypergraph, made_arc, pending

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

    def margin_loss_stejjj(self, oracle_action, scores, map):

        left = oracle_action[0]
        right = oracle_action[1]
        derived = oracle_action[2]
        correct_head = derived[2]
        correct_mod = right[2] if right[2] != derived[2] else left[2]
        score_incorrect = torch.max(scores)

        # try:
        correct_head = map[correct_head.item()]
        # except:
        #    # correct head already has a head itself (bottom up parsing)
        #    return nn.ReLU()(1-score_incorrect)
        # try:
        correct_mod = map[correct_mod.item()]
        # except:
        #    # mod already has a head
        #    return nn.ReLU()(1-score_incorrect)
        score_correct = scores[correct_head, correct_mod]

        loss = nn.ReLU()(1 - score_correct + score_incorrect)
        ##print(loss)
        return loss

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
            heads[v] = u.item()
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
        left_reps = x[list(left_children), :]#.unsqueeze(1)
        right_reps = x[list(right_children), :].unsqueeze(1)
        left_reps = torch.flip(left_reps, dims=[0, 1]).unsqueeze(1)
        # print_green(left_reps.shape)
        # print_green(right_reps.shape)
        _, (lh, _) = self.lstm_tree(left_reps)
        _, (rh, _) = self.lstm_tree(right_reps)
        # print_yellow(lh.shape)
        # print_yellow(rh.shape)
        c = torch.cat([lh, rh], dim=-1).to(device=constants.device)
        c = self.linear_tree(c)
        return c

    def forward(self, x, transitions, relations, map, heads, rels):

        # average of last 4 hidden layers
        with torch.no_grad():
            out = self.bert(x[0].to(device=constants.device))[2]
            x_emb = torch.stack(out[-8:]).mean(0)

        heads_batch = torch.ones((x_emb.shape[0], heads.shape[1])).to(device=constants.device)# * -1

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
        initial_weights_logits = self.get_head_logits(h_t, sent_lens)
        tree_loss = 0
        for i in range(initial_weights_logits.shape[0]):

            curr_sentence_length = sent_lens[i]

            curr_init_weights = initial_weights_logits[i]
            curr_init_weights = curr_init_weights[:curr_sentence_length + 1, :curr_sentence_length + 1]
            # curr_init_weights = torch.exp(curr_init_weights)

            oracle_hypergraph = transitions[i]
            oracle_hypergraph = oracle_hypergraph[oracle_hypergraph[:, 0, 0] != -1, :, :]
            oracle_agenda = self.init_agenda_oracle(oracle_hypergraph)

            s = h_t[i, :curr_sentence_length, :]
            s_ind = list(range(curr_sentence_length))
            chart = Chart()
            pending = self.init_pending(curr_sentence_length)
            hypergraph = self.hypergraph(curr_sentence_length, chart)

            trees = torch.exp(curr_init_weights)
            arcs = []
            history = defaultdict(lambda: 0)
            loss = 0
            popped = []
            """
            steps: 1. compute tree_rep of gold trajectories
                   2. parse this sentence
                   3. minimize between gold and computed
            FAILED....
            """

            """
            steps: 1. tree-lstm
                   2. score with mlp
                   
            """

            # 1. compute tree
            # gold_tree = self.compute_tree(s, heads[i, :curr_sentence_length], rels[i, :curr_sentence_length])
            # s_wrong = s.clone().detach()

            right_children = {i: [i] for i in range(curr_sentence_length)}
            left_children = {i: [i] for i in range(curr_sentence_length)}
            remaining = list(range(curr_sentence_length))
            current_representations = s.clone()
            h_tree = self.linear_head(s.unsqueeze(0).to(device=constants.device))
            d_tree = self.linear_dep(s.unsqueeze(0).to(device=constants.device))
            # wrong_right_children = {i:[i] for i in range(curr_sentence_length)}
            # wrong_left_children = {i:[i] for i in range(curr_sentence_length)}
            # wrong_current_representations = s.clone()
            for step in range(len(oracle_hypergraph)):
                # print(s_ind)
                # good luck with this lol
                # ind_map = {ind: i for i, ind in enumerate(s_ind)}
                # map_ind = {i: ind for i, ind in enumerate(s_ind)}
                # h_tree = self.linear_head(s.unsqueeze(0).to(device=constants.device))
                # d_tree = self.linear_dep(s.unsqueeze(0).to(device=constants.device))
                # trees = torch.exp(self.biaffine(h_tree,d_tree))
                # trees = trees.squeeze(0)
                # scores_orig = trees
                # all_picks = []
                # for en, item in enumerate(pending):
                #    picks = hypergraph.new_trees(item, popped)
                #    all_picks.append(picks)
                #    # scores = hypergraph.make_legal(scores,picks)
                #    # #print(colored("{}".format(scores),"blue"))
                # picks = [item for sublist in all_picks for item in sublist]
                # scores, scores_all = self.biaffineChart(h_tree, d_tree, picks, hypergraph, ind_map)
                # scores = hypergraph.make_legal(scores_orig,picks)
                ##print(scores).
                # item_to_make =self.pick_best(scores,hypergraph)
                # mx = torch.amax(scores, (0, 1))
                ##print(colored("max elem {}".format(mx), "red"))
                # mx_ind = (torch.eq(scores, mx)).nonzero(as_tuple=True)
                ##print(colored("max ind {}".format(mx_ind), "red"))

                # this index to items
                # if len(mx_ind[0]) > 1 or len(mx_ind[1]) > 1:
                #    # ind_x = map_ind[mx_ind[0][0].item()]
                #    ind_x = mx_ind[0][0].item()
                #    # ind_y = map_ind[mx_ind[1][0].item()]
                #    ind_y = mx_ind[1][0].item()
                #    select = 1
                #    while ind_x == ind_y:
                #        # ind_y = map_ind[mx_ind[1][1].item()]
                #        ind_y = mx_ind[1][1].item()
                #        select += 1
                # else:
                #    # ind_x = map_ind[mx_ind[0].item()]
                #    ind_x = mx_ind[0].item()
                #    # ind_y = map_ind[mx_ind[1].item()]
                #    ind_y = mx_ind[1].item()
                # key = (ind_x, ind_y)
                # item_to_make = hypergraph.locator[key]
                #item_tensor, item_to_make, arc_made, \
                #scores, pending, gold_index,hypergraph = self.possible_arcs(current_representations, pending,popped, hypergraph,
                #                                                history, oracle_hypergraph[step])
                #print_green(curr_sentence_length)
                #print_blue(scores.shape)
                #print_green(scores)
                scores, oracle_score, predicted_arc = self.possible_arcs_simple(current_representations,
                                                                                remaining,oracle_hypergraph[step])
                remaining, made_arc = self.take_action_simple(predicted_arc,oracle_hypergraph[step],remaining)
                #history[(item_to_make.i, item_to_make.j, item_to_make.h)] = item_to_make

                #hypergraph, made_arc, pending = self.take_step(oracle_hypergraph[step], hypergraph, oracle_agenda,
                #                                               item_to_make, pending)
                #loss += 0.5*nn.CrossEntropyLoss()(scores.unsqueeze(0),gold_index)+ 0.5*nn.ReLU()(1-scores[gold_index]+torch.max(scores))
                #loss += self.margin_loss_step(oracle_hypergraph[step], scores)
                # loss += self.item_oracle_loss_single_step(scores_all, oracle_hypergraph[step])
                if self.training:
                    loss += nn.CrossEntropyLoss()(scores.unsqueeze(0),oracle_score)
                # h = ind_map[made_arc[0].item()]
                h = made_arc[0].item()
                m = made_arc[1].item()
                #else:
                #    h = made_arc[0]
                #    m = made_arc[1]
                # print_blue(made_arc)
                if h < m:
                    # m is a right child
                    right_children[h].append(m)
                else:
                    # m is a left child
                    left_children[h].append(m)

                h_rep = self.tree_lstm(current_representations, left_children[h], right_children[h])
                current_representations[h, :] = h_rep

                # m = ind_map[made_arc[1].item()]
                # h_w = item_to_make.h
                # m_w = item_to_make.i if item_to_make.i != item_to_make.h else item_to_make.j
                # if h_w < m_w:
                #    wrong_right_children[h_w] = wrong_right_children[h_w].append(m_w)
                # else:
                #    wrong_left_children[h_w] = wrong_left_children[h_w].append(m_w)
                # h_wrong_rep = self.tree_lstm(wrong_current_representations,wrong_left_children[h_w],wrong_right_children[h_w])
                # wrong_current_representations[h_w,:] = h_wrong_rep
                # label = self.linear_label(torch.cat([s[h, :], s[m, :]], dim=-1)
                #                          .to(device=constants.device))
                # new_rep = self.tree_representation(s[h, :].to(device=constants.device), s[m, :]
                #                                   .to(device=constants.device), label.to(device=constants.device))
                # if h_w < curr_sentence_length and m_w < curr_sentence_length:
                #    label_wrong = self.linear_label(torch.cat([s_wrong[h_w, :], s_wrong[m_w, :]], dim=-1)
                #                                    .to(device=constants.device))
                #    new_rep_wrong = self.tree_representation(s_wrong[h_w, :].to(device=constants.device), s_wrong[m_w, :]
                #                                       .to(device=constants.device), label_wrong.to(device=constants.device))
                #    s_wrong[h_w, :] = new_rep_wrong.unsqueeze(0)
                # s = s.clone().to(device=constants.device)
                # s[h, :] = new_rep.unsqueeze(0)
                # tmp1 = s.clone().detach().to(device=constants.device)
                # tmp1[h, :] = new_rep.to(device=constants.device)
                # s_ind.remove(made_arc[1].item())
                # tmp = torch.zeros((s.shape[0] - 1, s.shape[1])).to(device=constants.device)
                # tmp[:m, :] = tmp1[:m, :]
                # tmp[m:, :] = tmp1[m + 1:, :]
                # s = tmp
                popped.append(m)
                #remaining.remove(m)
                # s[made_arc[1],:] = torch.zeros(1,1,trees.shape[2]).to(device=constants.device)

                # loss += self.item_oracle_loss_single_step(scores, oracle_hypergraph[step])
                arcs.append(made_arc)

            # made_tree = s[0,:]
            # print(gold_tree.shape)
            # print(s.shape)
            # print(made_tree.shape)
            # tree_loss += nn.MSELoss()(gold_tree, s_wrong)  # self.tree_loss(gold_tree,made_tree)
            # tree_loss += nn.CosineEmbeddingLoss(margin=1.0)(gold_tree, s_wrong,torch.ones(gold_tree.shape[0]).to(device=constants.device))  # self.tree_loss(gold_tree,made_tree)
            pred_heads = self.heads_from_arcs(arcs, curr_sentence_length)
            heads_batch[i, :curr_sentence_length] = pred_heads
            # if self.training:
            #    print_yellow(pred_heads)
            #    print_blue(heads[i])
            loss /= len(oracle_hypergraph)
            batch_loss += loss
        batch_loss /= x_emb.shape[0]
        heads = heads_batch
        # tree_loss /= x_emb.shape[0]
        l_logits = self.get_label_logits(h_t, heads)
        rels_batch = torch.argmax(l_logits, dim=-1)
        #rels_batch = rels_batch.permute(1, 0)
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
