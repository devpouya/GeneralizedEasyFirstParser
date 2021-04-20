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


class ChartParser(BertParser):
    def __init__(self, vocabs, embedding_size, rel_embedding_size, batch_size, hypergraph,
                 dropout=0.33, beam_size=10, max_sent_len=190, easy_first=False):
        super().__init__(vocabs, embedding_size, rel_embedding_size, batch_size, dropout=dropout,
                         beam_size=beam_size)

        self.hypergraph = hypergraph
        weight_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=8)
        self.weight_encoder = nn.TransformerEncoder(weight_encoder_layer, num_layers=2)
        self.prune = True  # easy_first
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Linear(self.embedding_size * 2 + 200, 1)

        self.linear_tree = nn.Linear(500 * 2 + self.rel_embedding_size, 500)
        self.linear_label = nn.Linear(500 * 2, self.rel_embedding_size)
        self.max_size = max_sent_len
        self.linear_dep = nn.Linear(500, 100).to(device=constants.device)
        self.linear_head = nn.Linear(500, 100).to(device=constants.device)
        self.biaffine = Biaffine(100, 100)
        self.biaffineChart = BiaffineChart(100, 100)

        self.linear_labels_dep = nn.Linear(500, 100).to(device=constants.device)
        self.linear_labels_head = nn.Linear(500, 100).to(device=constants.device)
        self.bilinear_label = Bilinear(100, 100, self.num_rels)

        self.weight_matrix = nn.MultiheadAttention(868, num_heads=1, dropout=dropout).to(device=constants.device)
        self.root_selector = nn.LSTM(
            868, 1, 1, dropout=(dropout if 1 > 1 else 0),
            batch_first=True, bidirectional=False).to(device=constants.device)

        self.lstm = nn.LSTM(868, 500, 2, batch_first=True, bidirectional=False).to(device=constants.device)

    def init_pending(self, n):
        pending = []
        for i in range(n):
            k, j, h = i, i + 1, i
            pending.append(Item(k, j, h, k, k))
        return pending

    def run_lstm(self, x, sent_lens):
        lstm_in = pack_padded_sequence(x, sent_lens.to(device=torch.device("cpu")), batch_first=True, enforce_sorted=False)
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


    def possible_arcs(self, words, pending, hypergraph, history):
        all_options = []
        all_items = []
        arcs = []
        # #print("pending len {}".format(len(pending)))
        item_index2_pending_index = {}

        counter_all_items = 0
        for iter, item in enumerate(pending):
            if item.l in hypergraph.bucket or item.r in hypergraph.bucket:
                ####print(colored("PRUNE {}".format(item),"red"))
                continue
            ####print(colored(item, "blue"))
            hypergraph = hypergraph.update_chart(item)
            # ###print(colored("Item {} should be added".format(item),"red"))
            # for tang in hypergraph.chart:
            #    ###print(colored(tang,"red"))
            possible_arcs = hypergraph.outgoing(item)
            for tree in possible_arcs:
                item_index2_pending_index[counter_all_items] = iter
                counter_all_items += 1
                if (tree.i, tree.j, tree.h) in history:
                    continue
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

        triples = torch.stack(all_options)
        scores = []
        for (u, v) in arcs:
            w = torch.cat([words[u], words[v]], dim=-1).to(device=constants.device)
            s = self.mlp(w)
            scores.append(s)
        scores = torch.tensor(scores).to(device=constants.device)
        winner = torch.argmax(scores)
        # pending.pop(item_index2_pending_index[winner.item()])
        winner_item = all_items[winner]
        # if not self.training:
        #    pending.append(winner_item)
        return triples[winner], winner_item, arcs[winner], scores, pending

    def take_step(self, transitions, hypergraph, oracle_agenda, pred_item, pending):
        if self.training:
            left = transitions[0]
            right = transitions[1]
            derived = transitions[2]
            di = oracle_agenda[(derived[0].item(), derived[1].item(), derived[2].item())]
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

    def margin_loss_step(self, oracle_action, scores,map):

        left = oracle_action[0]
        right = oracle_action[1]
        derived = oracle_action[2]
        correct_head = derived[2]
        correct_mod = right[2] if right[2] != derived[2] else left[2]
        score_incorrect = torch.max(scores)

        try:
            correct_head = map[correct_head.item()]
        except:
            # correct head already has a head itself (bottom up parsing)
            return nn.ReLU()(1-score_incorrect)
        try:
            correct_mod = map[correct_mod.item()]
        except:
            # mod already has a head
            return nn.ReLU()(1-score_incorrect)
        score_correct = scores[correct_head,correct_mod]

        loss = nn.ReLU()(1-score_correct+score_incorrect)
        ##print(loss)
        return loss

    def margin_loss_stehhhp(self, words, oracle_action, score_incorrect):
        # correct action is the oracle action for now
        left = oracle_action[0]
        right = oracle_action[1]
        derived = oracle_action[2]
        correct_head = derived[2]
        correct_mod = right[2] if right[2] != derived[2] else left[2]
        score_incorrect = nn.Softmax(dim=-1)(score_incorrect)
        score_correct = self.mlp(
            torch.cat([words[correct_head], words[correct_mod]], dim=-1).to(device=constants.device))

        return nn.ReLU()(1 - score_correct + torch.max(score_incorrect))

    def heads_from_arcs(self, arcs, sent_len):
        heads = [0] * sent_len
        for (u, v) in arcs:
            heads[v] = u.item()
        return torch.tensor(heads).to(device=constants.device)



    def forward(self, x, transitions, relations, map, heads, rels):

        # average of last 4 hidden layers
        with torch.no_grad():
            out = self.bert(x[0].to(device=constants.device))[2]
            x_emb = torch.stack(out[-8:]).mean(0)

        heads_batch = torch.ones((x_emb.shape[0], heads.shape[1])).to(device=constants.device)

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

        sent_lens = (x_mapped[:,:,0] != 0).sum(-1).to(device=constants.device)
        max_len = torch.max(sent_lens)
        h_t = self.run_lstm(x_mapped, sent_lens)
        initial_weights_logits = self.get_head_logits(h_t, sent_lens)

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
            for step in range(len(oracle_hypergraph)):
                #print(s_ind)
                # good luck with this lol
                ind_map = {ind:i for i,ind in enumerate(s_ind)}
                map_ind = {i:ind for i, ind in enumerate(s_ind)}
                h_tree = self.linear_head(s.unsqueeze(0))
                d_tree = self.linear_dep(s.unsqueeze(0))
                # trees = torch.exp(self.biaffine(h_tree,d_tree))
                # trees = trees.squeeze(0)
                # scores_orig = trees
                all_picks = []
                for en, item in enumerate(pending):
                    picks = hypergraph.new_trees(item,popped)
                    all_picks.append(picks)
                    # scores = hypergraph.make_legal(scores,picks)
                    # #print(colored("{}".format(scores),"blue"))
                picks = [item for sublist in all_picks for item in sublist]
                scores,scores_all = self.biaffineChart(h_tree, d_tree, picks,hypergraph,ind_map)
                # scores = hypergraph.make_legal(scores_orig,picks)

                ##print(scores)
                #item_to_make =self.pick_best(scores,hypergraph)
                mx = torch.amax(scores, (0, 1))
                ##print(colored("max elem {}".format(mx), "red"))
                mx_ind = (torch.eq(scores, mx)).nonzero(as_tuple=True)
                ##print(colored("max ind {}".format(mx_ind), "red"))

                #this index to items
                if len(mx_ind[0]) > 1 or len(mx_ind[1]) > 1:
                   ind_x = map_ind[mx_ind[0][0].item()]
                   ind_y = map_ind[mx_ind[1][0].item()]
                   select = 1
                   while ind_x == ind_y:
                       ind_y = map_ind[mx_ind[1][1].item()]
                       select+=1

                else:
                   ind_x = map_ind[mx_ind[0].item()]
                   ind_y = map_ind[mx_ind[1].item()]
                key = (ind_x, ind_y)

                item_to_make = hypergraph.locator[key]

                ##print(colored("making {}".format(item_to_make), "yellow"))
                # either make this or score this
                ##print(colored("oracle hyp{}".format(oracle_hypergraph[step]), "green"))
                hypergraph, made_arc, pending = self.take_step(oracle_hypergraph[step], hypergraph, oracle_agenda,
                                                               item_to_make, pending)
                ##print(colored(made_arc,"yellow"))
                ##print(colored(item_to_make,"red"))
                ##print(colored("TRAINING {}".format(self.training),"green"))
                loss += self.margin_loss_step(oracle_hypergraph[step], scores_all,ind_map)

                # item_tensor, item_to_make, arc_made, scores, pending = self.possible_arcs(s, pending, hypergraph,
                #                                                                          history)

                # make tree and replace in trees_matrix
                # ----> TODO
                ##print(s.shape)
                ##print(torch.cat([s[made_arc[0], :], s[made_arc[1], :]], dim=-1).shape)
                h = ind_map[made_arc[0].item()]
                m = ind_map[made_arc[1].item()]
                label = self.linear_label(torch.cat([s[h, :], s[m, :]], dim=-1)
                                          .to(device=constants.device))
                new_rep = self.tree_representation(s[h, :], s[m, :], label)
                tmp1 = s.clone().detach()
                tmp1[h, :] = new_rep
                s_ind.remove(made_arc[1].item())
                tmp = torch.zeros((s.shape[0]-1,s.shape[1]))
                tmp[:m,:] = tmp1[:m,:]
                tmp[m:,:] = tmp1[m+1:,:]
                s = tmp
                popped.append(made_arc[1].item())
                # s[made_arc[1],:] = torch.zeros(1,1,trees.shape[2]).to(device=constants.device)

                history[(item_to_make.i, item_to_make.j, item_to_make.h)] = item_to_make

                #loss += self.item_oracle_loss_single_step(scores, oracle_hypergraph[step])
                arcs.append(made_arc)

            pred_heads = self.heads_from_arcs(arcs, curr_sentence_length)
            heads_batch[i, :curr_sentence_length] = pred_heads
            loss /= len(oracle_hypergraph)
            batch_loss += loss

        heads = heads_batch
        l_logits = self.get_label_logits(h_t, heads)
        rels_batch = torch.argmax(l_logits, dim=-1)
        rels_batch = rels_batch.permute(1, 0)
        batch_loss += self.loss(batch_loss, l_logits, rels)

        return batch_loss, heads_batch, rels_batch

    def item_oracle_loss_single_step(self, scores, oracle_item):

        criterion_head = nn.CrossEntropyLoss().to(device=constants.device)
        criterion_pick = nn.CrossEntropyLoss().to(device=constants.device)
        left = oracle_item[0]
        right = oracle_item[1]
        derived = oracle_item[2]

        head = derived[2]
        mod = right[2] if right[2] != head else left[2]

        # one loss for the probability of assigning the correct head
        # one loss for picking the correct rules
        ##print(scores[mod].shape)
        ##print(head.shape)
        mod_t = torch.zeros(1,dtype=torch.long).to(device=constants.device)
        mod_t[0] = mod
        left_t = torch.zeros(1,dtype=torch.long).to(device=constants.device)
        left_t[0] = left[0]
        right_t = torch.zeros(1,dtype=torch.long).to(device=constants.device)
        right_t[0] = right[1]


        loss = criterion_head(scores[head].unsqueeze(0),mod_t)

        which_to_pick_0 = torch.sum(nn.Softmax(dim=0)(scores),dim=-1).unsqueeze(0)
        which_to_pick_1 = torch.sum(nn.Softmax(dim=-1)(scores),dim=0).unsqueeze(0)
        loss += criterion_pick(which_to_pick_0,left_t)
        loss += criterion_pick(which_to_pick_1,right_t)
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
        #head_int = torch.zeros_like(head,dtype=torch.int64)
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

