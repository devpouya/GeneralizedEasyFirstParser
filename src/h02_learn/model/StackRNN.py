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
from .modules import Biaffine, Bilinear
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
        self.lstm = nn.LSTM(
            embedding_size + 100, 100, 1, dropout=(dropout if 1 > 1 else 0),
            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Linear(self.embedding_size * 2 + 200, 1)

        self.max_size = max_sent_len
        self.linear_dep = nn.Linear(868, 500).to(device=constants.device)
        self.linear_head = nn.Linear(868, 500).to(device=constants.device)
        self.biaffine = Biaffine(500, 500)

        self.linear_labels_dep = nn.Linear(868, 100).to(device=constants.device)
        self.linear_labels_head = nn.Linear(868, 100).to(device=constants.device)
        self.bilinear_label = Bilinear(100, 100, self.num_rels)

        self.weight_matrix = nn.MultiheadAttention(868, num_heads=1, dropout=dropout).to(device=constants.device)
        self.root_selector = nn.LSTM(
            868, 1, 1, dropout=(dropout if 1 > 1 else 0),
            batch_first=True, bidirectional=False).to(device=constants.device)

    def calculate_weights_cook(self, sentence, pending, heads):
        n = len(sentence)
        sentence = sentence.unsqueeze(1)
        atn_out, _ = self.weight_matrix(sentence, sentence, sentence)
        # s = torch.cat([sentence, torch.zeros(1, sentence.shape[1])], dim=0)
        # probability of predicting the correct parse from the predicted scores
        pred_dep = self.linear_dep(atn_out)
        pred_head = self.linear_head(atn_out)
        h_logits = self.biaffine(pred_head.permute(1, 0, 2), pred_dep.permute(1, 0, 2))
        h_logits = h_logits.squeeze(0)
        if not self.training:
            heads = h_logits.argmax(-1)
        l_logits = self.get_label_logits(atn_out, heads)
        w = h_logits
        w = w.squeeze(0)
        root_scores, _ = self.root_selector(sentence)
        root_scores = root_scores.squeeze(1)
        w = torch.cat([w, root_scores], dim=-1)
        bottom_row = torch.ones((w.shape[1], 1)).to(device=constants.device)
        bottom_row[:-1, :] = root_scores * -1
        w = torch.cat([w, bottom_row.transpose(1, 0)], dim=0)
        w = nn.Softmax(dim=-1)(w)
        # w = torch.exp(w)
        for i in range(n):
            k, j, h = i, i + 1, i
            pending[(k, j, h)] = Item(k, j, h, w[j, k], k, k)

        return w, nn.Softmax(dim=-1)(h_logits), l_logits, pending

    def calculate_weights(self, sentence, pending):
        n = len(sentence)
        sentence = sentence.unsqueeze(1)
        atn_out, w = self.weight_matrix(sentence, sentence, sentence)
        # s = torch.cat([sentence, torch.zeros(1, sentence.shape[1])], dim=0)
        # probability of predicting the correct parse from the predicted scores
        w = w.squeeze(0)
        root_scores, _ = self.root_selector(sentence)
        root_scores = root_scores.squeeze(1)
        w = torch.cat([w, root_scores], dim=-1)
        bottom_row = torch.ones((w.shape[1], 1)).to(device=constants.device)
        bottom_row[:-1, :] = root_scores * -1
        w = torch.cat([w, bottom_row.transpose(1, 0)], dim=0)
        w = nn.Softmax(dim=-1)(w)
        # w = torch.exp(w)
        for i in range(n):
            k, j, h = i, i + 1, i
            pending.append(Item(k, j, h, k, k))

        return atn_out, w, pending

    def run_lstm(self, x, sent_lens):
        # lstm_in = pack_padded_sequence(x, sent_lens, batch_first=True, enforce_sorted=False)
        # ##print(lstm_in)
        lstm_out, _ = self.lstm(x)
        # h_t, _ = pad_packed_sequence(lstm_out, batch_first=True)
        h_t = self.dropout(lstm_out).contiguous()
        return h_t

    def get_head_logits(self, h_t, sent_lens):
        h_dep = self.dropout(F.relu(self.linear_arc_dep(h_t)))
        h_arc = self.dropout(F.relu(self.linear_arc_head(h_t)))

        h_logits = self.biaffine(h_arc, h_dep)

        # Zero logits for items after sentence length
        # for i, sent_len in enumerate(sent_lens):
        #    h_logits[i, sent_len:, :] = 0
        #    h_logits[i, :, sent_len:] = 0

        return h_logits

    def decode_weights(self, smt):
        pass

    def possible_arcs(self, words, pending, hypergraph, history):
        all_options = []
        all_items = []
        arcs = []
        #print("pending len {}".format(len(pending)))
        item_index2_pending_index = {}

        counter_all_items = 0
        for iter, item in enumerate(pending):
            if item.l in hypergraph.bucket or item.r in hypergraph.bucket:
                ###print(colored("PRUNE {}".format(item),"red"))
                continue
            ###print(colored(item, "blue"))
            hypergraph = hypergraph.update_chart(item)
            # ##print(colored("Item {} should be added".format(item),"red"))
            # for tang in hypergraph.chart:
            #    ##print(colored(tang,"red"))
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
            ##print(colored("Item {} should be deleted".format(item), "blue"))
            # for tang in hypergraph.chart:
            #    ##print(colored(tang, "blue"))

        triples = torch.stack(all_options)
        scores = []
        for (u, v) in arcs:
            w = torch.cat([words[u], words[v]], dim=-1).to(device=constants.device)
            s = self.mlp(w)
            scores.append(s)
        scores = torch.tensor(scores).to(device=constants.device)
        winner = torch.argmax(scores)
        #pending.pop(item_index2_pending_index[winner.item()])
        winner_item = all_items[winner]
        #if not self.training:
        #    pending.append(winner_item)
        return triples[winner], winner_item, arcs[winner], scores, pending

    def take_step(self, transitions, hypergraph, oracle_agenda, pred_item,pending):
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

        return hypergraph, made_arc,pending

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
        # ##print("cherry pies")
        # for item in pending.values():
        #    ##print(item)
        return pending

    def margin_loss_step(self, words, oracle_action, score_incorrect):
        # correct action is the oracle action for now
        left = oracle_action[0]
        right = oracle_action[1]
        derived = oracle_action[2]
        correct_head = derived[2]
        correct_mod = right[2] if right[2] != derived[2] else left[2]

        score_correct = self.mlp(
            torch.cat([words[correct_head], words[correct_mod]], dim=-1).to(device=constants.device))
        return nn.ReLU()(1 - score_correct + torch.max(score_incorrect))

    def heads_from_arcs(self, arcs, sent_len):
        heads = [0] * sent_len
        for (u, v) in arcs:
            heads[v] = u.item()
        return torch.tensor(heads).to(device=constants.device)

    def forward(self, x, transitions, relations, map, heads, rels):
        sent_lens = (x[0] != 0).sum(-1).to(device=constants.device)
        tags = self.tag_embeddings(x[1].to(device=constants.device))
        transit_lens = (transitions != -1)#.sum(-1).to(device=constants.device)
        # average of last 4 hidden layers
        with torch.no_grad():
            out = self.bert(x[0].to(device=constants.device))[2]
            x_emb = torch.stack(out[-8:]).mean(0)
        # ##print(transitions)
        heads_batch = torch.ones((x_emb.shape[0], heads.shape[1])).to(device=constants.device)
        tree_batch = torch.ones((x_emb.shape[0], heads.shape[1])).to(device=constants.device)
        rels_batch = torch.ones((x_emb.shape[0], heads.shape[1], self.num_rels)).to(device=constants.device)
        # heads_batch *= -1
        # rels_batch *= -1
        # ##print(x_emb.shape)
        # ##print(sent_lens)
        # x_emb = x_emb.permute(1,0,2)
        # ##print(h_t.shape)
        prob_sum = 0
        batch_loss = 0
        h_t = torch.zeros((x_emb.shape[0], heads.shape[1], 868)).to(device=constants.device)
        for i, sentence in enumerate(x_emb):
            # initialize a parser
            mapping = map[i, map[i] != -1]
            tag = self.tag_embeddings(x[1][i][x[1][i] != 0].to(device=constants.device))

            s = self.get_bert_embeddings(mapping, sentence, tag)
            curr_sentence_length = s.shape[0]
            s = s[:curr_sentence_length, :]
            oracle_hypergraph = transitions[i]
            oracle_hypergraph = oracle_hypergraph[oracle_hypergraph[:,0,0]!=-1,:,:]
            oracle_agenda = self.init_agenda_oracle(oracle_hypergraph)

            # s = torch.cat([s,torch.zeros(1,s.shape[1])],dim=0)

            chart = Chart()
            # pending = defaultdict(lambda: 0)  # Agenda()
            pending = []  # defaultdict(lambda: 0)  # Agenda()

            atn_out, w, pending = self.calculate_weights(s, pending)
            h_t[i, :curr_sentence_length, :] = atn_out.squeeze(1)
            # ##print(atn_out.shape)
            # heads_batch[i, :curr_sentence_length, :curr_sentence_length] = h_logits
            # rels_batch[i, :curr_sentence_length, :] = l_logits
            hypergraph = self.hypergraph(curr_sentence_length, chart, self.mlp, s)

            arcs = []
            history = defaultdict(lambda: 0)
            loss = 0
            for step in range(len(oracle_hypergraph)):
                ##print(colored("HYPERGRAPH CHART", "yellow"))
                #for item in hypergraph.chart:
                #    #print(colored("ITEM {}".format(item), "yellow"))
                item_tensor, item_to_make, arc_made, scores, pending = self.possible_arcs(s, pending, hypergraph,
                                                                                          history)
                ##print(colored("timen tensor {}".format(item_to_make), "green"))
                history[(item_to_make.i, item_to_make.j, item_to_make.h)] = item_to_make
                hypergraph, made_arc,pending = self.take_step(oracle_hypergraph[step], hypergraph, oracle_agenda,
                                                              item_to_make,pending)

                loss += self.margin_loss_step(s, oracle_hypergraph[step], scores)
                arcs.append(made_arc)
            pred_heads = self.heads_from_arcs(arcs, curr_sentence_length)
            ##print("predheads {}".format(pred_heads))
            ##print("realheads {}".format(heads))
            heads_batch[i, :curr_sentence_length] = pred_heads
            loss /= len(oracle_hypergraph)
            batch_loss += loss

        if not self.training:
            heads = heads_batch
        l_logits = self.get_label_logits(h_t, heads)
        rels_batch = torch.argmax(l_logits, dim=-1)
        rels_batch = rels_batch.permute(1, 0)
        batch_loss += self.loss(batch_loss, l_logits, rels)

        return batch_loss, heads_batch, rels_batch

    def get_label_logits(self, h_t, head):
        ##print(h_t.shape)
        ##print(head.shape)
        l_dep = self.dropout(F.relu(self.linear_labels_dep(h_t)))
        l_head = self.dropout(F.relu(self.linear_labels_head(h_t)))
        if self.training:
            assert head is not None, 'During training head should not be None'
        #l_head = l_head.gather(dim=1, index=head.unsqueeze(2).expand(l_head.size()))

        ##print(l_dep.shape)
        ##print(l_head.shape)
        ghead = torch.tensor(head,dtype=torch.int64).to(device=constants.device)
        #l_head = l_head.gather(dim=1, index=ghead.unsqueeze(2).expand(l_head.size()))
        ##print("came ")
        l_logits = self.bilinear_label(l_dep, l_head)
        ##print("settled")

        return l_logits

    def loss(self, batch_loss, l_logits, rels):
        criterion_l = nn.CrossEntropyLoss(ignore_index=0).to(device=constants.device)
        loss = criterion_l(l_logits.reshape(-1, l_logits.shape[-1]), rels.reshape(-1))
        return loss + batch_loss


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

    def parse_step_easy_first(self, parser, labeled_transitions, mode):
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

    def forward(self, x, transitions, relations, map, mode):
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


class NeuralTransitionParser(BertParser):
    def __init__(self, vocabs, embedding_size, rel_embedding_size, batch_size,
                 dropout=0.33, beam_size=10, transition_system=None):
        super().__init__(vocabs, embedding_size, rel_embedding_size, batch_size,
                         dropout=dropout, beam_size=beam_size, transition_system=transition_system)

        # neural model parameters
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

        # MLP
        if self.transition_system == constants.arc_eager:
            self.mlp_lin1 = nn.Linear(stack_lstm_size * 2,
                                      self.embedding_size).to(device=constants.device)
            self.mlp_lin1_rel = nn.Linear(stack_lstm_size * 2,
                                          self.embedding_size).to(device=constants.device)
        else:
            self.mlp_lin1 = nn.Linear(stack_lstm_size * 2 + self.action_embeddings_size,
                                      self.embedding_size).to(device=constants.device)
            self.mlp_lin1_rel = nn.Linear(stack_lstm_size * 2 + self.action_embeddings_size,
                                          self.embedding_size).to(device=constants.device)

        self.mlp_act = nn.Linear(self.embedding_size, self.num_actions).to(device=constants.device)
        self.mlp_rel = nn.Linear(self.embedding_size, self.num_rels).to(device=constants.device)

        torch.nn.init.xavier_uniform_(self.mlp_lin1.weight)
        torch.nn.init.xavier_uniform_(self.mlp_lin1_rel.weight)
        torch.nn.init.xavier_uniform_(self.mlp_act.weight)
        torch.nn.init.xavier_uniform_(self.mlp_rel.weight)

        self.linear_tree = nn.Linear(self.rel_embedding_size + 2 * stack_lstm_size, stack_lstm_size).to(
            device=constants.device)
        torch.nn.init.xavier_uniform_(self.linear_tree.weight)

        self.stack = StackCell(self.stack_lstm, self.lstm_init_state, self.lstm_init_state, self.dropout,
                               self.empty_initial)
        self.buffer = StackCell(self.buffer_lstm, self.lstm_init_state, self.lstm_init_state, self.dropout,
                                self.empty_initial)
        self.action = StackCell(self.action_lstm, self.lstm_init_state_actions, self.lstm_init_state_actions,
                                self.dropout,
                                self.empty_initial_act)

    def parser_probabilities(self, parser, labeled_transitions, mode):
        if self.transition_system == constants.arc_eager:
            parser_state = torch.cat([self.stack.embedding().reshape(1, self.embedding_size + self.rel_embedding_size),
                                      self.buffer.embedding().reshape(1,
                                                                      self.embedding_size + self.rel_embedding_size)],
                                     dim=-1)
        else:
            parser_state = torch.cat(
                [self.stack.embedding().reshape(1, self.embedding_size + self.rel_embedding_size),
                 self.buffer.embedding().reshape(1, self.embedding_size + self.rel_embedding_size),
                 self.action.embedding().reshape(1, self.action_embeddings_size)], dim=-1)

        best_action = labeled_transitions[0].item()
        rel = labeled_transitions[1]
        action_target = torch.tensor([best_action], dtype=torch.long).to(device=constants.device)
        rel_target = torch.tensor([rel], dtype=torch.long).to(device=constants.device)
        rel_embed = self.rel_embeddings(rel_target).to(device=constants.device)

        state1 = self.dropout(F.relu(self.mlp_lin1(parser_state)))

        state2 = self.dropout(F.relu(self.mlp_act(state1)))
        if mode == 'eval':
            action_probabilities = SoftmaxActions(dim=-1, parser=parser, transition_system=self.transition_system,
                                                  temperature=2)(state2)
        else:
            action_probabilities = nn.Softmax(dim=-1)(state2).squeeze(0)
        state2 = self.dropout(F.relu(self.mlp_rel(state1)))
        rel_probabilities = nn.Softmax(dim=-1)(state2).squeeze(0)

        if mode == 'eval':
            best_action = torch.argmax(action_probabilities, dim=-1).item()
            rel = torch.argmax(rel_probabilities, dim=-1).item()  # +1
            rel_ind = torch.tensor([rel], dtype=torch.long).to(device=constants.device)
            rel_embed = self.rel_embeddings(rel_ind).to(device=constants.device)

        return action_probabilities, rel_probabilities, best_action, rel, rel_embed, action_target, rel_target

    def parse_step_arc_standard(self, parser, labeled_transitions, mode):
        action_probabilities, rel_probabilities, best_action, \
        rel, rel_embed, action_target, rel_target = self.parser_probabilities(parser, labeled_transitions, mode)

        # do the action
        self.action.push(self.action_embeddings(torch.tensor(best_action, dtype=torch.long).to(device=constants.device))
                         .unsqueeze(0).to(device=constants.device))
        if best_action == 0:
            # shift
            self.stack.push(self.buffer.pop())
            parser.shift()
        elif best_action == 1:
            # reduce-l
            ret = parser.reduce_l(rel, rel_embed, self.linear_tree)
            self.stack.pop(-2)
            self.stack.replace(ret.unsqueeze(0))
        elif best_action == 2:
            # reduce-r
            ret = parser.reduce_r(rel, rel_embed, self.linear_tree)
            self.stack.pop()
            self.stack.replace(ret.unsqueeze(0))
        return parser, (action_probabilities, rel_probabilities), (action_target, rel_target)

    def parse_step_hybrid(self, parser, labeled_transitions, mode):
        action_probabilities, rel_probabilities, best_action, \
        rel, rel_embed, action_target, rel_target = self.parser_probabilities(parser, labeled_transitions, mode)
        self.action.push(self.action_embeddings(torch.tensor(best_action, dtype=torch.long).to(device=constants.device))
                         .unsqueeze(0).to(device=constants.device))
        if best_action == 0:
            parser.shift()
            self.stack.push(self.buffer.pop())
        elif best_action == 1:
            ret = parser.left_arc_eager(rel, rel_embed, self.linear_tree)
            self.stack.pop()
            self.buffer.replace(ret.unsqueeze(0))

        elif best_action == 2:
            ret = parser.reduce_r(rel, rel_embed, self.linear_tree)
            self.stack.pop()
            self.stack.replace(ret.unsqueeze(0))

        return parser, (action_probabilities, rel_probabilities), (action_target, rel_target)

    def parse_step_arc_eager(self, parser, labeled_transitions, mode):
        action_probabilities, rel_probabilities, best_action, \
        rel, rel_embed, action_target, rel_target = self.parser_probabilities(parser, labeled_transitions, mode)

        self.action.push(self.action_embeddings(torch.tensor(best_action, dtype=torch.long).to(device=constants.device))
                         .unsqueeze(0).to(device=constants.device))
        if best_action == 0:
            parser.shift()
            self.stack.push(self.buffer.pop())
        elif best_action == 1:
            ret = parser.left_arc_eager(rel, rel_embed, self.linear_tree)
            self.stack.pop()
            self.buffer.replace(ret.unsqueeze(0))
        elif best_action == 2:
            ret = parser.right_arc_eager(rel, rel_embed, self.linear_tree)
            self.stack.replace(ret.unsqueeze(0))
            self.stack.push(self.buffer.pop())
        elif best_action == 3:
            parser.reduce()
            self.stack.pop()
        else:
            self.stack.pop()
            elem = parser.stack.pop()
            parser.arcs.append((elem[1], elem[1], rel))

        return parser, (action_probabilities, rel_probabilities), (action_target, rel_target)

    def parse_step_mh4(self, parser, labeled_transitions, mode):
        # get parser state
        action_probabilities, rel_probabilities, best_action, \
        rel, rel_embed, action_target, rel_target = self.parser_probabilities(parser, labeled_transitions, mode)

        # do the action
        self.action.push(self.action_embeddings(torch.tensor(best_action, dtype=torch.long).to(device=constants.device))
                         .unsqueeze(0).to(device=constants.device))
        # do the action
        if best_action == 0:
            # shift
            self.stack.push(self.buffer.pop())
            parser.shift()
        elif best_action == 1:
            # left-arc-eager
            ret = parser.left_arc_eager(rel, rel_embed, self.linear_tree)
            self.stack.pop(-1)
            self.buffer.replace(ret.unsqueeze(0))

        elif best_action == 2:
            # reduce-r
            ret = parser.reduce_r(rel, rel_embed, self.linear_tree)
            self.stack.pop()
            self.stack.replace(ret.unsqueeze(0))
        elif best_action == 3:
            # left-arc-prime
            self.stack.pop()
            # self.stack.pop()
            ret = parser.left_arc_prime(rel, rel_embed, self.linear_tree)
            self.stack.push(ret.unsqueeze(0))
        elif best_action == 4:
            # right-arc-prime
            ret = parser.right_arc_prime(rel, rel_embed, self.linear_tree)
            item = self.stack.pop()
            self.stack.pop()
            self.stack.pop()
            self.stack.push(ret.unsqueeze(0))
            self.stack.push(item)
        elif best_action == 5:
            # left-arc-2
            ret = parser.left_arc_2(rel, rel_embed, self.linear_tree)
            item = self.stack.pop()
            self.stack.pop()
            self.stack.push(item)
            self.buffer.replace(ret.unsqueeze(0))
        elif best_action == 6:
            # right-arc-2
            ret = parser.right_arc_2(rel, rel_embed, self.linear_tree)
            self.stack.pop()
            item = self.stack.pop()
            self.stack.replace(ret.unsqueeze(0))
            self.stack.push(item)
        return parser, (action_probabilities, rel_probabilities), (action_target, rel_target)

    def forward(self, x, transitions, relations, map, mode):
        # sent_lens = (x[0] != 0).sum(-1).to(device=constants.device)
        transit_lens = (transitions != -1).sum(-1).to(device=constants.device)
        # ##print(x[1])
        tags = self.tag_embeddings(x[1].to(device=constants.device))
        # average of last 4 hidden layers
        with torch.no_grad():
            out = self.bert(x[0].to(device=constants.device))[2]
            x_emb = torch.stack(out[-8:]).mean(0)

        probs_action_batch = torch.ones((x_emb.shape[0], transitions.shape[1], self.num_actions), dtype=torch.float).to(
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
            labeled_transitions = self.labeled_action_pairs(transitions[i, :curr_transition_length],
                                                            relations[i, :curr_sentence_length])
            parser = ShiftReduceParser(s, self.rel_embedding_size, self.transition_system)

            for word in reversed(s):
                self.buffer.push(word.unsqueeze(0))

            for step in range(len(labeled_transitions)):
                parser, probs, target = self.parse_step(parser,
                                                        labeled_transitions[step],
                                                        mode)

                (action_probs, rel_probs) = probs
                (action_target, rel_target) = target
                probs_action_batch[i, step, :] = action_probs
                probs_rel_batch[i, step, :] = rel_probs
                targets_action_batch[i, step, :] = action_target
                targets_rel_batch[i, step, :] = rel_target

            heads_batch[i, :curr_sentence_length] = parser.heads_from_arcs()[0]
            rels_batch[i, :curr_sentence_length] = parser.heads_from_arcs()[1]
            self.stack.back_to_init()
            self.buffer.back_to_init()
            self.action.back_to_init()

        batch_loss = self.loss(probs_action_batch, targets_action_batch, probs_rel_batch, targets_rel_batch)

        return batch_loss, heads_batch, rels_batch
