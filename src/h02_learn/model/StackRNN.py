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

    def calculate_weights(self, sentence, agenda, heads):
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
        w = nn.Softmax()(w,dim=-1)
        #w = torch.exp(w)
        for i in range(n):
            k, j, h = i, i + 1, i
            agenda[(k, j, h)] = Item(k, j, h, w[j, k], k, k)

        return w, nn.Softmax()(h_logits,dim=-1), l_logits, agenda

    def run_lstm(self, x, sent_lens):
        # lstm_in = pack_padded_sequence(x, sent_lens, batch_first=True, enforce_sorted=False)
        # print(lstm_in)
        lstm_out, _ = self.lstm(x)
        # h_t, _ = pad_packed_sequence(lstm_out, batch_first=True)
        h_t = self.dropout(lstm_out).contiguous()
        return h_t

    def get_head_logits(self, h_t, sent_lens):
        h_dep = self.dropout(F.relu(self.linear_arc_dep(h_t)))
        h_arc = self.dropout(F.relu(self.linear_arc_head(h_t)))

        h_logits = self.biaffine(h_arc, h_dep)

        # Zero logits for items after sentence length
        for i, sent_len in enumerate(sent_lens):
            h_logits[i, sent_len:, :] = 0
            h_logits[i, :, sent_len:] = 0

        return h_logits

    def decode_weights(self, smt):
        pass

    def forward(self, x, map, heads=None, rels=None):
        sent_lens = (x[0] != 0).sum(-1).to(device=constants.device)
        tags = self.tag_embeddings(x[1].to(device=constants.device))
        # average of last 4 hidden layers
        with torch.no_grad():
            out = self.bert(x[0].to(device=constants.device))[2]
            x_emb = torch.stack(out[-8:]).mean(0)

        heads_batch = torch.ones((x_emb.shape[0], heads.shape[1], heads.shape[1])).to(device=constants.device)
        tree_batch = torch.ones((x_emb.shape[0], heads.shape[1])).to(device=constants.device)
        rels_batch = torch.ones((x_emb.shape[0], heads.shape[1], self.num_rels)).to(device=constants.device)
        heads_batch *= -1
        rels_batch *= -1
        # print(x_emb.shape)
        # print(sent_lens)
        # x_emb = x_emb.permute(1,0,2)
        # print(h_t.shape)
        prob_sum = 0
        for i, sentence in enumerate(x_emb):
            # initialize a parser
            mapping = map[i, map[i] != -1]
            tag = self.tag_embeddings(x[1][i][x[1][i] != 0].to(device=constants.device))

            s = self.get_bert_embeddings(mapping, sentence, tag)
            curr_sentence_length = s.shape[0]
            s = s[:curr_sentence_length, :]
            # s = torch.cat([s,torch.zeros(1,s.shape[1])],dim=0)
            chart = Chart()
            agenda = Agenda()
            w, h_logits, l_logits, agenda = self.calculate_weights(s, agenda, heads)
            heads_batch[i, :curr_sentence_length, :curr_sentence_length] = h_logits
            rels_batch[i, :curr_sentence_length, :] = l_logits
            hypergraph = self.hypergraph(curr_sentence_length, chart, w, self.mlp, s)
            bucket = defaultdict(lambda: 0)
            pops = 0
            popped = defaultdict(lambda: 0)
            while not agenda.empty():
                # for step in range(len(labeled_transitions)):
                item = agenda.pop()
                if self.prune:
                    if item.l in bucket or item.r in bucket:
                        # pruned
                        continue
                    bucket[item.l] += 1
                    bucket[item.r] += 1
                chart[item] = item
                popped[(item.i, item.j, item.h)] = item
                pops += 1
                for item_new in hypergraph.outgoing(item):
                    agenda[(item_new.i, item_new.j, item_new.h)] = item_new

            tree_batch[i,:curr_sentence_length] = hypergraph.best_path()

        batch_loss = self.loss(heads_batch, rels_batch, heads, rels)
        rels_batch = torch.argmax(rels_batch,dim=-1)
        return batch_loss, tree_batch, rels_batch

    def get_label_logits(self, h_t, head):
        l_dep = self.dropout(F.relu(self.linear_labels_dep(h_t)))
        l_head = self.dropout(F.relu(self.linear_labels_head(h_t)))
        if self.training:
            assert head is not None, 'During training head should not be None'

        # l_head = l_head.gather(dim=1, index=head.unsqueeze(2).expand(l_head.size()))
        l_logits = self.bilinear_label(l_dep.permute(1, 0, 2), l_head.permute(1, 0, 2))
        return l_logits

    def loss(self, h_logits, l_logits, heads, rels):
        criterion_h = nn.CrossEntropyLoss(ignore_index=-1).to(device=constants.device)
        criterion_l = nn.CrossEntropyLoss(ignore_index=0).to(device=constants.device)
        loss = criterion_h(h_logits.reshape(-1, h_logits.shape[-1]), heads.reshape(-1))
        loss += criterion_l(l_logits.reshape(-1, l_logits.shape[-1]), rels.reshape(-1))
        return loss


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
        # print(x[1])
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
        # print(x[1])
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
