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
from .hypergraph import ArcStandard, ArcEager, Hybrid, MH4
from collections import defaultdict

# loool
from termcolor import colored






class NeuralTransitionParser(BertParser):
    def __init__(self, language, vocabs, embedding_size, rel_embedding_size, batch_size,
                 dropout=0.33, transition_system=None):
        super().__init__(language, vocabs, embedding_size, rel_embedding_size, batch_size,
                         dropout=dropout,transition_system=transition_system)

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

    def parser_probabilities(self, parser, labeled_transitions):
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
        if not self.training:# == 'eval':
            action_probabilities = SoftmaxActions(dim=-1, parser=parser, transition_system=self.transition_system,
                                                  temperature=2)(state2)
        else:
            action_probabilities = nn.Softmax(dim=-1)(state2).squeeze(0)
        state2 = self.dropout(F.relu(self.mlp_rel(state1)))
        rel_probabilities = nn.Softmax(dim=-1)(state2).squeeze(0)

        if self.training:# == 'eval':
            best_action = torch.argmax(action_probabilities, dim=-1).item()
            rel = torch.argmax(rel_probabilities, dim=-1).item()  # +1
            rel_ind = torch.tensor([rel], dtype=torch.long).to(device=constants.device)
            rel_embed = self.rel_embeddings(rel_ind).to(device=constants.device)

        return action_probabilities, rel_probabilities, best_action, rel, rel_embed, action_target, rel_target

    def parse_step_arc_standard(self, parser, labeled_transitions):
        action_probabilities, rel_probabilities, best_action, \
        rel, rel_embed, action_target, rel_target = self.parser_probabilities(parser, labeled_transitions)

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

    def parse_step_hybrid(self, parser, labeled_transitions):
        action_probabilities, rel_probabilities, best_action, \
        rel, rel_embed, action_target, rel_target = self.parser_probabilities(parser, labeled_transitions)
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

    def parse_step_arc_eager(self, parser, labeled_transitions):
        action_probabilities, rel_probabilities, best_action, \
        rel, rel_embed, action_target, rel_target = self.parser_probabilities(parser, labeled_transitions)

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

    def parse_step_mh4(self, parser, labeled_transitions):
        # get parser state
        action_probabilities, rel_probabilities, best_action, \
        rel, rel_embed, action_target, rel_target = self.parser_probabilities(parser, labeled_transitions)

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

    def forward(self, x, transitions, relations, map, heads, rels):
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
                                                        labeled_transitions[step]
                                                        )

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

        l_logits = self.get_label_logits(h_t_noeos, heads_batch)
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