import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

import numpy as np

from utils import constants
from .base import BaseParser
from .modules import Biaffine, Bilinear, StackLSTM
from .word_embedding import WordEmbedding, ActionEmbedding
from ..algorithm.transition_parsers import ShiftReduceParser
from .modules import StackRNN


def get_arcs(word2head):
    arcs = []
    for word in word2head:
        arcs.append((word2head[word], word))

    return arcs


def has_head(node, arcs):
    for (u, v, _) in arcs:
        if v == node:
            return True
    return False


# root = (torch.tensor(1).to(device=constants.device), torch.tensor(1).to(device=constants.device))


class NeuralTransitionParser(BaseParser):
    def __init__(self, vocabs, embedding_size, hidden_size, arc_size, label_size, batch_size,
                 nlayers=3, dropout=0.33, pretrained_embeddings=None, transition_system=None):
        super().__init__()
        # basic parameters
        self.vocabs = vocabs
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.arc_size = arc_size
        self.label_size = label_size
        self.batch_size = batch_size
        self.nlayers = nlayers
        self.dropout_prob = dropout

        # transition system
        self.transition_system = transition_system
        # print(self.transition_system)
        self.actions = transition_system[0]  # [shift, reduce_l, reduce_r]
        self.num_actions = len(self.actions)
        self.action2id = {act: i for i, act in enumerate(self.actions)}
        if self.transition_system == constants.arc_standard:
            self.parse_step = self.parse_step_arc_standard
        elif self.transition_system == constants.arc_eager:
            self.parse_step = self.parse_step_arc_eager
        elif self.transition_system == constants.hybrid:
            self.parse_step = self.parse_step_hybrid
        else:
            raise Exception("A transition system needs to be satisfied")

        # word, tag and action embeddings
        self.word_embeddings, self.tag_embeddings, self.learned_embeddings, self.action_embeddings, self.rel_embeddings = \
            self.create_embeddings(vocabs, pretrained=pretrained_embeddings)

        # self.root = (torch.tensor(1).to(device=constants.device), torch.tensor(1).to(device=constants.device))

        """
            MLP[stack_lstm_output,buffer_lstm_output,action_lstm_output] ---> decide next transition
        """
        _, _, rels = vocabs
        # self.num_actions = 1 + rels.size * 2
        # self.num_actions = 3
        self.num_rels = rels.size + 1  # 0 is no rel (for shift)
        # neural model parameters
        self.dropout = nn.Dropout(self.dropout_prob)
        # lstms
        self.stack_lstm = nn.LSTM(self.embedding_size * 3, self.embedding_size * 3).to(device=constants.device)
        self.buffer_lstm = nn.LSTM(self.embedding_size * 3, self.embedding_size * 3).to(device=constants.device)
        self.action_lstm = nn.LSTM(16, 16).to(device=constants.device)

        # parser state
        # self.parser_state = nn.Parameter(torch.zeros((self.batch_size, self.hidden_size * 3 * 2))).to(
        #    device=constants.device)
        # init params
        input_init = torch.zeros((1, 1, self.embedding_size * 3)).to(
            device=constants.device)
        hidden_init = torch.zeros((1, 1, self.embedding_size * 3)).to(
            device=constants.device)

        input_init_act = torch.zeros((1, 1, 16)).to(
            device=constants.device)
        hidden_init_act = torch.zeros((1, 1, 16)).to(
            device=constants.device)

        self.lstm_init_state = (nn.init.xavier_uniform_(input_init), nn.init.xavier_uniform_(hidden_init))
        self.lstm_init_state_actions = (
            nn.init.xavier_uniform_(input_init_act), nn.init.xavier_uniform_(hidden_init_act))

        self.empty_initial = nn.Parameter(torch.zeros(1, 1, self.embedding_size * 3))
        self.empty_initial_act = nn.Parameter(torch.zeros(1, 1, 16))

        # MLP
        self.mlp_lin1 = nn.Linear(self.embedding_size * 6 + 16,
                                  self.embedding_size * 5).to(device=constants.device)
        self.mlp_lin2 = nn.Linear(self.embedding_size * 5, self.embedding_size * 3).to(device=constants.device)
        self.mlp_lin3 = nn.Linear(self.embedding_size * 3, self.embedding_size).to(device=constants.device)
        #self.mlp_lin4 = nn.Linear(self.embedding_size * 3,self.embedding_size * 2).to(device=constants.device)
        #self.mlp_lin5 = nn.Linear(self.embedding_size * 2,self.embedding_size).to(device=constants.device)
        #self.mlp_lin6 = nn.Linear(self.embedding_size * 2,
        #                          self.embedding_size).to(device=constants.device)

        self.mlp_lin1_rel = nn.Linear(self.embedding_size * 6 + 16,
                                      self.embedding_size * 5).to(device=constants.device)
        self.mlp_lin2_rel = nn.Linear(self.embedding_size * 5, self.embedding_size * 3).to(device=constants.device)
        self.mlp_lin3_rel = nn.Linear(self.embedding_size * 3, self.embedding_size).to(device=constants.device)
        #self.mlp_lin4_rel = nn.Linear(self.embedding_size * 3, self.embedding_size * 2).to(device=constants.device)
        #self.mlp_lin5_rel = nn.Linear(self.embedding_size * 2, self.embedding_size).to(device=constants.device)
        #self.mlp_lin6_rel = nn.Linear(self.embedding_size * 2,
        #                          self.embedding_size).to(device=constants.device)
        # print(self.num_actions)
        self.mlp_act = nn.Linear(self.embedding_size, self.num_actions).to(device=constants.device)
        self.mlp_rel = nn.Linear(self.embedding_size, self.num_rels).to(device=constants.device)
        # self.mlp_both = nn.Linear(self.embedding_size, self.num_rels*2+1).to(device=constants.device)
        torch.nn.init.xavier_uniform_(self.mlp_lin1.weight)
        torch.nn.init.xavier_uniform_(self.mlp_lin2.weight)
        torch.nn.init.xavier_uniform_(self.mlp_lin3.weight)
        #torch.nn.init.xavier_uniform_(self.mlp_lin4.weight)
        #torch.nn.init.xavier_uniform_(self.mlp_lin5.weight)
        #torch.nn.init.xavier_uniform_(self.mlp_lin6.weight)
        torch.nn.init.xavier_uniform_(self.mlp_lin1_rel.weight)
        torch.nn.init.xavier_uniform_(self.mlp_lin2_rel.weight)
        torch.nn.init.xavier_uniform_(self.mlp_lin3_rel.weight)
        ##torch.nn.init.xavier_uniform_(self.mlp_lin4_rel.weight)
        #torch.nn.init.xavier_uniform_(self.mlp_lin5_rel.weight)
        #torch.nn.init.xavier_uniform_(self.mlp_lin6_rel.weight)
        torch.nn.init.xavier_uniform_(self.mlp_act.weight)
        torch.nn.init.xavier_uniform_(self.mlp_rel.weight)

        self.linear_tree = nn.Linear(7 * embedding_size + 16, 3 * embedding_size).to(device=constants.device)
        #self.linear_tree2 = nn.Linear(5 * embedding_size, 3 * embedding_size).to(device=constants.device)
        torch.nn.init.xavier_uniform_(self.linear_tree.weight)
        #torch.nn.init.xavier_uniform_(self.linear_tree2.weight)

        self.stack = StackRNN(self.stack_lstm, self.lstm_init_state, self.lstm_init_state, self.dropout,
                              self.empty_initial)
        self.buffer = StackRNN(self.buffer_lstm, self.lstm_init_state, self.lstm_init_state, self.dropout,
                               self.empty_initial)
        self.action = StackRNN(self.action_lstm, self.lstm_init_state_actions, self.lstm_init_state_actions,
                               self.dropout,
                               self.empty_initial_act)

    def create_embeddings(self, vocabs, pretrained):
        words, tags, rels = vocabs
        word_embeddings = WordEmbedding(words, self.embedding_size, pretrained=pretrained)
        tag_embeddings = nn.Embedding(tags.size, self.embedding_size)
        rel_embeddings = nn.Embedding(rels.size+1, self.embedding_size)

        learned_embeddings = nn.Embedding(words.size, self.embedding_size)
        action_embedding = nn.Embedding(self.num_actions, 16)
        return word_embeddings, tag_embeddings, learned_embeddings, action_embedding, rel_embeddings

    def get_embeddings(self, x):
        return torch.cat([self.word_embeddings(x[0]), self.learned_embeddings(x[0]), self.tag_embeddings(x[1])],
                         dim=-1).to(device=constants.device)

    def get_action_embed(self, act):
        idx = self.action2id[act]
        t = torch.tensor(idx, dtype=torch.long).to(device=constants.device)
        return self.action_embeddings(t).unsqueeze(0).unsqueeze(1).to(device=constants.device)

    def labeled_action_pairs(self, actions, relations):
        labeled_acts = []
        tmp_rels = relations.clone().detach().tolist()

        for act in actions:
            # very fragile!!
            # if act == 0 or act is None or act == 3:
            #    labeled_acts.append((act, -1))
            # elif act is not None:
            #    labeled_acts.append((act, tmp_rels[0]))
            #    tmp_rels.pop(0)

            # slightly better
            if act == 1 or act == 2:
                labeled_acts.append((act, tmp_rels[0]))
                tmp_rels.pop(0)
            else:
                labeled_acts.append((act, -1))

        return labeled_acts

    def parser_probabilities(self):
        parser_state = torch.cat([self.stack.embedding(), self.buffer.embedding(), self.action.embedding()], dim=-1)

        state1 = self.dropout(F.relu(self.mlp_lin1(parser_state)))
        state1 = self.dropout(F.relu(self.mlp_lin2(state1)))
        state1 = self.dropout(F.relu(self.mlp_lin3(state1)))
        #state1 = self.dropout(F.relu(self.mlp_lin4(state1)))
        #state1 = self.dropout(F.relu(self.mlp_lin5(state1)))
        #state1 = self.dropout(F.relu(self.mlp_lin6(state1)))

        action_probabilities = nn.Softmax(dim=-1)(self.mlp_act(state1)).squeeze(0)

        state2 = self.dropout(F.relu(self.mlp_lin1_rel(parser_state)))
        state2 = self.dropout(F.relu(self.mlp_lin2_rel(state2)))
        state2 = self.dropout(F.relu(self.mlp_lin3_rel(state2)))
        #state2 = self.dropout(F.relu(self.mlp_lin4_rel(state2)))
        #state2 = self.dropout(F.relu(self.mlp_lin5_rel(state2)))
        #state2 = self.dropout(F.relu(self.mlp_lin6_rel(state2)))
        rel_probabilities = nn.Softmax(dim=-1)(self.mlp_rel(state2)).squeeze(0)
        # print(rel_probabilities)
        return action_probabilities, rel_probabilities

    def parse_step_arc_standard(self, parser, labeled_transitions, mode):
        # get parser state
        action_probabilities, rel_probabilities = self.parser_probabilities()
        if labeled_transitions is not None:
            best_action = labeled_transitions[0].item()
            rel = labeled_transitions[1]
        else:
            best_action = -2
            rel = 0
        if best_action != -2:
            action_target = torch.tensor([best_action], dtype=torch.long).to(device=constants.device)
        else:
            action_target = torch.tensor([1], dtype=torch.long).to(device=constants.device)

        if best_action == 0 or best_action == -2:
            rel = 0

        rel_target = torch.tensor([rel], dtype=torch.long).to(device=constants.device)
        rel_embed = self.rel_embeddings(rel_target).to(device=constants.device)
        if mode == 'eval':

            if len(parser.stack) < 1:
                # can't left or right
                best_action = torch.argmax(action_probabilities[:, 0], dim=-1).item()

            elif len(parser.buffer) == 1:
                # can't shift
                tmp = action_probabilities.clone().detach().to(device=constants.device)
                tmp[:, 0] = -float('inf')
                best_action = torch.argmax(tmp, dim=-1).item()

            elif len(parser.stack) == 1 and len(parser.buffer) == 0:
                # best_action = torch.argmax(action_probabilities[:, 1], dim=-1).item()
                best_action = -2

            else:
                best_action = torch.argmax(action_probabilities, dim=-1).item()

            rel = torch.argmax(rel_probabilities, dim=-1).item()  # +1
            rel_ind = torch.tensor([rel], dtype=torch.long).to(device=constants.device)
            rel_embed = self.rel_embeddings(rel_ind).to(device=constants.device)

        # do the action
        if best_action == 0:
            # shift
            self.stack.push(self.buffer.pop())
            self.action.push(self.get_action_embed(constants.shift))
            parser.shift()
        elif best_action == 1:

            # reduce-l
            self.stack.pop()
            act_embed = self.get_action_embed(constants.reduce_l)
            self.action.push(act_embed)
            ret = parser.reduce_l(act_embed, rel, rel_embed, self.linear_tree)
            self.buffer.pop()
            self.buffer.push(ret.unsqueeze(0).unsqueeze(1))
            # self.stack.push(ret.unsqueeze(0).unsqueeze(1))

        elif best_action == 2:

            # reduce-r
            # buffer.push(stack.pop())  # not sure, should replace in buffer actually...
            act_embed = self.get_action_embed(constants.reduce_l)
            self.action.push(act_embed)
            self.stack.pop()
            ret = parser.reduce_r(act_embed, rel, rel_embed, self.linear_tree)
            self.buffer.pop()
            self.buffer.push(ret.unsqueeze(0).unsqueeze(1))
            # self.buffer.push_first(ret,self.stack.s[-1])
            # self.stack.push(ret.unsqueeze(0).unsqueeze(1))
        else:
            self.stack.pop()
            self.action.push(self.get_action_embed(constants.reduce_l))

            elem = parser.stack.pop()
            parser.arcs.append((elem[1], elem[1], rel))

        return parser, (action_probabilities, rel_probabilities), (action_target, rel_target)

    def parse_step_arc_eager(self, parser, labeled_transitions, mode):
        action_probabilities, rel_probabilities = self.parser_probabilities()
        if labeled_transitions is not None:
            best_action = labeled_transitions[0].item()
            rel = labeled_transitions[1]
        else:
            best_action = -2
            rel = 0

        if best_action != -2:
            action_target = torch.tensor([best_action], dtype=torch.long).to(device=constants.device)
        else:
            action_target = torch.tensor([3], dtype=torch.long).to(device=constants.device)

        if best_action == 0 or best_action == -2 or best_action == 3:
            rel = 0

        rel_target = torch.tensor([rel], dtype=torch.long).to(device=constants.device)
        rel_embed = self.rel_embeddings(rel_target).to(device=constants.device)

        if mode == 'eval':

            if len(parser.stack) < 1:
                # can only shift
                best_action = torch.argmax(action_probabilities[:, 0], dim=-1).item()


            elif len(parser.buffer) == 1:
                # can't shift
                tmp = action_probabilities.clone().detach().to(device=constants.device)
                tmp[:, 0] = -float('inf')
                best_action = torch.argmax(tmp, dim=-1).item()

            elif len(parser.stack) == 1 and len(parser.buffer) == 0:
                # best_action = torch.argmax(action_probabilities[:, 1], dim=-1).item()
                best_action = -2

            elif len(parser.buffer) < 1:
                if has_head(parser.stack[-1], parser.arcs):
                    best_action = 3
                else:
                    best_action = -2

            else:
                # reduce if has head
                # left-arc if does not have head
                tmp = action_probabilities.clone().detach().to(device=constants.device)
                if not has_head(parser.stack[-1], parser.arcs):
                    # can left, can't reduce
                    tmp[:, 3] = -float('inf')
                else:
                    # can't left, can reduce
                    tmp[:, 1] = -float('inf')
                best_action = torch.argmax(tmp, dim=-1).item()

            rel = torch.argmax(rel_probabilities, dim=-1).item()  # +1
            rel_ind = torch.tensor([rel], dtype=torch.long).to(device=constants.device)
            rel_embed = self.rel_embeddings(rel_ind).to(device=constants.device)

        if best_action == 0:
            self.action.push(self.get_action_embed(constants.shift))
            parser.shift()
            ting = self.buffer.pop()
            self.stack.push(ting)
        elif best_action == 1:
            self.stack.pop()
            act_embed = self.get_action_embed(constants.left_arc_eager)
            self.action.push(act_embed)
            ret = parser.left_arc_eager(act_embed, rel, rel_embed, self.linear_tree)
            self.buffer.pop()
            self.buffer.push(ret.unsqueeze(0).unsqueeze(1))
        elif best_action == 2:
            act_embed = self.get_action_embed(constants.right_arc_eager)
            self.action.push(act_embed)
            ret = parser.right_arc_eager(act_embed, rel, rel_embed, self.linear_tree)
            self.stack.pop()
            self.stack.push(ret.unsqueeze(0).unsqueeze(1))
            self.stack.push(self.buffer.pop())
        elif best_action == 3:
            act_embed = self.get_action_embed(constants.reduce)
            self.action.push(act_embed)
            parser.reduce()
            self.stack.pop()
        else:
            self.stack.pop()
            self.action.push(self.get_action_embed(constants.left_arc_eager))
            elem = parser.stack.pop()
            parser.arcs.append((elem[1], elem[1], rel))

        return parser, (action_probabilities, rel_probabilities), (action_target, rel_target)

    def parse_step_hybrid(self, parser, labeled_transitions, mode):
        action_probabilities, rel_probabilities = self.parser_probabilities()
        if labeled_transitions is not None:
            best_action = labeled_transitions[0].item()
            rel = labeled_transitions[1]
        else:
            best_action = -2
            rel = 0

        if best_action != -2:
            action_target = torch.tensor([best_action], dtype=torch.long).to(device=constants.device)
        else:
            action_target = torch.tensor([1], dtype=torch.long).to(device=constants.device)

        if best_action == 0 or best_action == -2:
            rel = 0
        rel_target = torch.tensor([rel], dtype=torch.long).to(device=constants.device)
        rel_embed = self.rel_embeddings(rel_target).to(device=constants.device)

        if mode == 'eval':

            if len(parser.stack) < 1:
                # can only shift
                best_action = 0  # torch.argmax(action_probabilities[:, 0], dim=-1).item()


            elif len(parser.buffer) < 1:
                if len(parser.stack) >= 2:
                    # can't shift or left
                    # tmp = action_probabilities.clone().detach().to(device=constants.device)
                    # tmp[:, 0] = -float('inf')
                    best_action = 2  # torch.argmax(tmp, dim=-1).item()
                else:
                    best_action = -2

            elif len(parser.stack) < 2:
                tmp = action_probabilities.clone().detach().to(device=constants.device)
                tmp[:, 2] = -float('inf')
                best_action = torch.argmax(tmp, dim=-1).item()

            else:

                tmp = action_probabilities.clone().detach().to(device=constants.device)
                best_action = torch.argmax(tmp, dim=-1).item()

            rel = torch.argmax(rel_probabilities, dim=-1).item()  # +1
            rel_ind = torch.tensor([rel], dtype=torch.long).to(device=constants.device)
            rel_embed = self.rel_embeddings(rel_ind).to(device=constants.device)

        if best_action == 0:
            self.action.push(self.get_action_embed(constants.shift))
            parser.shift()
            ting = self.buffer.pop()
            self.stack.push(ting)
        elif best_action == 1:
            self.stack.pop()
            act_embed = self.get_action_embed(constants.left_arc_eager)
            self.action.push(act_embed)
            parser.left_arc_hybrid(act_embed, rel, rel_embed, self.linear_tree)

        elif best_action == 2:
            act_embed = self.get_action_embed(constants.reduce_r)
            self.action.push(act_embed)
            ret = parser.reduce_r_hybrid(act_embed, rel, rel_embed, self.linear_tree)
            self.stack.pop()
            self.stack.pop()
            self.stack.push(ret.unsqueeze(0).unsqueeze(1))
        else:
            self.stack.pop()
            self.action.push(self.get_action_embed(constants.left_arc_eager))
            elem = parser.stack.pop()
            parser.arcs.append((elem[1], elem[1], rel))

        return parser, (action_probabilities, rel_probabilities), (action_target, rel_target)

    def forward(self, x, transitions, relations, mode):

        sent_lens = (x[0] != 0).sum(-1).to(device=constants.device)
        transit_lens = (transitions != -1).sum(-1).to(device=constants.device)
        x_emb = self.get_embeddings(x)

        probs_action_batch = torch.ones((x_emb.shape[0], transitions.shape[1]-1, self.num_actions)).to(
            device=constants.device)
        probs_rel_batch = torch.ones((x_emb.shape[0], transitions.shape[1]-1, self.num_rels)).to(device=constants.device)
        targets_action_batch = torch.ones((x_emb.shape[0], transitions.shape[1]-1, 1), dtype=torch.long).to(
            device=constants.device)
        targets_rel_batch = torch.ones((x_emb.shape[0], transitions.shape[1]-1, 1), dtype=torch.long).to(
            device=constants.device)

        heads_batch = torch.ones((x_emb.shape[0], x_emb.shape[1])).to(device=constants.device)
        rels_batch = torch.ones((x_emb.shape[0], x_emb.shape[1])).to(device=constants.device)
        heads_batch *= -1
        rels_batch *= -1
        probs_rel_batch *= -1
        probs_action_batch *= -1
        targets_rel_batch *= -1
        targets_action_batch *= -1

        # for testing
        # labeled_transitions = self.labeled_action_pairs(transitions[0], relations[0])
        # tr = [t.item() for (t,_) in labeled_transitions]
        # self.sanity_parse(tr,heads,sent_lens)
        #if mode == 'eval':
        #    probs_action_all_batches = []
        #    probs_rel_all_batches = []
        #    targets_action_all_batches = []
        #    targets_rel_all_batches = []
        #    max_steps = 0
        # parse every sentence in batch

        for i, sentence in enumerate(x_emb):
            # initialize a parser
            curr_sentence_length = sent_lens[i]
            curr_transition_length = transit_lens[i]
            sentence = sentence[:curr_sentence_length, :]

            labeled_transitions = self.labeled_action_pairs(transitions[i, :curr_transition_length],
                                                            relations[i, :curr_sentence_length])

            parser = ShiftReduceParser(sentence, self.embedding_size, self.transition_system)
            # initialize buffer first
            # for word in sentence:
            #    self.buffer.push(word.reshape(1, 1, word.shape[0]))
            if self.transition_system == constants.hybrid:
                for word in sentence:
                    self.buffer.push(word.unsqueeze(0).unsqueeze(1))
            else:
                for word in reversed(sentence):
                    self.buffer.push(word.unsqueeze(0).unsqueeze(1))

            for step in range(len(labeled_transitions)):
                parser, probs, target = self.parse_step(parser,
                                                        labeled_transitions[step],
                                                        mode)

                if step < len(labeled_transitions)-1:
                    (action_probs, rel_probs) = probs
                    (action_target, rel_target) = target
                    probs_action_batch[i, step, :] = action_probs
                    probs_rel_batch[i, step, :] = rel_probs
                    targets_action_batch[i, step, :] = action_target
                    targets_rel_batch[i, step, :] = rel_target

            heads_batch[i, :sent_lens[i]] = parser.heads_from_arcs()[0]
            rels_batch[i, :sent_lens[i]] = parser.heads_from_arcs()[1]
            self.stack.back_to_init()
            self.buffer.back_to_init()
            self.action.back_to_init()
            #del parser

            """
            if mode == 'train':

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

            else:
                step = 0
                probs_action_this_batch = []
                probs_rel_this_batch = []
                targets_action_this_batch = []
                targets_rel_this_batch = []
                for i in range(curr_sentence_length):
                #while not parser.is_parse_complete():
                    if step < len(labeled_transitions):
                        parser, probs, target = self.parse_step(parser,
                                                                labeled_transitions[i],
                                                                mode)

                    #else:
                    #    parser, probs, target = self.parse_step(parser,
                    #                                            None,
                    #                                            mode)

                    (action_probs, rel_probs) = probs
                    (action_target, rel_target) = target
                    probs_action_this_batch.append(action_probs)
                    probs_rel_this_batch.append(rel_probs)
                    targets_action_this_batch.append(action_target)
                    targets_rel_this_batch.append(rel_target)
                    step += 1
                if step > max_steps:
                    max_steps = step
        
                probs_action_all_batches.append(torch.stack(probs_action_this_batch).permute(1, 0, 2))
                probs_rel_all_batches.append(torch.stack(probs_rel_this_batch).permute(1, 0, 2))
                targets_action_all_batches.append(torch.stack(targets_action_this_batch).permute(1, 0))
                targets_rel_all_batches.append(torch.stack(targets_rel_this_batch).permute(1, 0))
            """

        """
        if mode == 'eval':
            probs_action_batch = torch.ones((x_emb.shape[0], max_steps, self.num_actions)).to(device=constants.device)
            probs_rel_batch = torch.ones((x_emb.shape[0], max_steps, self.num_rels)).to(device=constants.device)
            targets_action_batch = torch.ones((x_emb.shape[0], max_steps, 1), dtype=torch.long).to(
                device=constants.device)
            targets_rel_batch = torch.ones((x_emb.shape[0], max_steps, 1), dtype=torch.long).to(device=constants.device)
            probs_action_batch *= -1
            probs_rel_batch *= -1
            targets_action_batch *= -1
            targets_rel_batch *= -1
            for i in range(x_emb.shape[0]):
                probs_action_batch[i, :probs_action_all_batches[i].shape[1], :] = probs_action_all_batches[i]
                probs_rel_batch[i, :probs_rel_all_batches[i].shape[1], :] = probs_rel_all_batches[i]
                targets_action_batch[i, :targets_action_all_batches[i].shape[1], :] = targets_action_all_batches[
                    i].unsqueeze(2)
                targets_rel_batch[i, :targets_rel_all_batches[i].shape[1], :] = targets_rel_all_batches[i].unsqueeze(2)
                
        """
        batch_loss = self.loss(probs_action_batch, targets_action_batch, probs_rel_batch, targets_rel_batch)


        return batch_loss, heads_batch, rels_batch

    def loss(self, probs, targets, probs_rel, targets_rel):
        criterion1 = nn.CrossEntropyLoss(ignore_index=-1).to(device=constants.device)
        criterion2 = nn.CrossEntropyLoss(ignore_index=0).to(device=constants.device)
        orig_size = probs.shape[0]
        probs = probs.reshape(-1, probs.shape[-1])
        targets = targets.reshape(-1)
        targets = targets[probs[:, 0] != -1]
        probs = probs[probs[:, 0] != -1, :]

        probs_rel = probs_rel.reshape(-1, probs_rel.shape[-1])

        targets_rel = targets_rel.reshape(-1)
        # targets_rel = targets_rel[probs_rel[:, 0] != -1]
        # probs_rel = probs_rel[probs_rel[:, 0] != -1, :]
        probs_rel = probs_rel[targets_rel != 0, :]
        targets_rel = targets_rel[targets_rel != 0]
        targets_rel = targets_rel[probs_rel[:, 0] != -1]
        probs_rel = probs_rel[probs_rel[:, 0] != -1, :]

        loss = 2/3 * criterion1(probs, targets)
        loss += 1/3 * criterion2(probs_rel, targets_rel)
        # loss /= (2*orig_size)
        # print(loss)
        # print(l2)
        #
        return loss

    def get_args(self):
        return {
            'vocabs': self.vocabs,
            'embedding_size': self.embedding_size,
            'hidden_size': self.hidden_size,
            'arc_size': self.arc_size,
            'label_size': self.label_size,
            'nlayers': self.nlayers,
            'dropout': self.dropout_prob,
        }

    def check_if_good(self, built, heads, sent_lens):
        heads_proper = heads[:sent_lens].tolist()[0]

        # heads_proper = [0] + heads_proper

        sentence_proper_ind = list(range(len(heads_proper)))
        word2head = {w: h for (w, h) in zip(sentence_proper_ind, heads_proper)}
        true_arcs = get_arcs(word2head)

        better = [(u, v) for (u, v, _) in built]
        for (u, v) in true_arcs:
            if (u, v) in better:
                continue
            else:
                return False
        return True

    def sanity_parse(self, actions, heads, sent_lens):
        stack = []
        buffer = []
        arcs = []
        heads_proper = heads[:sent_lens].tolist()[0]

        # heads_proper = [0] + heads_proper

        sentence_proper_ind = list(range(len(heads_proper)))
        word2head = {w: h for (w, h) in zip(sentence_proper_ind, heads_proper)}
        true_arcs = get_arcs(word2head)
        buffer = sentence_proper_ind.copy()
        for act in actions:
            if act == 0:
                stack.append(buffer.pop(0))
            elif act == 1:
                t = stack[-1]
                l = buffer[0]
                arcs.append((l, t))
                stack.pop(-1)
            elif act == 2:
                t = stack[-1]
                l = buffer[0]
                arcs.append((t, l))
                buffer[0] = t
                stack.pop(-1)
            else:
                item = stack.pop(-1)
                arcs.append((item, item))

        print(set(true_arcs) == set(arcs))
