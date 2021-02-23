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
from .oracle import arc_standard_oracle
from .word_embedding import WordEmbedding, ActionEmbedding
from ..algorithm.transition_parsers import ShiftReduceParser


def get_arcs(word2head):
    arcs = []
    for word in word2head:
        arcs.append((word2head[word], word))
    # for i in range(len(heads)):
    #    arcs.append((heads[i], i))
    return arcs


# root = (torch.tensor(1).to(device=constants.device), torch.tensor(1).to(device=constants.device))

# adapted from stack-lstm-ner (https://github.com/clab/stack-lstm-ner)
class StackRNN(nn.Module):
    def __init__(self, cell, initial_state, initial_hidden, dropout, p_empty_embedding=None):
        super().__init__()
        self.cell = cell
        self.dropout = dropout
        # self.s = [(initial_state, None)]
        self.s = [(initial_state, initial_hidden)]

        self.empty = None
        if p_empty_embedding is not None:
            self.empty = p_empty_embedding

    def push_first(self, expr):
        expr = expr.unsqueeze(0).unsqueeze(1)
        out, hidden = self.cell(expr, self.s[0][1])
        self.s[0] = (out, hidden)

    def push(self, expr, extra=None):
        # print(self.s[-1][0][0].shape)
        # print(expr.shape)
        # self.dropout(self.s[-1][0][0])
        # print("(seqlen,batchsize,indim) {}".format(expr.shape))
        # print("has to be a tuple {}".format(self.s[-1][0][0].shape))
        # print("has to be a tuple {}".format(self.s[-1][0][1].shape))
        # print(len(self.s[-1][0]))
        # self.s.append((self.cell(expr, (self.s[-1][0][0],self.s[-1][0][1])), extra))
        # print(len(self.s[-1]))
        out, hidden = self.cell(expr, self.s[-1][1])
        self.s.append((out, hidden))

    def pop(self):
        # x = self.s.pop()#[1]
        ##y = self.s.pop([0]
        # print("x {}".format(x[0]))
        # print("< {}".format(x[1]))
        return self.s.pop(-1)[0]  # [0]

    def embedding(self):
        return self.s[-1][0] if len(self.s) > 1 else self.empty

    def back_to_init(self):
        while self.__len__() > 0:
            self.pop()

    def clear(self):
        self.s.reverse()
        self.back_to_init()

    def __len__(self):
        return len(self.s) - 1


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
        self.actions = transition_system[0]  # [shift, reduce_l, reduce_r]
        self.num_actions = len(self.actions)

        # word, tag and action embeddings
        self.word_embeddings, self.tag_embeddings, self.action_embeddings = \
            self.create_embeddings(vocabs, pretrained=pretrained_embeddings)
        self.root = (torch.tensor(1).to(device=constants.device), torch.tensor(1).to(device=constants.device))

        # self.root_embed = self.get_embeddings(root)
        self.shift_embedding = self.action_embeddings(torch.LongTensor([0]).to(device=constants.device)).unsqueeze(0)
        self.reduce_l_embedding = self.action_embeddings(torch.LongTensor([1]).to(device=constants.device)).unsqueeze(0)
        self.reduce_r_embedding = self.action_embeddings(torch.LongTensor([2]).to(device=constants.device)).unsqueeze(0)

        """
            MLP[stack_lstm_output,buffer_lstm_output,action_lstm_output] ---> decide next transition
        """
        _, _, rels = vocabs
        # microsoft dude in presentation said: shift + num_rels*2
        # [0....rels.size] == (reduce_l,label)
        # [rels.size()+1:rels.size*2] == (reduce_r,label)
        # [-1] == shift
        self.num_actions = 1 + rels.size * 2
        # neural model parameters
        self.dropout = nn.Dropout(self.dropout_prob)
        # lstms
        self.stack_lstm = nn.LSTM(self.embedding_size * 2, int(self.hidden_size / 2)).to(device=constants.device)
        self.buffer_lstm = nn.LSTM(self.embedding_size * 2, int(self.hidden_size / 2)).to(device=constants.device)
        self.action_lsttm = nn.LSTM(self.embedding_size, int(self.hidden_size / 2)).to(device=constants.device)

        # parser state
        self.parser_state = nn.Parameter(torch.zeros((self.batch_size, self.hidden_size * 3 * 2))).to(
            device=constants.device)
        # init params
        input_init = torch.zeros((1, 1, int(self.hidden_size / 2))).to(
            device=constants.device)
        hidden_init = torch.zeros((1, 1, int(self.hidden_size / 2))).to(
            device=constants.device)

        input_init_act = torch.zeros((1, 1, int(self.hidden_size / 2))).to(
            device=constants.device)
        hidden_init_act = torch.zeros((1, 1, int(self.hidden_size / 2))).to(
            device=constants.device)

        self.lstm_init_state = (nn.init.xavier_normal_(input_init), nn.init.xavier_normal_(hidden_init))
        self.lstm_init_state_actions = (nn.init.xavier_normal_(input_init_act), nn.init.xavier_normal_(hidden_init_act))
        self.gaurd = torch.zeros((1, 1, self.embedding_size * 2)).to(device=constants.device)
        self.empty_initial = nn.Parameter(torch.randn(self.batch_size, self.hidden_size))
        # MLP it's actually a one layer network
        self.mlp_lin1 = nn.Linear(int(self.hidden_size / 2) * 3,
                                  300)
        self.mlp_lin2 = nn.Linear(300,
                                  self.num_actions)
        # self.mlp_relu = nn.ReLU()
        self.mlp_softmax = nn.Softmax(dim=-1)

    def create_embeddings(self, vocabs, pretrained):
        words, tags, _ = vocabs
        word_embeddings = WordEmbedding(words, self.embedding_size, pretrained=pretrained)
        tag_embeddings = nn.Embedding(tags.size, self.embedding_size)
        action_embedding = ActionEmbedding(self.actions, self.embedding_size).embedding
        return word_embeddings, tag_embeddings, action_embedding

    def get_embeddings(self, x):
        return torch.cat([self.word_embeddings(x[0]), self.tag_embeddings(x[1])], dim=-1).to(device=constants.device)

    def get_parser_state(self, stack, buffer, action):
        return torch.cat([stack.embedding(), buffer.embedding(), action.embedding()], dim=-1)

    def labeled_action_pairs(self, actions, relations):
        labeled_acts = []
        tmp_rels = relations.clone().detach().tolist()

        for act in actions:
            if act == 0 or act is None:
                labeled_acts.append((act, -1))
            elif act is not None:
                labeled_acts.append((act, tmp_rels[0]))
                tmp_rels.pop(0)

        return labeled_acts

    def action_rel2index(self, action, rel):
        # goodluck with this
        ret = 0
        if action == 0:
            ret = self.num_actions-1
        elif action == 1:
            ret =  rel
        elif action == 2:
            ret =  rel * 2
        else:
            # action == -2
            # 7 is root index
            ret =  7
        return torch.tensor([ret]).to(device=constants.device)

    def index2action(self,index):
        num_not_shift = self.num_actions-1
        if index == num_not_shift:
            return 0
        elif index < num_not_shift:
            return 1
        else:
            return 2

    def parse_step(self, parser, stack, buffer, action, labeled_transitions, mode):
        # get parser state
        parser_state = torch.cat([stack.embedding(), buffer.embedding(), action.embedding()], dim=-1)

        parser_state = nn.ReLU()(self.mlp_lin1(parser_state))
        action_probabilities = nn.ReLU()(self.mlp_lin2(parser_state))
        # action_probabilities = self.mlp_relu(action_probabilities)
        action_probabilities = self.mlp_softmax(action_probabilities).squeeze(0)  # .clone()
        criterion_a = nn.CrossEntropyLoss().to(device=constants.device)
        l = None
        if labeled_transitions is not None:
            best_action = labeled_transitions[0].item()
            rel = labeled_transitions[1]
            target = self.action_rel2index(best_action, rel)
        else:
            best_action = -2
            rel = 7
            target = self.action_rel2index(-2,7)

        l = criterion_a(action_probabilities,target)

        if mode == 'eval':
            final = False
            if len(parser.stack) < 1:
                # can't left or right
                index = self.num_actions-1#torch.argmax(action_probabilities[:, 0], dim=-1).item()
            elif len(parser.buffer) == 1:
                # can't shift
                tmp = action_probabilities.detach().clone().to(device=constants.device)
                tmp[:, -1] = -float('inf')
                index = torch.argmax(tmp, dim=-1).item()
            elif len(parser.stack) == 1 and len(parser.buffer) == 0:
                best_action = -2
                index = 7
                final = True
            else:
                index = torch.argmax(action_probabilities.clone().detach(), dim=-1).item()
            if not final:
                best_action = self.index2action(index)
            if best_action != 0:
                rel = index if index < self.num_actions else int(index/2)
        # do the action
        if best_action == 0:
            # shift
            stack.push(buffer.pop())
            action.push(self.shift_embedding)
            parser.shift(self.shift_embedding)
            # print(constants.shift)
        elif best_action == 1:
            # reduce-l
            stack.pop()
            action.push(self.reduce_l_embedding)
            ret = parser.reduce_l(self.reduce_l_embedding,rel)
            # buffer.push_first(ret)
            # print(constants.reduce_l)
            stack.push(ret.unsqueeze(0).unsqueeze(1))

        elif best_action == 2:
            # reduce-r
            # buffer.push(stack.pop())  # not sure, should replace in buffer actually...
            action.push(self.reduce_r_embedding)
            ret = parser.reduce_r(self.reduce_r_embedding,rel)
            # buffer.push_first(ret)
            stack.pop()
            stack.push(ret.unsqueeze(0).unsqueeze(1))
        else:
            stack.pop()
            elem = parser.stack.pop()
            parser.arcs.append((elem[1], elem[1],rel))

        return parser, action_probabilities, (stack, buffer, action), l



    def forward(self, x, transitions, relations, mode):

        stack = StackRNN(self.stack_lstm, self.lstm_init_state, self.lstm_init_state, self.dropout, self.empty_initial)
        buffer = StackRNN(self.buffer_lstm, self.lstm_init_state, self.lstm_init_state, self.dropout,
                          self.empty_initial)
        action = StackRNN(self.action_lsttm, self.lstm_init_state_actions, self.lstm_init_state_actions, self.dropout,
                          self.empty_initial)
        sent_lens = (x[0] != 0).sum(-1).to(device=constants.device)
        # print(sent_lens)

        x_emb = self.get_embeddings(x)
        # print(x_emb.shape)
        max_num_actions_taken = 0

        actions_batch = []
        stack.push(self.gaurd)
        buffer.push(self.gaurd)

        act_loss = 0
        # if mode == 'train':

        # if oracle_actions[-1] != -2:
        #    oracle_actions = torch.cat([oracle_actions, torch.tensor([-2]).to(device=constants.device)], dim=0)
        # else:
        heads_batch = torch.ones((x_emb.shape[0], x_emb.shape[1]), requires_grad=False).to(device=constants.device)
        rels_batch = torch.ones((x_emb.shape[0], x_emb.shape[1]), requires_grad=False).to(device=constants.device)
        heads_batch *= -1

        # for testing
        # self.sanity_parse(transitions[0],heads,sent_lens)
        #print(transitions.shape)
        #print(relations.shape)
        # parse every sentence in batch
        for i, sentence in enumerate(x_emb):
            # initialize a parser
            curr_sentence_length = sent_lens[i]
            sentence = sentence[:curr_sentence_length, :]
            #if mode == 'train':
            labeled_transitions = self.labeled_action_pairs(transitions[i,:sent_lens[i]], relations[i,:sent_lens[i]])
            # for testing, uncomment
            # heads_proper = heads[i,:sent_lens[i]].tolist()
            ## heads_proper = [0] + heads_proper
            # sentence_proper_ind = list(range(len(heads_proper)))
            # word2head = {w: h for (w, h) in zip(sentence_proper_ind, heads_proper)}
            # true_arcs = get_arcs(word2head)

            parser = ShiftReduceParser(sentence, self.embedding_size, self.transition_system)
            # initialize buffer first
            for word in sentence:
                buffer.push(word.reshape(1, 1, word.shape[0]))
            # push first word to stack

            stack.push(buffer.pop())
            action.push(self.shift_embedding)
            parser.shift(self.shift_embedding)
            #oracle_actions_redundant = transitions[i]
            #oracle_actions_ind = torch.where(oracle_actions_redundant != -1)[0]
            #oracle_actions = oracle_actions_redundant[oracle_actions_ind]
            #oracle_actions = oracle_actions[1:]
            labeled_transitions = labeled_transitions[1:]

            if mode == 'train':
                # print(labeled_transitions)
                for step in range(len(labeled_transitions)):
                    parser, probs, configuration, l = self.parse_step(parser, stack, buffer, action,
                                                                      labeled_transitions[step],
                                                                      mode)
                    (stack, buffer, action) = configuration
                    act_loss += l
                act_loss /= len(labeled_transitions)
                heads_batch[i, :sent_lens[i]] = parser.heads_from_arcs()[0]
                rels_batch[i, :sent_lens[i]] = parser.heads_from_arcs()[1]

            else:
                step = 0
                while not parser.is_parse_complete():
                    if step < len(labeled_transitions):
                        parser, probs, configuration, l = self.parse_step(parser, stack, buffer, action,
                                                                          labeled_transitions[step],
                                                                          mode)
                    else:
                        parser, probs, configuration, l = self.parse_step(parser, stack, buffer, action,
                                                                          None,
                                                                          mode)

                    (stack, buffer, action) = configuration
                    act_loss += l
                    step += 1
                act_loss /= len(labeled_transitions)
                heads_batch[i, :sent_lens[i]] = parser.heads_from_arcs()[0]
                rels_batch[i, :sent_lens[i]] = parser.heads_from_arcs()[1]

        act_loss /= x_emb.shape[0]

        return act_loss, heads_batch

    def get_args(self):
        return {
            'vocabs': self.vocabs,
            'embedding_size': self.embedding_size,
            'hidden_size': self.hidden_size,
            'arc_size': self.arc_size,
            'label_size': self.label_size,
            'nlayers': self.nlayers,
            'dropout': self.dropout_p,
        }

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
        print(len(sentence_proper_ind))
        print(len(heads_proper))
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
