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


# root = (torch.tensor(1).to(device=constants.device), torch.tensor(1).to(device=constants.device))

# taken from stack-lstm-ner (will give credit)
class StackRNN(object):
    def __init__(self, cell, initial_state, initial_hidden, dropout, p_empty_embedding=None):
        self.cell = cell
        self.dropout = dropout
        # self.s = [(initial_state, None)]
        self.s = [(initial_state, initial_hidden)]

        self.empty = None
        if p_empty_embedding is not None:
            self.empty = p_empty_embedding

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
        return self.s.pop()[0]  # [0]

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


class NeuralTransitionParser(nn.Module):
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
        root = (torch.tensor(1).to(device=constants.device), torch.tensor(1).to(device=constants.device))

        self.root_embed = self.get_embeddings(root)
        self.shift_embedding = self.action_embeddings(torch.LongTensor([0]).to(device=constants.device)).unsqueeze(0)
        self.reduce_l_embedding = self.action_embeddings(torch.LongTensor([1]).to(device=constants.device)).unsqueeze(0)
        self.reduce_r_embedding = self.action_embeddings(torch.LongTensor([2]).to(device=constants.device)).unsqueeze(0)

        """
            MLP[stack_lstm_output,buffer_lstm_output,action_lstm_output] ---> decide next transition
        """

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
        self.mlp_lin = nn.Linear(int(self.hidden_size / 2) * 3,
                                 self.num_actions)  # nn.Softmax(dim=-1)(nn.ReLU()(nn.Linear(self.hidden_size * 3, self.num_actions)())())()
        self.mlp_relu = nn.ReLU()
        self.mlp_softmax = nn.Softmax(dim=-1)

    def create_embeddings(self, vocabs, pretrained):
        words, tags, _ = vocabs
        word_embeddings = WordEmbedding(words, self.embedding_size, pretrained=pretrained)
        tag_embeddings = nn.Embedding(tags.size, self.embedding_size)
        action_embedding = ActionEmbedding(self.actions, self.embedding_size).embedding
        return word_embeddings, tag_embeddings, action_embedding

    def get_embeddings(self, x):
        return torch.cat([self.word_embeddings(x[0]), self.tag_embeddings(x[1])], dim=-1)

    def parse_step(self, parser, stack, buffer, action, oracle, mode):
        # get parser state
        parser_state = torch.cat([stack.embedding(), buffer.embedding(), action.embedding()], dim=-1)
        action_probabilities = self.mlp_lin(parser_state)
        action_probabilities = self.mlp_relu(action_probabilities)
        action_probabilities = self.mlp_softmax(action_probabilities).squeeze(0)  # .clone()
        criterion_a = nn.CrossEntropyLoss().to(device=constants.device)
        l = None
        if mode == 'train':
            if oracle.item() == -2:
                # last action
                parser.stack.pop()
                stack.pop()
                target = torch.tensor([1])
                l = criterion_a(action_probabilities, target)
                return parser, action_probabilities, (stack, buffer, action), l
            best_action = oracle.item()
            target = oracle.reshape(1)

            l = criterion_a(action_probabilities, target)
        else:
            target = oracle.reshape(1)
            l = criterion_a(action_probabilities, target)
            if parser.stack.get_len() < 1:
                # can't left or right
                best_action = torch.argmax(action_probabilities[:, 0], dim=-1).item()
            elif parser.buffer.get_len() == 1:
                # can't shift
                tmp = action_probabilities.detach().clone().to(device=constants.device)
                tmp[:, 0] = -float('inf')
                best_action = torch.argmax(tmp, dim=-1).item()
            elif parser.stack.get_len() == 1 and parser.buffer.get_len() == 0:
                best_action = -2
            else:
                best_action = torch.argmax(action_probabilities.clone().detach(), dim=-1).item()

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
            parser.reduce_l(self.reduce_l_embedding)
            # print(constants.reduce_l)
        elif best_action == -2:
            stack.pop()
            elem = parser.stack.pop()
            parser.arcs.append((elem[1], elem[1]))
        else:
            # reduce-r
            buffer.push(stack.pop())  # not sure, should replace in buffer actually...
            action.push(self.reduce_r_embedding)
            parser.reduce_r(self.reduce_r_embedding)

        return parser, action_probabilities, (stack, buffer, action), l

    def forward(self, x, transitions=None):
        if transitions is None:
            mode = "predict"
        else:
            mode = "train"

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
        head_probs_batch = torch.zeros((x_emb.shape[0], x_emb.shape[1], x_emb.shape[1])).to(device=constants.device)

        act_loss = 0
        heads_batch = None
        # if mode == 'train':

        # if oracle_actions[-1] != -2:
        #    oracle_actions = torch.cat([oracle_actions, torch.tensor([-2]).to(device=constants.device)], dim=0)
        # else:
        heads_batch = torch.ones((x_emb.shape[0], x_emb.shape[1]), requires_grad=False).to(device=constants.device)
        heads_batch *= -1

        # parse every sentence in batch
        for i, sentence in enumerate(x_emb):
            # initialize a parser
            curr_sentence_length = sent_lens[i]
            sentence = sentence[:curr_sentence_length, :]
            # heads_proper = torch.cat([torch.tensor([0]).to(device=constants.device),heads[i][:sent_lens[i]]],dim=0)#[0] + heads
            sentence_proper = torch.cat([self.root_embed.unsqueeze(0), sentence],
                                        dim=0)  # list(range(len(heads_proper)))
            # print("heads proper len {}".format(len(heads_proper)))
            # print("proper len {}".format(len(sentence_proper)))
            # print("sentence len {}".format(len(sentence)))
            # print(sentence)
            # word2head = {w: h for (w, h) in zip(sentence_proper, heads_proper)}
            parser = ShiftReduceParser(sentence_proper, self.embedding_size, self.transition_system)
            # initialize buffer first
            for word in sentence:
                buffer.push(word.reshape(1, 1, word.shape[0]))
            # push first word to stack

            stack.push(buffer.pop())
            action.push(self.shift_embedding)
            parser.shift(self.shift_embedding)
            oracle_actions_redundant = transitions[i]
            oracle_actions_ind = torch.where(oracle_actions_redundant != -1)[0]
            oracle_actions = oracle_actions_redundant[oracle_actions_ind]
            oracle_actions = oracle_actions[1:]
            for step in range(len(oracle_actions)):
                parser, probs, configuration, l = self.parse_step(parser, stack, buffer, action, oracle_actions[step],
                                                                  mode)
                (stack, buffer, action) = configuration
                act_loss += l
            heads_batch[i, :sent_lens[i]] = parser.heads_from_arcs()

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
