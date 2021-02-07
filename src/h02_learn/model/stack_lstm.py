import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

import numpy as np

from utils import constants
from .base import BaseParser
from .modules import Biaffine, Bilinear, StackLSTM
from .word_embedding import WordEmbedding, ActionEmbedding

"""
TODO: Need to add the built tree embeddings to each 
"""


class Arcs:
    def __init__(self):
        self.arcs = []#torch.tensor([]).to(device=constants.device)

    def add_arc(self, i, j):
        self.arcs.append((i, j))

    def has_incoming(self, a):
        for (src, dst) in self.arcs:
            if dst == a:
                return True
        return False


class Stack:
    def __init__(self):
        self.stack =[]# torch.tensor([]).to(device=constants.device)

    def push(self, x):
        self.stack.append(x)

    def pop(self):
        return self.stack.pop(-1)

    def pop_second(self):
        return self.stack.pop(-2)

    def second(self):
        return self.stack[-2]

    def set_second(self, x):
        (_,ind) = self.stack[-2]
        self.stack[-2] = (x,ind)

    def top(self):
        return self.stack[-1]

    def set_top(self, x):
        (_,ind) = self.stack[-1]
        self.stack[-1] = (x,ind)
    def get_len(self):
        return len(self.stack)


class Buffer:
    def __init__(self, sentence):
        self.buffer = [(word, i) for i, word in enumerate(sentence)]  # .split()

    def pop_left(self):
        return self.buffer.pop(0)

    def left(self):
        return self.buffer[0]

    def get_len(self):
        return len(self.buffer)


# implement as lambda functions?!!
root = (torch.tensor(1).to(device=constants.device), torch.tensor(1).to(device=constants.device))
shift = "SHIFT"
shift_idx = 0
fshift = lambda sigma, beta, A: sigma.push(beta.pop_left())

reduce_l = "REDUCE_L"
reduce_l_idx = 1
freduce_l = lambda sigma, beta, A: A.add_arc(sigma.top(), sigma.pop_second())

reduce_r = "REDUCE_R"
reduce_r_idx = 2
freduce_r = lambda sigma, beta, A: A.add_arc(sigma.second(), sigma.pop())

reduce = "REDUCE"
freduce = lambda sigma, beta, A: sigma.pop() if A.has_incoming(sigma.top()) else None

left_arc_eager = "LEFT_ARC_EAGER"
fleft_arc_eager = lambda sigma, beta, A: A.add_arc(beta.left(), sigma.pop()) if not A.has_incoming(
    sigma.top()) else None

right_arc_eager = "RIGHT_ARC_EAGER"
fright_arc_eager = lambda sigma, beta, A: A.add_arc(sigma.top(), sigma.push(beta.pop_left()))

left_arc = "LEFT_ARC_H"
fleft_arc = lambda sigma, beta, A: A.add_arc(beta.left(), sigma.pop())

arc_standard = {shift: 0, reduce_l: 1, reduce_r: 2}
arc_standard_actions = {0: fshift, 1: freduce_l, 2: freduce_r}

arc_eager = {shift: 0, left_arc_eager: 1, right_arc_eager: 2, reduce: 3}
arc_eager_actions = {0: fshift, 1: fleft_arc_eager, 2: fright_arc_eager, 3: freduce}

hybrid = {shift: 0, left_arc: 1, reduce_r: 2}
hybrid_actions = {0: fshift, 1: fleft_arc, 2: freduce_r}


class ShiftReduceParser():

    def __init__(self, sentence, embedding_size):
        self.stack = Stack()
        self.buffer = Buffer(sentence)
        self.arcs = Arcs()
        self.action_history = []
        self.action_history_names = []
        self.sentence = sentence
        self.learned_repr = sentence
        self.embedding_size = embedding_size

        self.ind2continous = {i: vec for (vec, i) in self.buffer.buffer}
        self.linear = nn.Linear(5 * embedding_size, 2 * embedding_size)
        self.tanh = nn.Tanh()

    def get_stack_content(self):
        return [item[0] for item in self.stack.stack]

    def get_buffer_content(self):
        return [item[0] for item in self.buffer.buffer]

    def shift(self, act_emb):
        item = self.buffer.pop_left()
        self.stack.push(item)
        self.action_history_names.append(shift)
        self.action_history.append(act_emb)

    def reduce_l(self, act_emb):
        item_top = self.stack.top()
        item_second = self.stack.pop_second()
        # self.arcs.add_arc(item_top, item_second)
        self.arcs.add_arc(item_top[1], item_second[1])
        self.action_history_names.append(reduce_l)
        self.action_history.append(act_emb)

        repr = torch.cat([item_top[0], item_second[0], act_emb.reshape(self.embedding_size)], dim=-1).to(device=constants.device)
        c = self.tanh(self.linear(repr))
        #self.ind2continous[item_top[1]] = c
        self.stack.set_top(c)

    def reduce_r(self, act_emb):
        second_item = self.stack.second()
        top_item = self.stack.pop()
        self.arcs.add_arc(second_item[1], top_item[1])
        self.action_history_names.append(reduce_r)
        self.action_history.append(act_emb)

        repr = torch.cat([second_item[0], top_item[0], act_emb.reshape(self.embedding_size)], dim=-1).to(device=constants.device)
        c = self.tanh(self.linear(repr))
        #self.ind2continous[second_item[1]] = c
        self.stack.set_top(c)

    def is_parse_complete(self):
        buffer_empty = self.buffer.get_len() == 0
        stack_empty = self.stack.get_len() == 1
        complete = buffer_empty and stack_empty
        return complete

    def get_heads(self):
        # heads = np.empty((1, len(self.sentence)))
        heads = torch.zeros((1, len(self.sentence), self.embedding_size * 2)).to(device=constants.device)
        with torch.no_grad():
            for i in range(len(self.sentence)):
                for j in range(len(self.sentence)):
                    if (j, i) in self.arcs.arcs:
                        heads[0, i, :] = self.ind2continous[j]
                        continue

        return heads


class ArcStandardStackLSTM(BaseParser):
    def __init__(self, vocabs, embedding_size, hidden_size, arc_size, label_size,
                 nlayers=3, dropout=0.33, pretrained_embeddings=None, transition_system=None):
        super().__init__()

        self.vocabs = vocabs
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.arc_size = arc_size
        self.label_size = label_size
        self.nlayers = nlayers
        self.dropout_p = dropout
        self.actions = [shift, reduce_l, reduce_r]
        self.transition_ids = arc_standard
        _, _, rels = vocabs

        # continous representations
        self.word_embeddings, self.tag_embeddings, self.action_embeddings = \
            self.create_embeddings(vocabs, pretrained=pretrained_embeddings)

        #print(self.action_embeddings)
        # should have a mapping from actions to embedding
        self.act2embed = {}
        self.embed2act = {}

        self.shift_embedding = self.action_embeddings(torch.LongTensor([0]).to(device=constants.device))
        self.reduce_l_embedding = self.action_embeddings(torch.LongTensor([1]).to(device=constants.device))
        self.reduce_r_embedding = self.action_embeddings(torch.LongTensor([2]).to(device=constants.device))

        self.shift_embedding = self.shift_embedding.reshape(1, self.shift_embedding.shape[0],
                                                            self.shift_embedding.shape[1])
        self.reduce_l_embedding = self.reduce_l_embedding.reshape(1, self.reduce_l_embedding.shape[0],
                                                                  self.reduce_l_embedding.shape[1])
        self.reduce_r_embedding = self.reduce_r_embedding.reshape(1, self.reduce_r_embedding.shape[0],
                                                                  self.reduce_r_embedding.shape[1])

        # stack lstms
        # dimensions are just random now
        # new lstms for every sentence?????
        self.buffer_lstm = StackLSTM(embedding_size * 2, int(hidden_size / 2), dropout=(dropout if nlayers > 1 else 0),
                                     batch_first=True, bidirectional=False)
        self.stack_lstm = StackLSTM(embedding_size * 2, int(hidden_size / 2), dropout=(dropout if nlayers > 1 else 0),
                                    batch_first=True, bidirectional=False)
        self.action_lstm = nn.LSTM(embedding_size, int(hidden_size / 2), dropout=(dropout if nlayers > 1 else 0),
                                   batch_first=True, bidirectional=False)

        # mlp for deciding actions
        self.chooser_linear = nn.Linear(embedding_size * 2 * 3, 3).to(device=constants.device)
        self.chooser_relu = nn.ReLU().to(device=constants.device)
        self.chooser_softmax = nn.Softmax().to(device=constants.device)
        self.dropout = nn.Dropout(dropout).to(device=constants.device)

        self.linear_arc_dep = nn.Linear(self.embedding_size*2, arc_size).to(device=constants.device)
        self.linear_arc_head = nn.Linear(self.embedding_size*2, arc_size).to(device=constants.device)
        self.biaffine = Biaffine(arc_size, arc_size)

        self.linear_label_dep = nn.Linear(self.embedding_size * 2, label_size).to(device=constants.device)
        self.linear_label_head = nn.Linear(self.embedding_size * 2, label_size).to(device=constants.device)
        self.bilinear_label = Bilinear(label_size, label_size, rels.size)

    def create_embeddings(self, vocabs, pretrained=None):
        words, tags, _ = vocabs
        word_embeddings = WordEmbedding(words, self.embedding_size, pretrained=pretrained)
        tag_embeddings = nn.Embedding(tags.size, self.embedding_size)
        action_embedding = ActionEmbedding(self.actions, self.embedding_size).embedding
        return word_embeddings, tag_embeddings, action_embedding

    def get_embeddings(self, x):
        return torch.cat([self.word_embeddings(x[0]), self.tag_embeddings(x[1])], dim=-1).to(device=constants.device)

    def get_action_embeddings(self, action_history):
        # ret_emb = torch.empty((len(action_history), self.action_embeddings.embedding_dim))
        # action_history = torch.stack(action_history)
        # for i, act in enumerate(action_history):
        #    #act_emb = self.action_embeddings(act)
        #    ret_emb[i,:] = act

        return torch.stack(action_history).squeeze_(0)

    def get_stack_or_buffer_embeddings(self, structure):
        ret_emb = torch.empty((len(structure), self.embedding_size))
        ret_emb = self.get_embeddings(structure)
        # for i, item in enumerate(structure):
        #    print(item.type)
        #    ret_emb[i, :] = self.get_embeddings(item)
        return ret_emb

    def shift(self):
        self.stack_lstm.create_and_push()
        self.action_lstm(self.shift_embedding)

    def reduce_l(self):
        self.action_lstm(self.reduce_l_embedding)

    def reduce_r(self):
        self.action_lstm(self.reduce_r_embedding)

    def decide_action(self, parser):

        parser_state = self.get_parser_state(parser)

        prob = self.chooser_linear(parser_state)
        prob = self.chooser_relu(prob)
        # prob = self.chooser_softmax(prob)

        return torch.argmax(prob).item(), prob

    def get_parser_state(self, parser):
        parser_stack = parser.get_stack_content()
        parser_buffer = parser.get_buffer_content()
        if len(parser_stack) > 1:
            #print("parser_stack")
            #for ting in parser_stack:
            #    print(ting.shape)
            stack_state = torch.stack(parser_stack)
        else:
            stack_state = parser_stack[0]

        if len(parser_buffer) > 1:

            buffer_state = torch.stack(parser_buffer)
        elif len(parser_buffer) == 0 and len(parser_stack) > 1:
            buffer_state = torch.zeros_like(stack_state).to(device=constants.device)
        else:
            buffer_state = parser_buffer[0]

        action_history = parser.action_history
        # stack_embeddings = self.get_stack_or_buffer_embeddings(stack_state.stack)
        # buffer_embeddings = self.get_stack_or_buffer_embeddings(buffer_state.buffer)
        action_embeddings = action_history[-1]
        # action_embeddings = action_embeddings.reshape(action_embeddings.shape[0],1,action_embeddings.shape[1])
        with torch.no_grad():
            so = self.stack_lstm(stack_state)[0]
            bo = self.buffer_lstm(buffer_state)[0]

            ao = self.action_lstm(action_embeddings)[0].reshape(1, bo.shape[1])
            state = torch.cat([so, bo, ao], dim=-1).to(constants.device)

        return state

    def run_lstm(self, lstm, x, sent_lens):
        lstm_in = pack_padded_sequence(x, sent_lens, batch_first=True, enforce_sorted=False)
        lstm_out, _ = lstm(lstm_in)
        h_t, _ = pad_packed_sequence(lstm_out, batch_first=True)
        h_t = self.dropout(h_t).contiguous()

        return h_t

    def best_legal_action(self, best_action, parser, probs):
        if (len(parser.stack.stack) > 1) and (len(parser.buffer.buffer) > 0):
            return best_action
        elif len(parser.buffer.buffer) == 0:
            probs = torch.cat([probs[:, 1], probs[:, 2]]).to(device=constants.device)
            return torch.argmax(probs).item() + 1
        else:
            return 0

    def parse_step(self, parser):

        # get action_id
        best_action, probs = self.decide_action(parser)
        best_action = self.best_legal_action(best_action, parser, probs)
        # print("BEST ACTION {}".format(best_action))
        # take action and update state
        # if(len(parser.stack.stack) > 1) and len(parser.buffer.buffer > 0):

        if best_action == 0:
            # print("SHIFT STACK {}".format(len(parser.stack.stack)))
            # print("SHIFT BUFFER {}".format(len(parser.buffer.buffer)))
            self.shift()
            parser.shift(self.shift_embedding)
        elif best_action == 1:
            # print("REDUCE-L STACK {}".format(len(parser.stack.stack)))
            # print("REDUCE-L BUFFER {}".format(len(parser.buffer.buffer)))
            self.reduce_l()
            parser.reduce_l(self.reduce_l_embedding)
        else:
            # print("REDUCE-R STACK {}".format(len(parser.stack.stack)))
            # print("REDUCE-R BUFFER {}".format(len(parser.buffer.buffer)))
            self.reduce_r()
            parser.reduce_r(self.reduce_r_embedding)
        # elif len(parser.buffer.buffer > 0):
        #    print("SHIFT STACK {}".format(len(parser.stack.stack)))
        #    print("SHIFT BUFFER {}".format(len(parser.buffer.buffer)))
        #    self.shift()
        #    parser.shift(self.shift_embedding)

        return parser

    def forward(self, x, head=None):
        # parse each sentence completley, then compare rels and heads
        #print(x[0][0, :])
        x_emb = self.dropout(self.get_embeddings(x))
        sent_lens = (x[0] != 0).sum(-1)
        # shape is [batch_size,3 concatenaded embeddings]
        # h_t = torch.zeros((x_emb.shape[0], torch.max(sent_lens).item()))
        h_t = torch.zeros((x_emb.shape[0], torch.max(sent_lens).item(), self.embedding_size * 2))\
            .to(device=constants.device)
        for i, sentence in enumerate(x_emb):
            #print(sentence.shape)

            parser = ShiftReduceParser(sentence, self.embedding_size)
            # init stack_lstm pointer and buffer_lstm pointer
            if i > 0:
                self.stack_lstm.set_top(0)
                self.buffer_lstm.set_top(0)
            for word in sentence:
                self.buffer_lstm.create_and_push()
            self.buffer_lstm(sentence)
            # push ROOT to the stack
            parser.stack.push((self.get_embeddings(root), -1))
            self.shift()
            parser.shift(self.shift_embedding)
            steps = 0
            while not parser.is_parse_complete():
                steps += 1
                parser = self.parse_step(parser)

            # parsed_state = self.get_parser_state(parser)
            heads = parser.get_heads()
            # action_emb = self.get_action_embeddings(parser.action_history)
            h_t[i, :, :] = heads  # self.word_embeddings(heads)

        # h_t = self.get_embeddings(h_t)
        # print(h_t.shape)

        #print(h_t[0])
        # h_t = self.run_lstm(h_t, sent_lens)
        h_logits = self.get_head_logits(h_t, sent_lens)
        if head is None:
            head = h_logits.argmax(-1)

        l_logits = self.get_label_logits(h_t, head)

        return h_logits, l_logits

    @staticmethod
    def loss(h_logits, l_logits, heads, rels):
        criterion_h = nn.CrossEntropyLoss(ignore_index=-1).to(device=constants.device)
        criterion_l = nn.CrossEntropyLoss(ignore_index=0).to(device=constants.device)
        loss = criterion_h(h_logits.reshape(-1, h_logits.shape[-1]), heads.reshape(-1))
        loss += criterion_l(l_logits.reshape(-1, l_logits.shape[-1]), rels.reshape(-1))
        return loss

    def get_head_logits(self, h_t, sent_lens):
        h_dep = self.dropout(F.relu(self.linear_arc_dep(h_t)))
        h_arc = self.dropout(F.relu(self.linear_arc_head(h_t)))

        h_logits = self.biaffine(h_arc, h_dep)

        # Zero logits for items after sentence length
        for i, sent_len in enumerate(sent_lens):
            h_logits[i, sent_len:, :] = 0
            h_logits[i, :, sent_len:] = 0

        return h_logits

    def get_label_logits(self, h_t, head):
        l_dep = self.dropout(F.relu(self.linear_label_dep(h_t)))
        l_head = self.dropout(F.relu(self.linear_label_head(h_t)))

        if self.training:
            assert head is not None, 'During training head should not be None'

        l_head = l_head.gather(dim=1, index=head.unsqueeze(2).expand(l_head.size()))
        l_logits = self.bilinear_label(l_dep, l_head)
        return l_logits

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
