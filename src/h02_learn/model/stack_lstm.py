import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

import numpy as np

from utils import constants
from .base import BaseParser
from .modules import Biaffine, Bilinear, StackLSTM
from .word_embedding import WordEmbedding, ActionEmbedding
from ..algorithm.transition_parsers import ShiftReduceParser, arc_standard, arc_eager, hybrid

# CONSTANTS and NAMES
# root used to initialize the stack with \sigma_0
root = (torch.tensor(1).to(device=constants.device), torch.tensor(1).to(device=constants.device))


class ExtendibleStackLSTMParser(BaseParser):
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
        self.actions = transition_system[0]  # [shift, reduce_l, reduce_r]
        self.transition_ids = transition_system[1]  # arc_standard
        _, _, rels = vocabs

        # continous representations
        self.word_embeddings, self.tag_embeddings, self.action_embeddings = \
            self.create_embeddings(vocabs, pretrained=pretrained_embeddings)

        # stack lstms
        self.buffer_lstm = StackLSTM(embedding_size * 2, int(hidden_size / 2), dropout=(dropout if nlayers > 1 else 0),
                                     batch_first=True, bidirectional=False)
        self.stack_lstm = StackLSTM(embedding_size * 2, int(hidden_size / 2), dropout=(dropout if nlayers > 1 else 0),
                                    batch_first=True, bidirectional=False)
        self.action_lstm = nn.LSTM(embedding_size, int(hidden_size / 2), dropout=(dropout if nlayers > 1 else 0),
                                   batch_first=True, bidirectional=False)

        # mlp for deciding actions
        self.chooser_linear = nn.Linear(embedding_size * 2 * 3, len(self.actions)).to(device=constants.device)
        self.chooser_relu = nn.ReLU().to(device=constants.device)
        self.chooser_softmax = nn.Softmax().to(device=constants.device)
        self.dropout = nn.Dropout(dropout).to(device=constants.device)

        self.linear_arc_dep = nn.Linear(self.embedding_size * 2, arc_size).to(device=constants.device)
        self.linear_arc_head = nn.Linear(self.embedding_size * 2, arc_size).to(device=constants.device)
        # self.arc_relu = nn.ReLU().to(device=constants.device)
        # self.linear_arc = nn.Linear(arc_size*2,arc_size).to(device=constants.device)
        self.biaffine = Biaffine(arc_size, arc_size)

        self.linear_label_dep = nn.Linear(self.embedding_size * 2, label_size).to(device=constants.device)
        self.linear_label_head = nn.Linear(self.embedding_size * 2, label_size).to(device=constants.device)
        # self.label_relu = nn.ReLU().to(device=constants.device)
        # self.linear_label = nn.Linear(label_size*2, label_size).to(device=constants.device)
        self.bilinear_label = Bilinear(label_size, label_size, rels.size)

    def create_embeddings(self, vocabs, pretrained=None):
        words, tags, _ = vocabs
        word_embeddings = WordEmbedding(words, self.embedding_size, pretrained=pretrained)
        tag_embeddings = nn.Embedding(tags.size, self.embedding_size)
        action_embedding = ActionEmbedding(self.actions, self.embedding_size).embedding
        return word_embeddings, tag_embeddings, action_embedding

    def get_embeddings(self, x):
        return torch.cat([self.word_embeddings(x[0]), self.tag_embeddings(x[1])], dim=-1)

    def get_action_embeddings(self, action_history):
        # ret_emb = torch.empty((len(action_history), self.action_embeddings.embedding_dim))
        # action_history = torch.stack(action_history)
        # for i, act in enumerate(action_history):
        #    #act_emb = self.action_embeddings(act)
        #    ret_emb[i,:] = act

        return torch.stack(action_history).squeeze_(0)

    def get_stack_or_buffer_embeddings(self, structure):
        # ret_emb = torch.empty((len(structure), self.embedding_size))
        ret_emb = self.get_embeddings(structure)

        return ret_emb

    def decide_action(self, parser):

        parser_state = self.get_parser_state(parser)

        prob = self.chooser_linear(parser_state)
        prob = self.chooser_relu(prob)

        return torch.argmax(prob).item(), prob

    def get_parser_state(self, parser):
        parser_stack = parser.get_stack_content()
        parser_buffer = parser.get_buffer_content()
        if len(parser_stack) > 1:
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

        action_embeddings = action_history[-1]
        with torch.no_grad():
            so = self.stack_lstm(stack_state)[0]
            bo = self.buffer_lstm(buffer_state)[0]

            ao = self.action_lstm(action_embeddings)[0].reshape(1, bo.shape[1])
            state = torch.cat([so, bo, ao], dim=-1)

        return state

    def best_legal_action(self, best_action, parser, probs):
        pass

    def parse_step(self, parser):
        pass

    def shift(self):
        pass

    def forward(self, x, head=None):
        x_emb = self.dropout(self.get_embeddings(x))
        sent_lens = (x[0] != 0).sum(-1)
        h_t = torch.zeros((x_emb.shape[0], torch.max(sent_lens).item(), self.embedding_size * 2)) \
            .to(device=constants.device)
        predicted_heads = torch.zeros((x_emb.shape[0], torch.max(sent_lens).item(), torch.max(sent_lens).item())) \
            .to(device=constants.device)
        for i, sentence in enumerate(x_emb):
            parser = ShiftReduceParser(sentence, self.embedding_size)
            # init stack_lstm pointer and buffer_lstm pointer
            if i > 0:
                self.stack_lstm.set_top(0)
                self.buffer_lstm.set_top(0)
            for word in sentence:
                self.buffer_lstm.push()
                self.buffer_lstm(word)
            # self.buffer_lstm(sentence)
            # push ROOT to the stack
            parser.stack.push((self.get_embeddings(root), -1))
            self.shift()
            parser.shift(self.shift_embedding)
            while not parser.is_parse_complete():
                parser = self.parse_step(parser)

            # parsed_state = self.get_parser_state(parser)
            head_i, heads_embed = parser.get_heads()
            # action_emb = self.get_action_embeddings(parser.action_history)
            h_t[i, :, :] = heads_embed  # self.word_embeddings(heads)
            predicted_heads[i, :, :] = head_i
        # print(h_t[0,:])
        h_logits = self.get_head_logits(h_t, sent_lens)
        if head is None:
            head = h_logits.argmax(-1)
        l_logits = self.get_label_logits(h_t, head)
        return predicted_heads, l_logits

    @staticmethod
    def loss(h_logits, l_logits, heads, rels):
        # loss over words fuck it
        # combine batch*sent_len

        # h_logits = h_logits.reshape(-1, h_logits.shape[-1])
        criterion_h = nn.CrossEntropyLoss(ignore_index=-1).to(device=constants.device)
        criterion_l = nn.CrossEntropyLoss(ignore_index=0).to(device=constants.device)
        loss = criterion_h(h_logits.reshape(h_logits.shape[0]*h_logits.shape[1],h_logits.shape[2]), heads.reshape(-1))
        loss += criterion_l(l_logits.reshape(-1, l_logits.shape[-1]), rels.reshape(-1))
        return loss

    def get_head_logits(self, h_t, sent_lens):
        h_dep = self.dropout(F.relu(self.linear_arc_dep(h_t)))
        h_arc = self.dropout(F.relu(self.linear_arc_head(h_t)))

        h_logits = self.biaffine(h_arc, h_dep)
        # h_logits = self.dropout(F.relu(self.linear_arc(torch.cat([h_arc, h_dep],dim=-1))))

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
        # l_logits = self.dropout(F.relu(self.linear_label(torch.cat([l_dep, l_head],dim=-1))))
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


class ArcStandardStackLSTM(ExtendibleStackLSTMParser):
    def __init__(self, vocabs, embedding_size, hidden_size, arc_size, label_size,
                 nlayers=3, dropout=0.33, pretrained_embeddings=None, transition_system=arc_standard):
        super().__init__(vocabs, embedding_size, hidden_size, arc_size, label_size,
                         nlayers=nlayers, dropout=dropout, pretrained_embeddings=pretrained_embeddings,
                         transition_system=transition_system)

        self.shift_embedding = self.action_embeddings(torch.LongTensor([0]).to(device=constants.device))
        self.reduce_l_embedding = self.action_embeddings(torch.LongTensor([1]).to(device=constants.device))
        self.reduce_r_embedding = self.action_embeddings(torch.LongTensor([2]).to(device=constants.device))

        self.shift_embedding = self.shift_embedding.reshape(1, self.shift_embedding.shape[0],
                                                            self.shift_embedding.shape[1])
        self.reduce_l_embedding = self.reduce_l_embedding.reshape(1, self.reduce_l_embedding.shape[0],
                                                                  self.reduce_l_embedding.shape[1])
        self.reduce_r_embedding = self.reduce_r_embedding.reshape(1, self.reduce_r_embedding.shape[0],
                                                                  self.reduce_r_embedding.shape[1])

    def shift(self):
        self.stack_lstm.push()
        self.buffer_lstm.pop()
        self.action_lstm(self.shift_embedding)

    def reduce_l(self):
        self.stack_lstm.pop()
        self.action_lstm(self.reduce_l_embedding)

    def reduce_r(self):
        self.stack_lstm.pop()
        self.action_lstm(self.reduce_r_embedding)

    def best_legal_action(self, best_action, parser, probs):
        if (len(parser.stack.stack) > 1) and (len(parser.buffer.buffer) > 0):
            return best_action
        elif len(parser.buffer.buffer) == 0:
            probs = torch.cat([probs[:, 1], probs[:, 2]])
            return torch.argmax(probs).item() + 1
        else:
            return 0

    def parse_step(self, parser):
        best_action, probs = self.decide_action(parser)
        best_action = self.best_legal_action(best_action, parser, probs)
        if best_action == 0:
            self.shift()
            shifted = parser.shift(self.shift_embedding)
            self.stack_lstm(shifted)
        elif best_action == 1:
            self.reduce_l()
            rep = parser.reduce_l(self.reduce_l_embedding)
            self.stack_lstm(rep)
        else:
            self.reduce_r()
            rep = parser.reduce_r(self.reduce_r_embedding)
            self.stack_lstm(rep)
        return parser


class ArcEagerStackLSTM(ExtendibleStackLSTMParser):
    def __init__(self, vocabs, embedding_size, hidden_size, arc_size, label_size,
                 nlayers=3, dropout=0.33, pretrained_embeddings=None, transition_system=arc_eager):
        super().__init__(vocabs, embedding_size, hidden_size, arc_size, label_size,
                         nlayers=nlayers, dropout=dropout, pretrained_embeddings=pretrained_embeddings,
                         transition_system=transition_system)

        self.shift_embedding = self.action_embeddings(torch.LongTensor([0]).to(device=constants.device))
        self.reduce_l_embedding = self.action_embeddings(torch.LongTensor([1]).to(device=constants.device))
        self.reduce_r_embedding = self.action_embeddings(torch.LongTensor([2]).to(device=constants.device))
        self.reduce_embedding = self.action_embeddings(torch.LongTensor([3]).to(device=constants.device))

        self.shift_embedding = self.shift_embedding.reshape(1, self.shift_embedding.shape[0],
                                                            self.shift_embedding.shape[1])
        self.reduce_l_embedding = self.reduce_l_embedding.reshape(1, self.reduce_l_embedding.shape[0],
                                                                  self.reduce_l_embedding.shape[1])
        self.reduce_r_embedding = self.reduce_r_embedding.reshape(1, self.reduce_r_embedding.shape[0],
                                                                  self.reduce_r_embedding.shape[1])
        self.reduce_embedding = self.reduce_embedding.reshape(1, self.reduce_embedding.shape[0],
                                                              self.reduce_embedding.shape[1])

    def shift(self):
        self.stack_lstm.push()
        self.buffer_lstm.pop()
        self.action_lstm(self.shift_embedding)

    def reduce_l(self):
        self.stack_lstm.pop()
        self.action_lstm(self.reduce_l_embedding)

    def reduce_r(self):
        self.buffer_lstm.pop()
        self.stack_lstm.push()
        self.action_lstm(self.reduce_r_embedding)

    def reduce(self):
        self.stack_lstm.pop()
        self.action_lstm(self.reduce_embedding)

    def best_legal_action(self, best_action, parser, probs):
        # first check the conditions
        stack_non_empty = len(parser.stack.stack) > 1
        buffer_non_empty = len(parser.buffer.buffer) > 0
        if stack_non_empty:
            stack_top = parser.stack.top()
            top_has_incoming = parser.arcs.has_incoming(stack_top)
            # left-arc eager and right-arc eager need buffer to be non-empty
            if not buffer_non_empty:
                # can only reduce
                return 3
            else:
                if top_has_incoming:
                    legal_action_probs = torch.index_select(probs, -1, torch.tensor([0, 2, 3]).to(
                        device=constants.device))  # legal action ids
                    map_index2action = {0: 0, 1: 2, 2: 3}
                else:  # if buffer_non_empty:
                    legal_action_probs = probs[:, :3]
                    map_index2action = {0: 0, 1: 1, 2: 2}
                return map_index2action[torch.argmax(legal_action_probs, dim=-1).item()]

        else:
            if buffer_non_empty:
                return 0
            else:
                parser.is_parse_complete(special_op=True)

    def parse_step(self, parser):
        best_action, probs = self.decide_action(parser)
        best_action = self.best_legal_action(best_action, parser, probs)
        if best_action == 0:
            self.shift()
            parser.shift(self.shift_embedding)
        elif best_action == 1:
            self.reduce_l()
            parser.left_arc_eager(self.reduce_l_embedding)
        elif best_action == 2:
            self.reduce_r()
            parser.right_arc_eager(self.reduce_r_embedding)
        else:
            self.reduce()
            parser.reduce(self.reduce_embedding)
        return parser


class HybridStackLSTM(ExtendibleStackLSTMParser):
    def __init__(self, vocabs, embedding_size, hidden_size, arc_size, label_size,
                 nlayers=3, dropout=0.33, pretrained_embeddings=None, transition_system=hybrid):
        super().__init__(vocabs, embedding_size, hidden_size, arc_size, label_size,
                         nlayers=nlayers, dropout=dropout, pretrained_embeddings=pretrained_embeddings,
                         transition_system=transition_system)

        self.shift_embedding = self.action_embeddings(torch.LongTensor([0]).to(device=constants.device))
        self.reduce_l_embedding = self.action_embeddings(torch.LongTensor([1]).to(device=constants.device))
        self.reduce_r_embedding = self.action_embeddings(torch.LongTensor([2]).to(device=constants.device))

        self.shift_embedding = self.shift_embedding.reshape(1, self.shift_embedding.shape[0],
                                                            self.shift_embedding.shape[1])
        self.reduce_l_embedding = self.reduce_l_embedding.reshape(1, self.reduce_l_embedding.shape[0],
                                                                  self.reduce_l_embedding.shape[1])
        self.reduce_r_embedding = self.reduce_r_embedding.reshape(1, self.reduce_r_embedding.shape[0],
                                                                  self.reduce_r_embedding.shape[1])

    def shift(self):
        self.stack_lstm.push()
        self.buffer_lstm.pop()
        self.action_lstm(self.shift_embedding)

    def reduce_l(self):
        self.stack_lstm.pop()
        self.action_lstm(self.reduce_l_embedding)

    def reduce_r(self):
        self.stack_lstm.pop()
        self.action_lstm(self.reduce_r_embedding)

    def best_legal_action(self, best_action, parser, probs):
        if (len(parser.stack.stack) > 1) and (len(parser.buffer.buffer) > 0):
            return best_action
        elif len(parser.buffer.buffer) == 0:
            probs = torch.cat([probs[:, 1], probs[:, 2]])
            return torch.argmax(probs).item() + 1
        else:
            return 0

    def parse_step(self, parser):
        best_action, probs = self.decide_action(parser)
        best_action = self.best_legal_action(best_action, parser, probs)
        if best_action == 0:
            self.shift()
            parser.shift(self.shift_embedding)
        elif best_action == 1:
            self.reduce_l()
            parser.left_arc_hybrid(self.reduce_l_embedding)
        else:
            self.reduce_r()
            parser.reduce_r(self.reduce_r_embedding)
        return parser
