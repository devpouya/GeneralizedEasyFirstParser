import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

import numpy as np

from utils import constants
from .base import BaseParser
from .modules import Biaffine, Bilinear, StackLSTM
from .word_embedding import WordEmbedding, ActionEmbedding
from ..algorithm.transition_parsers import ShiftReduceParser, arc_standard, arc_eager, hybrid, non_projective, \
    easy_first

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
        self.chooser_linear = nn.Linear(embedding_size * 2 * 3, embedding_size).to(device=constants.device)
        self.chooser_relu = nn.ReLU().to(device=constants.device)
        self.chooser_softmax = nn.Softmax().to(device=constants.device)
        self.dropout = nn.Dropout(dropout).to(device=constants.device)

        self.label_linear_h = nn.Linear(embedding_size * 2 * 3, label_size)
        self.label_linear_d = nn.Linear(embedding_size * 2 * 3, label_size)
        self.rels_linear = nn.Linear(label_size, rels.size)

        # self.linear_arc_dep = nn.Linear(self.embedding_size * 2, arc_size).to(device=constants.device)
        # self.linear_arc_head = nn.Linear(self.embedding_size * 2, arc_size).to(device=constants.device)
        ## self.arc_relu = nn.ReLU().to(device=constants.device)
        ## self.linear_arc = nn.Linear(arc_size*2,arc_size).to(device=constants.device)
        # self.biaffine = Biaffine(arc_size, arc_size)
        self.linear_label_dep = nn.Linear(self.embedding_size * 2, label_size).to(device=constants.device)
        self.linear_label_head = nn.Linear(self.embedding_size * 2, label_size).to(device=constants.device)
        ## self.linear_label = nn.Linear(label_size*2, label_size).to(device=constants.device)
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
        # need to get legal actions here
        legal_actions, kept_ind = self.legal_action(parser)
        prob = nn.Softmax(dim=0)(torch.matmul(legal_actions, prob.permute(1, 0)))
        # print(prob.shape)
        # need to pad probs back to original length with zeros
        if len(kept_ind) < len(self.actions):
            tmp = torch.zeros((len(self.actions), prob.shape[1])).to(device=constants.device)
            tmp[kept_ind, :] = prob
            prob = tmp
        return torch.argmax(prob).item(), prob

    def update_head_prob(self, probs, parser):
        pass

    def legal_action(self, parser):
        pass

    def get_parser_state(self, parser):
        parser_stack = parser.get_stack_content()
        parser_buffer = parser.get_buffer_content()
        if len(parser_stack) > 1:
            stack_state = torch.stack(parser_stack)
        else:
            stack_state = parser_stack[0]

        if len(parser_buffer) > 1:

            buffer_state = torch.stack(parser_buffer)
        elif len(parser_buffer) == 0:
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
        parser_states = torch.zeros((x_emb.shape[0], self.embedding_size * 2 * 3))
        heads_list = torch.zeros((x_emb.shape[0], torch.max(sent_lens).item()), dtype=torch.int).to(
            device=constants.device)
        head_probs = torch.zeros((x_emb.shape[0], torch.max(sent_lens).item(), torch.max(sent_lens).item())) \
            .to(device=constants.device)
        for i, sentence in enumerate(x_emb):
            parser = ShiftReduceParser(sentence, self.embedding_size)
            # init stack_lstm pointer and buffer_lstm pointer
            #if i > 0:
            #    self.stack_lstm.set_top(0)
            #    self.buffer_lstm.set_top(0)
            for word in sentence:
                self.buffer_lstm.push()
                self.buffer_lstm(word)
            # self.buffer_lstm(sentence)
            # push ROOT to the stack
            parser.stack.push((self.get_embeddings(root), -1))
            self.stack_lstm.push()
            self.stack_lstm(self.get_embeddings(root))
            self.shift()
            parser.shift(self.shift_embedding)
            actions_taken = []
            while not parser.is_parse_complete():
                parser = self.parse_step(parser)
            #if i == 3:
            #    self.stack_lstm.plot_structure(show=True)
            # print((parser.stack.get_len(),parser.buffer.get_len()))
            head_probs[i, :, :] = parser.head_probs  # nn.Softmax(dim=-1)(parser.head_probs)#parser.head_probs
            # print(parser.head_probs)
            # action_probs.append(torch.stack(actions_taken))

            # parsed_state = self.get_parser_state(parser)
            # head_i, heads_embed = parser.get_heads()
            # action_emb = self.get_action_embeddings(parser.action_history)
            # h_t[i, :, :] = parser.get_head_embeddings()  # heads_embed  # self.word_embeddings(heads)
            # predicted_heads[i, :, :] = parser.heads  # head_i
            # parser_states[i,:] = self.get_parser_state(parser)
            heads_list[i, :] = parser.head_list
        # print(h_t[0,:])
        l_logits = self.get_label_logits(head_probs, heads_list)

        return head_probs, l_logits

    @staticmethod
    def loss(h_logits, l_logits, heads, rels):
        criterion_h = nn.CrossEntropyLoss(ignore_index=-1).to(device=constants.device)
        criterion_l = nn.CrossEntropyLoss(ignore_index=0).to(device=constants.device)
        loss = criterion_h(h_logits.reshape(-1, h_logits.shape[-1]), heads.reshape(-1))

        # print(l_logits.shape)
        # print(rels.reshape(-1).shape)
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
        h_t = self.dropout(F.relu(nn.Linear(h_t.shape[1], self.embedding_size * 2).to(device=constants.device)
                                  (h_t)))
        l_dep = self.dropout(F.relu(self.linear_label_dep(h_t)))
        l_head = self.dropout(F.relu(self.linear_label_head(h_t)))

        if self.training:
            assert head is not None, 'During training head should not be None'
        # print("lhead shape {}".format(l_head.shape))
        # print("head shape {}".format(head.shape))
        # print("stuff {}".format(head.unsqueeze(2).expand(l_head.size()).shape))
        # l_head = l_head.gather(dim=1, index=head.unsqueeze(2).expand(l_head.size()))
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
            return best_action, probs
        elif len(parser.buffer.buffer) == 0:
            # can't shift
            probs[:, 0] = 0
            # probs = torch.cat([probs[:, 1], probs[:, 2]])
            return torch.argmax(probs).item(), probs
        else:
            # can only shift
            probs[:, 1:] = 0
            return 0, probs

    def legal_action(self, parser):
        all_actions = self.action_embeddings.weight
        if (parser.stack.get_len() > 1) and (parser.buffer.get_len() > 0):
            return all_actions, range(len(self.actions))
        elif parser.buffer.get_len() == 0:
            # can't shift
            return all_actions[1:, :], [1, 2]
        else:
            # can only shift
            return all_actions[0, :].unsqueeze(0), [0]

    def update_head_prob(self, probs, parser):
        if parser.stack.get_len() >= 2:
            top = parser.stack.top()
            second = parser.stack.second()
            parser.head_probs[:, top[1], second[1]] = probs[1]
            parser.head_probs[:, second[1], top[1]] = probs[2]

    def parse_step(self, parser):
        best_action, probs = self.decide_action(parser)
        self.update_head_prob(probs, parser)
        # best_action, probs = self.best_legal_action(best_action, parser, probs)
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

    def update_head_prob(self, probs, parser):
        if parser.stack.get_len() >= 1 and parser.buffer.get_len() >= 1:
            top = parser.stack.top()
            buffer_first = parser.buffer.left()
            parser.head_probs[:, buffer_first[1], top[1]] = probs[1]
            parser.head_probs[:, top[1], buffer_first[1]] = probs[2]

    def legal_action(self, parser):
        # first check the conditions
        all_actions = self.action_embeddings.weight

        stack_non_empty = len(parser.stack.stack) > 1
        buffer_non_empty = len(parser.buffer.buffer) > 0
        if stack_non_empty:
            stack_top = parser.stack.top()
            top_has_incoming = parser.arcs.has_incoming(stack_top)
            # left-arc eager and right-arc eager need buffer to be non-empty
            if not buffer_non_empty:
                # can only reduce
                return all_actions[3, :].unsqueeze(0), [3]
            else:
                if top_has_incoming:
                    # legal_action_probs = torch.index_select(probs, -1, torch.tensor([0, 2, 3]).to(
                    #    device=constants.device))  # legal action ids
                    # map_index2action = {0: 0, 1: 2, 2: 3}
                    ind = [0, 2, 3]
                    return all_actions[ind, :], ind
                else:  # if buffer_non_empty:
                    # legal_action_probs = probs[:, :3]
                    # map_index2action = {0: 0, 1: 1, 2: 2}
                    ind = range(2)
                    return all_actions[ind, :], ind
        else:
            if buffer_non_empty:
                return all_actions[0, :].unsqueeze(0), [0]
            else:
                parser.is_parse_complete(special_op=True)

    def parse_step(self, parser):
        best_action, probs = self.decide_action(parser)
        # best_action = self.best_legal_action(best_action, parser, probs)
        self.update_head_prob(probs, parser)
        if best_action == 0:
            self.shift()
            parser.shift(self.shift_embedding)
        elif best_action == 1:
            self.reduce_l()
            item = parser.left_arc_eager(self.reduce_l_embedding)
            self.stack_lstm(item)
        elif best_action == 2:
            self.reduce_r()
            item = parser.right_arc_eager(self.reduce_r_embedding)
            self.stack_lstm(item)
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

    def update_head_prob(self, probs, parser):
        if parser.stack.get_len() >= 1 and parser.buffer.get_len() >= 1:
            top = parser.stack.top()
            buffer_first = parser.buffer.left()
            parser.head_probs[:, buffer_first[1], top[1]] = probs[1]
            parser.head_probs[:, top[1], buffer_first[1]] = probs[2]

    def legal_action(self, parser):
        all_actions = self.action_embeddings.weight
        if (parser.stack.get_len() > 1) and (parser.buffer.get_len() > 0):
            return all_actions, range(len(self.actions))
        elif parser.buffer.get_len() == 0:
            return all_actions[1:,:], [1,2]
        else:
            return all_actions[0,:].unsqueeze(0),[0]

    def parse_step(self, parser):
        best_action, probs = self.decide_action(parser)
        self.update_head_prob(probs, parser)
        if best_action == 0:
            self.shift()
            parser.shift(self.shift_embedding)
        elif best_action == 1:
            self.reduce_l()
            item = parser.left_arc_hybrid(self.reduce_l_embedding)
            self.stack_lstm(item)
        else:
            self.reduce_r()
            item = parser.reduce_r(self.reduce_r_embedding)
            self.stack_lstm(item)
        return parser


class NonProjectiveStackLSTM(ExtendibleStackLSTMParser):
    def __init__(self, vocabs, embedding_size, hidden_size, arc_size, label_size,
                 nlayers=3, dropout=0.33, pretrained_embeddings=None, transition_system=non_projective):
        super().__init__(vocabs, embedding_size, hidden_size, arc_size, label_size,
                         nlayers=nlayers, dropout=dropout, pretrained_embeddings=pretrained_embeddings,
                         transition_system=transition_system)

        self.shift_embedding = self.action_embeddings(torch.LongTensor([0]).to(device=constants.device))
        self.reduce_l_embedding = self.action_embeddings(torch.LongTensor([1]).to(device=constants.device))
        self.reduce_r_embedding = self.action_embeddings(torch.LongTensor([2]).to(device=constants.device))
        self.reduce_l2_embedding = self.action_embeddings(torch.LongTensor([3]).to(device=constants.device))
        self.reduce_r2_embedding = self.action_embeddings(torch.LongTensor([4]).to(device=constants.device))

        self.shift_embedding = self.shift_embedding.reshape(1, self.shift_embedding.shape[0],
                                                            self.shift_embedding.shape[1])
        self.reduce_l_embedding = self.reduce_l_embedding.reshape(1, self.reduce_l_embedding.shape[0],
                                                                  self.reduce_l_embedding.shape[1])
        self.reduce_r_embedding = self.reduce_r_embedding.reshape(1, self.reduce_r_embedding.shape[0],
                                                                  self.reduce_r_embedding.shape[1])
        self.reduce_l2_embedding = self.reduce_l2_embedding.reshape(1, self.reduce_l2_embedding.shape[0],
                                                                    self.reduce_l2_embedding.shape[1])
        self.reduce_r2_embedding = self.reduce_r2_embedding.reshape(1, self.reduce_r2_embedding.shape[0],
                                                                    self.reduce_r2_embedding.shape[1])

    def shift(self):
        self.stack_lstm.push()
        self.buffer_lstm.pop()
        self.action_lstm(self.shift_embedding)

    def reduce_any(self, action):
        self.stack_lstm.pop()
        self.action_lstm(action)

    def best_legal_action(self, best_action, parser, probs):
        buffer_non_empty = len(parser.buffer.buffer) > 0
        stack_gt_3 = len(parser.stack.stack) >= 3
        # stack_lt_2 = len(parser.stack.stack) <= 2
        stack_lt_1 = len(parser.stack.stack) <= 1
        if buffer_non_empty:
            if stack_gt_3:
                return best_action
            elif stack_lt_1:
                return 0
            else:
                probs = probs[:, :3]
                return torch.argmax(probs).item()
        else:
            probs = probs[:, 1:]
            ind2_act = {i: i + 1 for i in range(4)}
            if stack_gt_3:
                return ind2_act[torch.argmax(probs).item()]
            else:
                probs = probs[:, :2]
                return ind2_act[torch.argmax(probs).item()]

    def parse_step(self, parser):
        best_action, probs = self.decide_action(parser)
        best_action = self.best_legal_action(best_action, parser, probs)
        # can avoid all this if, else stuff with a list of actions since code is all the same lol
        if best_action == 0:
            self.shift()
            parser.shift(self.shift_embedding)
        elif best_action == 1:
            self.reduce_any(self.reduce_l_embedding)
            item = parser.reduce_l(self.reduce_l_embedding)
            self.stack_lstm(item)
        elif best_action == 2:
            self.reduce_any(self.reduce_r_embedding)
            item = parser.reduce_r(self.reduce_r_embedding)
            self.stack_lstm(item)
        elif best_action == 3:
            self.reduce_any(self.reduce_l2_embedding)
            item = parser.left_arc_second(self.reduce_l2_embedding)
            self.stack_lstm(item)
        elif best_action == 4:
            self.reduce_any(self.reduce_r2_embedding)
            item = parser.right_arc_second(self.reduce_r2_embedding)
            self.stack_lstm(item)

        return parser


class EasyFirstStackLSTM(ExtendibleStackLSTMParser):
    pass
