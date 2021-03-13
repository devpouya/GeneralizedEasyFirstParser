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
from .modules import StackRNN, StackCell
from transformers import BertModel

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

def pairwise(iterable):
    a = iter(iterable)
    return zip(a,a)

# root = (torch.tensor(1).to(device=constants.device), torch.tensor(1).to(device=constants.device))

class SoftmaxLegal(nn.Module):
    # __constants__ = ['dim']
    # dim: Optional[int]
    def __init__(self, dim, parser, actions,is_relation=False):
        super(SoftmaxLegal, self).__init__()
        self.dim = dim
        # self.parser = parser
        self.actions = actions
        self.num_actions = len(actions)
        self.indices = self.legal_indices(parser)
        #elf.relate = self.rel_or_not()
        self.is_relation = is_relation
        # self.inds_zero = list(set(range(self.num_actions)).difference(set(self.indices)))

    def legal_indices(self, parser):
        if len(parser.stack) < 2:
            return [0]
        elif len(parser.buffer) < 1:
            return [1, 2]
        else:

            return [0, 1, 2]

    def rel_or_not(self):
        if self.indices == [0]:
            return False
        else:
            return True

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input):
        if self.is_relation:
            if self.rel_or_not():
                tmp = F.softmax(input[1:], self.dim, _stacklevel=5)
                ret = torch.zeros_like(input)
                ret[1:] = tmp  #.detach().clone()
                return ret
            else:
                tmp = F.softmax(input[0], self.dim, _stacklevel=5)
                ret = torch.zeros_like(input)
                ret[0] = tmp  # .detach().clone()
                return ret
        else:
            tmp = F.softmax(input[self.indices], self.dim, _stacklevel=5)
            # print(tmp)
            ret = torch.zeros_like(input)
            ret[self.indices] = tmp  # .detach().clone()
            return ret  # F.softmax(input, self.dim, _stacklevel=5)

    def extra_repr(self):
        return 'dim={dim}'.format(dim=self.dim)





class NeuralTransitionParser(BaseParser):
    def __init__(self, vocabs, embedding_size, hidden_size, batch_size,
                 nlayers=3, dropout=0.33, transition_system=None):
        super().__init__()
        # basic parameters
        self.vocabs = vocabs
        self.embedding_size = 768#embedding_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.nlayers = nlayers
        self.dropout_prob = dropout
        self.bert = BertModel.from_pretrained('bert-base-cased',output_hidden_states=True).to(device=constants.device)
        #self.bert.eval()

        for param in self.bert.parameters():
            param.requires_grad = True

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
        _, _, rels = vocabs
        # self.num_actions = 1 + rels.size * 2
        # self.num_actions = 3
        self.num_rels = rels.size #+ 1  # 0 is no rel (for shift)
        self.action_embeddings_size = self.num_rels*2+1
        self.tag_rel_embeddings_size = embedding_size

        self.tag_embeddings, self.action_embeddings, self.rel_embeddings = self.create_embeddings(vocabs)
        # self.root = (torch.tensor(1).to(device=constants.device), torch.tensor(1).to(device=constants.device))

        """
            MLP[stack_lstm_output,buffer_lstm_output,action_lstm_output] ---> decide next transition
        """

        # neural model parameters
        self.dropout = nn.Dropout(self.dropout_prob)
        # lstms
        stack_lstm_size = self.embedding_size+self.tag_rel_embeddings_size
        self.stack_lstm = nn.LSTMCell(stack_lstm_size, stack_lstm_size).to(device=constants.device)
        self.buffer_lstm = nn.LSTMCell(stack_lstm_size, stack_lstm_size).to(device=constants.device)
        self.action_lstm = nn.LSTMCell(self.action_embeddings_size, self.action_embeddings_size).to(device=constants.device)

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
        self.empty_initial_act = nn.Parameter(torch.zeros(1,  self.action_embeddings_size)).to(device=constants.device)
        self.action_bias = nn.Parameter(torch.Tensor(1, self.num_actions), requires_grad=True).to(
            device=constants.device)
        self.rel_bias = nn.Parameter(torch.Tensor(1, self.num_rels), requires_grad=True).to(device=constants.device)
        # MLP
        self.mlp_lin1 = nn.Linear(stack_lstm_size*2+self.action_embeddings_size,
                                  self.embedding_size).to(device=constants.device)

        self.mlp_lin1_rel = nn.Linear(stack_lstm_size*2+self.action_embeddings_size,
                                      self.embedding_size).to(device=constants.device)

        self.mlp_act = nn.Linear(self.embedding_size, self.num_actions).to(device=constants.device)
        self.mlp_rel = nn.Linear(self.embedding_size, self.num_rels).to(device=constants.device)
        torch.nn.init.xavier_uniform_(self.mlp_lin1.weight)

        torch.nn.init.xavier_uniform_(self.mlp_lin1_rel.weight)

        torch.nn.init.xavier_uniform_(self.mlp_act.weight)
        torch.nn.init.xavier_uniform_(self.mlp_rel.weight)

        self.linear_tree = nn.Linear(self.tag_rel_embeddings_size+2*stack_lstm_size, stack_lstm_size).to(device=constants.device)
        # self.linear_tree2 = nn.Linear(5 * embedding_size, 3 * embedding_size).to(device=constants.device)
        torch.nn.init.xavier_uniform_(self.linear_tree.weight)

        # torch.nn.init.xavier_uniform_(self.linear_tree2.weight)

        self.stack = StackCell(self.stack_lstm, self.lstm_init_state, self.lstm_init_state, self.dropout,
                              self.empty_initial)
        self.buffer = StackCell(self.buffer_lstm, self.lstm_init_state, self.lstm_init_state, self.dropout,
                               self.empty_initial)
        self.action = StackCell(self.action_lstm, self.lstm_init_state_actions, self.lstm_init_state_actions,
                               self.dropout,
                               self.empty_initial_act)

    def create_embeddings(self, vocabs):
        words, tags, rels = vocabs
        #word_embeddings = WordEmbedding(words, self.embedding_size, pretrained=pretrained)
        tag_embeddings = nn.Embedding(tags.size, self.tag_rel_embeddings_size)
        rel_embeddings = nn.Embedding(self.num_rels, self.tag_rel_embeddings_size,scale_grad_by_freq=True)

        #learned_embeddings = nn.Embedding(words.size, self.embedding_size)
        action_embedding = nn.Embedding(self.num_actions, self.action_embeddings_size,scale_grad_by_freq=True)
        return tag_embeddings, action_embedding, rel_embeddings

    def get_embeddings(self, x):
        return torch.cat([self.word_embeddings(x[0]), self.learned_embeddings(x[0]), self.tag_embeddings(x[1])],
                         dim=-1).to(device=constants.device)

    def get_bert_embeddings(self, mapping, sentence,tags):
        s = []#torch.zeros((mapping.shape[0]+1, sentence.shape[1])).to(device=constants.device)
        for start, end in pairwise(mapping):
            m = torch.mean(sentence[start:end+1,:],dim=0)
            s.append(m)
        s = torch.stack(s,dim=0).to(device=constants.device)

        #self.tag_embeddings()
        return torch.cat([s,tags],dim=-1).to(device=constants.device)

    def get_action_embed(self, act):
        idx = self.action2id[act]
        t = torch.tensor(idx, dtype=torch.long).to(device=constants.device)
        return self.action_embeddings(t).unsqueeze(0).unsqueeze(1).to(device=constants.device)

    def stacked_action_embeddings(self):
        l = []
        for act in self.actions:
            id = self.action2id[act]
            t = torch.tensor(id, dtype=torch.long).to(device=constants.device)
            l.append(self.action_embeddings(t))
        return torch.stack(l)

    def labeled_action_pairs(self, actions, relations):
        labeled_acts = []
        tmp_rels = relations.clone().detach().tolist()

        for act in actions:

            if act == 1 or act == 2:
                labeled_acts.append((act, tmp_rels[0]))
                tmp_rels.pop(0)
            else:
                labeled_acts.append((act, 0))

        return labeled_acts

    def parser_probabilities(self, parser):
        parser_state = torch.cat([self.stack.embedding().reshape(1,self.embedding_size+self.tag_rel_embeddings_size),
                                  self.buffer.embedding().reshape(1,self.embedding_size+self.tag_rel_embeddings_size),
                                  self.action.embedding().reshape(1,self.action_embeddings_size)], dim=-1)

        state1 = self.dropout(F.relu(self.mlp_lin1(parser_state))).squeeze(0)



        action_probabilities = nn.Softmax(dim=-1)(self.mlp_act(state1)).squeeze(0)
        #action_probabilities = SoftmaxLegal(dim=-1, parser=parser, actions=self.actions)(self.mlp_act(state1)).squeeze(
        #    0)

        state2 = self.dropout(F.relu(self.mlp_lin1_rel(parser_state))).squeeze(0)

        #rel_probabilities = SoftmaxLegal(dim=-1, parser=parser, actions=self.actions,is_relation=True)\
        #    (self.mlp_rel(state2)).squeeze(0)

        rel_probabilities = nn.Softmax(dim=-1)(self.mlp_rel(state2)).squeeze(0)
        return action_probabilities, rel_probabilities

    def parse_step_arc_standard(self, parser, labeled_transitions, mode):
        # get parser state
        action_probabilities, rel_probabilities = self.parser_probabilities(parser)
        # if labeled_transitions is not None:
        best_action = labeled_transitions[0].item()
        rel = labeled_transitions[1]

        action_target = torch.tensor([best_action], dtype=torch.long).to(device=constants.device)
        rel_target = torch.tensor([rel], dtype=torch.long).to(device=constants.device)
        rel_embed = self.rel_embeddings(rel_target).to(device=constants.device)

        if mode == 'eval':

            if len(parser.stack) < 2:
                # can't left or right
                best_action = 0  # torch.argmax(action_probabilities[:, 0], dim=-1).item()

            elif len(parser.buffer) < 1:
                # can't shift
                tmp = action_probabilities.clone().detach().to(device=constants.device)
                tmp[0] = -float('inf')
                best_action = torch.argmax(tmp, dim=-1).item()

            else:
                best_action = torch.argmax(action_probabilities, dim=-1).item()

            rel = torch.argmax(rel_probabilities, dim=-1).item()  # +1
            rel_ind = torch.tensor([rel], dtype=torch.long).to(device=constants.device)
            rel_embed = self.rel_embeddings(rel_ind).to(device=constants.device)

        # do the action
        if best_action == 0:
            # shift
            self.stack.push(self.buffer.pop())

            self.action.push(self.get_action_embed(constants.shift).squeeze(0))
            #action_state = self.action_lstm(self.get_action_embed(constants.shift))
            parser.shift()
        elif best_action == 1:

            # reduce-l

            self.action.push(self.get_action_embed(constants.shift).squeeze(0))
            ret = parser.reduce_l(rel, rel_embed,self.linear_tree)
            self.stack.pop(-2)
            self.stack.replace(ret.unsqueeze(0))


        elif best_action == 2:

            # reduce-r

            self.action.push(self.get_action_embed(constants.reduce_l).squeeze(0))
            ret = parser.reduce_r(rel, rel_embed,self.linear_tree)
            self.stack.pop()
            self.stack.replace(ret.unsqueeze(0))

        else:
            self.stack.pop()
            self.action.push(self.get_action_embed(constants.reduce_l).squeeze(0))
            elem = parser.stack.pop()
            parser.arcs.append((elem[1], elem[1], rel))

        return parser, (action_probabilities, rel_probabilities), (action_target, rel_target)


    def forward(self, x, transitions, relations, map, mode):

        #sent_lens = (x[0] != 0).sum(-1).to(device=constants.device)
        transit_lens = (transitions != -1).sum(-1).to(device=constants.device)
        #print(x[1])
        tags = self.tag_embeddings(x[1].to(device=constants.device))
        # average of last 4 hidden layers
        if mode == "eval":
            with torch.no_grad():
                self.bert.eval()
                out = self.bert(x[0].to(device=constants.device))[2]
        else:
            self.bert.train()
            out = self.bert(x[0].to(device=constants.device))[2]

        x_emb = torch.stack(out[-4:]).mean(0)

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
        # for testing
        # labeled_transitions = self.labeled_action_pairs(transitions[0], relations[0])
        # tr = [t.item() for (t,_) in labeled_transitions]
        # self.sanity_parse(tr,heads,sent_lens)
        action_state = self.lstm_init_state_actions
        for i, sentence in enumerate(x_emb):
            # initialize a parser
            mapping = map[i,map[i]!=-1]
            tag = self.tag_embeddings(x[1][i][x[1][i]!=0].to(device=constants.device))


            s = self.get_bert_embeddings(mapping,sentence,tag)
            curr_sentence_length = s.shape[0]
            curr_transition_length = transit_lens[i]
            s = s[:curr_sentence_length, :]

            labeled_transitions = self.labeled_action_pairs(transitions[i, :curr_transition_length],
                                                            relations[i, :curr_sentence_length])

            parser = ShiftReduceParser(s, self.embedding_size, self.transition_system)


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

    def loss_hinge(self, probs, targets, probs_rel, targets_rel):
        predictions = torch.argmax(probs, dim=-1).float()

        c1 = nn.MultiLabelMarginLoss()(predictions, targets.squeeze(2))

        return c1
    def loss(self, probs, targets, probs_rel, targets_rel):
        criterion1 = nn.CrossEntropyLoss().to(device=constants.device)

        criterion2 = nn.CrossEntropyLoss().to(device=constants.device)

        probs = probs.reshape(-1, probs.shape[-1])
        targets = targets.reshape(-1)
        targets = targets[probs[:, 0] != -1]
        probs = probs[probs[:, 0] != -1, :]

        probs_rel = probs_rel.reshape(-1, probs_rel.shape[-1])

        targets_rel = targets_rel.reshape(-1)

        probs_rel = probs_rel[targets_rel != 0, :]
        targets_rel = targets_rel[targets_rel != 0]
        targets_rel = targets_rel[probs_rel[:, 0] != -1]
        probs_rel = probs_rel[probs_rel[:, 0] != -1, :]

        loss = 0.5*criterion1(probs, targets)
        loss += 0.5*criterion2(probs_rel, targets_rel)
        return loss

    def get_args(self):
        return {
            'vocabs': self.vocabs,
            'embedding_size': self.embedding_size,
            'hidden_size': self.hidden_size,
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
                s = stack[-2]
                arcs.append((t, s))
                stack.pop(-2)
            elif act == 2:
                t = stack[-1]
                s = stack[-2]
                arcs.append((s, t))
                # buffer[0] = t
                stack.pop(-1)
            else:
                item = stack.pop(-1)
                arcs.append((item, item))
        arcs.append((0, 0))
        print(set(true_arcs) == set(arcs))



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
        action_probabilities, rel_probabilities = self.parser_probabilities(parser)
        best_action = labeled_transitions[0].item()
        rel = labeled_transitions[1]
        action_target = torch.tensor([best_action], dtype=torch.long).to(device=constants.device)
        rel_target = torch.tensor([rel], dtype=torch.long).to(device=constants.device)
        rel_embed = self.rel_embeddings(rel_target).to(device=constants.device)

        if mode == 'eval':

            if len(parser.stack) < 1:
                # can only shift
                best_action = 0  # torch.argmax(action_probabilities[:, 0], dim=-1).item()

            elif len(parser.stack) == 1 and len(parser.buffer) > 0:
                # can't right reduce
                tmp = action_probabilities.clone().detach().to(device=constants.device)
                tmp[:, 2] = -float('inf')
                best_action = torch.argmax(tmp, dim=-1).item()

            elif len(parser.buffer) > 0:
                tmp = action_probabilities.clone().detach().to(device=constants.device)
                best_action = torch.argmax(tmp, dim=-1).item()
            else:
                best_action = 2

            rel = torch.argmax(rel_probabilities, dim=-1).item()  # +1
            rel_ind = torch.tensor([rel], dtype=torch.long).to(device=constants.device)
            rel_embed = self.rel_embeddings(rel_ind).to(device=constants.device)

        if best_action == 0:
            self.action.push(self.get_action_embed(constants.shift))
            parser.shift()
            self.stack.push(self.buffer.pop())
        elif best_action == 1:
            self.action.push(self.get_action_embed(constants.left_arc_eager))
            parser.left_arc_eager(rel, rel_embed, self.linear_tree)
            self.stack.pop()

        elif best_action == 2:
            self.action.push(self.get_action_embed(constants.reduce_r))
            ret = parser.reduce_r(rel, rel_embed, self.linear_tree)
            self.stack.pop()
            self.stack.pop()
            self.stack.push(ret.unsqueeze(0).unsqueeze(1))
        else:
            self.stack.pop()
            self.action.push(self.get_action_embed(constants.left_arc_eager))
            elem = parser.stack.pop()
            parser.arcs.append((elem[1], elem[1], rel))

        return parser, (action_probabilities, rel_probabilities), (action_target, rel_target)
