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

        out, hidden = self.cell(expr, self.s[-1][1])
        self.s.append((out, hidden))

    def pop(self):
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
        self.num_actions = 3
        self.num_rels = rels.size + 1 # 0 is no rel (for shift)
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
        self.gaurd_act = torch.zeros((1, 1, self.embedding_size)).to(device=constants.device)
        self.empty_initial = nn.Parameter(torch.randn(self.batch_size, self.hidden_size))
        # MLP it's actually a one layer network
        self.mlp_lin1 = nn.Linear(int(self.hidden_size / 2) * 3,
                                  self.embedding_size * 2).to(device=constants.device)
        self.mlp_lin2 = nn.Linear(self.embedding_size * 2,
                                  self.embedding_size).to(device=constants.device)

        self.mlp_act = nn.Linear(self.embedding_size,self.num_actions).to(device=constants.device)
        self.mlp_rel = nn.Linear(self.embedding_size,self.num_rels).to(device=constants.device)

        torch.nn.init.kaiming_uniform_(self.mlp_lin1.weight,nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.mlp_lin2.weight,nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.mlp_act.weight,nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.mlp_rel.weight,nonlinearity='relu')

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
            ret = 0# self.num_actions - 1
        elif action == 1:
            ret = rel+1
        elif action == 2:
            ret = rel * 2+1
        else:
            # action == -2
            # 7 is root index
            ret = 1
        return torch.tensor([ret],dtype=torch.long).to(device=constants.device)

    def index2action(self, index):
        if index == 0:
            return 0
        elif index <= self.num_actions:
            return 1
        else:
            return 2

    def parse_step(self, parser, stack, buffer, action, labeled_transitions, mode):
        # get parser state
        parser_state = torch.cat([stack.embedding(), buffer.embedding(), action.embedding()], dim=-1)

        parser_state = F.relu(self.mlp_lin1(parser_state))
        #print(parser_state.shape)
        # self.parser_state = parser_state
        state = F.relu(self.mlp_lin2(parser_state))
        action_probabilities = nn.Softmax(dim=-1)(self.mlp_act(state)).squeeze(0)
        rel_probabilities = nn.Softmax(dim=-1)(self.mlp_rel(state)).squeeze(0)
        #print(action_probabilities)
        #print(rel_probabilities.shape)
        # action_probabilities = self.mlp_relu(action_probabilities)
        #action_probabilities = self.mlp_softmax(action_probabilities).squeeze(0) # .clone()
        # criterion_a = nn.CrossEntropyLoss().to(device=constants.device)
        if labeled_transitions is not None:
            best_action = labeled_transitions[0].item()
            rel = labeled_transitions[1]
            #target = self.action_rel2index(best_action, rel)
        else:
            best_action = -2
            rel = 1
            target = self.action_rel2index(-2, 1)
        if best_action != -2:
            action_target = torch.tensor([best_action],dtype=torch.long).to(device=constants.device)
        else:
            action_target = torch.tensor([1],dtype=torch.long).to(device=constants.device)

        if best_action == 0 or best_action == -2:
            rel = 0

        rel_target = torch.tensor([rel],dtype=torch.long).to(device=constants.device)
        # l = criterion_a(action_probabilities,target)

        if mode == 'eval':
            #final = False
            if len(parser.stack) < 1:
                # can't left or right
                best_action =torch.argmax(action_probabilities[:, 0], dim=-1).item()

            elif len(parser.buffer) == 1:
                # can't shift
                tmp = action_probabilities.clone().detach().to(device=constants.device)
                tmp[:, 0] = -float('inf')
                best_action = torch.argmax(tmp, dim=-1).item()

            elif len(parser.stack) == 1 and len(parser.buffer) == 0:
                best_action = torch.argmax(action_probabilities[:, 1], dim=-1).item()
                best_action = -2

                #final = True
            else:
                best_action = torch.argmax(action_probabilities, dim=-1).item()
            #if not final:
            #    best_action = self.index2action(index)
            #if best_action != 0:
            #    rel = index if index < self.num_actions-1 else int(index / 2)


        # do the action
        if best_action == 0:
            # shift
            stack.push(buffer.pop())
            action.push(self.shift_embedding)
            parser.shift()
        elif best_action == 1:
            # reduce-l
            stack.pop()
            action.push(self.reduce_l_embedding)
            ret = parser.reduce_l(parser_state, rel)
            stack.push(ret.unsqueeze(0).unsqueeze(1))

        elif best_action == 2:
            # reduce-r
            # buffer.push(stack.pop())  # not sure, should replace in buffer actually...
            action.push(self.reduce_r_embedding)
            ret = parser.reduce_r(parser_state, rel)
            stack.pop()
            stack.push(ret.unsqueeze(0).unsqueeze(1))
        else:
            stack.pop()
            elem = parser.stack.pop()
            parser.arcs.append((elem[1], elem[1], rel))


        return parser, (stack, buffer, action), (action_probabilities,rel_probabilities), (action_target,rel_target)

    def forward(self, x, transitions, relations,mode):
        stack = StackRNN(self.stack_lstm, self.lstm_init_state, self.lstm_init_state, self.dropout, self.empty_initial)
        buffer = StackRNN(self.buffer_lstm, self.lstm_init_state, self.lstm_init_state, self.dropout,
                          self.empty_initial)
        action = StackRNN(self.action_lsttm, self.lstm_init_state_actions, self.lstm_init_state_actions, self.dropout,
                          self.empty_initial)
        sent_lens = (x[0] != 0).sum(-1).to(device=constants.device)
        transit_lens = (transitions!=-1).sum(-1).to(device=constants.device)
        x_emb = self.get_embeddings(x)

        stack.push(self.gaurd)
        buffer.push(self.gaurd)
        action.push(self.gaurd_act)
        probs_action_batch = torch.ones((x_emb.shape[0], transitions.shape[1], self.num_actions)).to(device=constants.device)
        probs_rel_batch = torch.ones((x_emb.shape[0], transitions.shape[1], self.num_rels)).to(device=constants.device)
        targets_action_batch = torch.ones((x_emb.shape[0], transitions.shape[1], 1),dtype=torch.long).to(device=constants.device)
        targets_rel_batch = torch.ones((x_emb.shape[0], transitions.shape[1], 1),dtype=torch.long).to(device=constants.device)


        heads_batch = torch.ones((x_emb.shape[0], x_emb.shape[1]), requires_grad=False).to(device=constants.device)
        rels_batch = torch.ones((x_emb.shape[0], x_emb.shape[1]), requires_grad=False).to(device=constants.device)
        heads_batch *= -1
        rels_batch *= -1
        probs_rel_batch *= -1
        probs_action_batch *= -1
        targets_rel_batch *= -1
        targets_action_batch *= -1

        # for testing
        #labeled_transitions = self.labeled_action_pairs(transitions[0], relations[0])
        #tr = [t.item() for (t,_) in labeled_transitions]
        #self.sanity_parse(tr,heads,sent_lens)
        if mode == 'eval':
            probs_action_all_batches = []
            probs_rel_all_batches = []
            targets_action_all_batches = []
            targets_rel_all_batches = []
            max_steps = 0
        # parse every sentence in batch

        for i, sentence in enumerate(x_emb):
            # initialize a parser
            curr_sentence_length = sent_lens[i]
            curr_transition_length = transit_lens[i]
            sentence = sentence[:curr_sentence_length, :]

            # if mode == 'train':
            labeled_transitions = self.labeled_action_pairs(transitions[i, :curr_transition_length],
                                                            relations[i, :curr_sentence_length])

            parser = ShiftReduceParser(sentence, self.embedding_size, self.transition_system)
            # initialize buffer first
            for word in sentence:
                buffer.push(word.reshape(1, 1, word.shape[0]))

            if mode == 'train':

                for step in range(len(labeled_transitions)):
                    parser, configuration, probs, target = self.parse_step(parser, stack, buffer, action,
                                                                           labeled_transitions[step],
                                                                           mode)
                    (stack, buffer, action) = configuration
                    (action_probs,rel_probs) = probs
                    (action_target,rel_target) = target
                    probs_action_batch[i, step, :] = action_probs
                    probs_rel_batch[i, step, :] = rel_probs
                    targets_action_batch[i, step, :] = action_target
                    targets_rel_batch[i, step, :] = rel_target
                    # act_loss += l
                #print(self.check_if_good(parser.arcs, heads, sent_lens))
                # act_loss /= len(labeled_transitions)
                heads_batch[i, :sent_lens[i]] = parser.heads_from_arcs()[0]
                rels_batch[i, :sent_lens[i]] = parser.heads_from_arcs()[1]

            else:
                step = 0
                probs_action_this_batch = []
                probs_rel_this_batch = []
                targets_action_this_batch = []
                targets_rel_this_batch = []
                while not parser.is_parse_complete():
                    if step < len(labeled_transitions):
                        parser, configuration, probs, target = self.parse_step(parser, stack, buffer, action,
                                                                               labeled_transitions[step],
                                                                               mode)
                        #probs_batch[i, step, :] = probs
                        #targets_batch[i, step, :] = target
                        #probs_this_batch.append(probs)
                        #targets_this_batch.append(target)

                    else:
                        parser, configuration, probs, target = self.parse_step(parser, stack, buffer, action,
                                                                               None,
                                                                               mode)
                        #probs_this_batch.append(probs)
                        #targets_this_batch.append(target)


                    (stack, buffer, action) = configuration
                    (action_probs, rel_probs) = probs
                    (action_target, rel_target) = target
                    probs_action_this_batch.append(action_probs)
                    probs_rel_this_batch.append(rel_probs)
                    targets_action_this_batch.append(action_target)
                    targets_rel_this_batch.append(rel_target)
                    # act_loss += l
                    step += 1
                if step > max_steps:
                    max_steps = step
                probs_action_all_batches.append(torch.stack(probs_action_this_batch).permute(1,0,2))
                probs_rel_all_batches.append(torch.stack(probs_rel_this_batch).permute(1,0,2))
                targets_action_all_batches.append(torch.stack(targets_action_this_batch).permute(1,0))
                targets_rel_all_batches.append(torch.stack(targets_rel_this_batch).permute(1,0))
                # act_loss /= len(labeled_transitions)
                heads_batch[i, :sent_lens[i]] = parser.heads_from_arcs()[0]
                rels_batch[i, :sent_lens[i]] = parser.heads_from_arcs()[1]

        if mode == 'eval':
            probs_action_batch = torch.ones((x_emb.shape[0],max_steps,self.num_actions)).to(device=constants.device)
            probs_rel_batch = torch.ones((x_emb.shape[0],max_steps,self.num_rels)).to(device=constants.device)
            targets_action_batch = torch.ones((x_emb.shape[0],max_steps,1),dtype=torch.long).to(device=constants.device)
            targets_rel_batch = torch.ones((x_emb.shape[0],max_steps,1),dtype=torch.long).to(device=constants.device)
            probs_action_batch *= -1
            probs_rel_batch *= -1
            targets_action_batch *= -1
            targets_rel_batch *= -1
            for i in range(x_emb.shape[0]):
                probs_action_batch[i,:probs_action_all_batches[i].shape[1],:] = probs_action_all_batches[i]
                probs_rel_batch[i,:probs_rel_all_batches[i].shape[1],:] = probs_rel_all_batches[i]
                targets_action_batch[i,:targets_action_all_batches[i].shape[1],:] = targets_action_all_batches[i].unsqueeze(2)
                targets_rel_batch[i,:targets_rel_all_batches[i].shape[1],:] = targets_rel_all_batches[i].unsqueeze(2)
         # act_loss /= x_emb.shape[0]
        batch_loss = self.loss(probs_action_batch, targets_action_batch,probs_rel_batch,targets_rel_batch)
        return batch_loss, heads_batch, rels_batch

    def loss(self, probs, targets,probs_rel,targets_rel):
        criterion1 = nn.CrossEntropyLoss(reduction='mean').to(device=constants.device)
        criterion2 = nn.CrossEntropyLoss(reduction='mean').to(device=constants.device)

        loss = 0
        for i in range(probs.shape[0]):
            p = probs[i]
            p = p[p[:,0]!=-1,:]

            p2 = probs_rel[i]
            p2 = p2[p2[:, 0] != -1, :]

            t = targets[i].squeeze(1)
            #t = torch.tensor(targets[i].squeeze(1),dtype=torch.long)
            t = t[:p.shape[0]]
            loss += criterion1(p,t)#[p!=-1]

            t2 = targets_rel[i].squeeze(1)
            # t = torch.tensor(targets[i].squeeze(1),dtype=torch.long)
            t2 = t2[:p2.shape[0]]
            loss += criterion2(p2, t2)  # [p!=-1]

        loss /= probs.shape[0]
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

        #heads_proper = [0] + heads_proper

        sentence_proper_ind = list(range(len(heads_proper)))
        word2head = {w: h for (w, h) in zip(sentence_proper_ind, heads_proper)}
        true_arcs = get_arcs(word2head)

        better = [(u,v) for (u,v,_) in built]
        for (u,v) in true_arcs:
            if (u,v) in better:
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
