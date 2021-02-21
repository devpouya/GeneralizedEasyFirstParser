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


# taken from stack-lstm-ner (will give credit)
class StackRNN(object):
    def __init__(self, cell, initial_state,initial_hidden,dropout, p_empty_embedding=None):
        self.cell = cell
        self.dropout = dropout
        #self.s = [(initial_state, None)]
        self.s = [(initial_state,initial_hidden)]

        self.empty = None
        if p_empty_embedding is not None:
            self.empty = p_empty_embedding

    def push(self, expr, extra=None):
        #print(self.s[-1][0][0].shape)
        #print(expr.shape)
        #self.dropout(self.s[-1][0][0])
        #print("(seqlen,batchsize,indim) {}".format(expr.shape))
        #print("has to be a tuple {}".format(self.s[-1][0][0].shape))
        #print("has to be a tuple {}".format(self.s[-1][0][1].shape))
        #print(len(self.s[-1][0]))
        #self.s.append((self.cell(expr, (self.s[-1][0][0],self.s[-1][0][1])), extra))
        #print(len(self.s[-1]))
        out,hidden = self.cell(expr,self.s[-1][1])
        self.s.append((out,hidden))

    def pop(self):
        #x = self.s.pop()#[1]
        ##y = self.s.pop([0]
        #print("x {}".format(x[0]))
        #print("< {}".format(x[1]))
        return self.s.pop()[0]#[0]

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

        self.shift_embedding = self.action_embeddings(torch.LongTensor([0]).to(device=constants.device)).unsqueeze(0)
        self.reduce_l_embedding = self.action_embeddings(torch.LongTensor([1]).to(device=constants.device)).unsqueeze(0)
        self.reduce_r_embedding = self.action_embeddings(torch.LongTensor([2]).to(device=constants.device)).unsqueeze(0)

        """
            MLP[stack_lstm_output,buffer_lstm_output,action_lstm_output] ---> decide next transition
        """

        # neural model parameters
        self.dropout = nn.Dropout(self.dropout_prob)
        # lstms
        self.stack_lstm = nn.LSTM(self.embedding_size * 2, int(self.hidden_size/2)).to(device=constants.device)
        self.buffer_lstm = nn.LSTM(self.embedding_size * 2, int(self.hidden_size/2)).to(device=constants.device)
        self.action_lsttm = nn.LSTM(self.embedding_size, int(self.hidden_size/2)).to(device=constants.device)

        # parser state
        self.parser_state = nn.Parameter(torch.zeros((self.batch_size, self.hidden_size * 3 * 2))).to(
            device=constants.device)
        # init params
        input_init = torch.zeros((1,1,int(self.hidden_size/2))).to(
            device=constants.device)
        hidden_init = torch.zeros((1,1,int(self.hidden_size/2))).to(
            device=constants.device)

        input_init_act = torch.zeros((1, 1,int(self.hidden_size/2))).to(
            device=constants.device)
        hidden_init_act = torch.zeros((1, 1,int(self.hidden_size/2))).to(
            device=constants.device)

        self.lstm_init_state = (nn.init.xavier_normal_(input_init), nn.init.xavier_normal_(hidden_init))
        self.lstm_init_state_actions = (nn.init.xavier_normal_(input_init_act), nn.init.xavier_normal_(hidden_init_act))
        self.gaurd = torch.zeros((1,1,self.embedding_size*2)).to(device=constants.device)
        self.empty_initial = nn.Parameter(torch.randn(self.batch_size, self.hidden_size))
        # MLP it's actually a one layer network
        self.mlp_lin = nn.Linear(int(self.hidden_size/2)*3,self.num_actions)#nn.Softmax(dim=-1)(nn.ReLU()(nn.Linear(self.hidden_size * 3, self.num_actions)())())()
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

    def parse_step(self, parser, stack, buffer, action, oracle=None):
        # get parser state
        #print(stack.embedding().shape)
        #print(buffer.embedding().shape)
        #print(action.embedding().shape)
        parser_state = torch.cat([stack.embedding(), buffer.embedding(), action.embedding()],dim=-1)
        action_probabilities = self.mlp_lin(parser_state)
        action_probabilities = self.mlp_relu(action_probabilities)
        action_probabilities = self.mlp_softmax(action_probabilities).squeeze(0)#.clone()
        criterion_a = nn.CrossEntropyLoss().to(device=constants.device)
        l = None
        if oracle is not None:
            if oracle.item() == -2:
                #last action
                parser.stack.pop()
                stack.pop()
                target = torch.tensor([1])
                l = criterion_a(action_probabilities, target)
                return parser, action_probabilities, (stack, buffer, action), l
            best_action = oracle.item()
            target = oracle.reshape(1)
            #print(target.shape)
            #print(target)
            #print(action_probabilities)
            l = criterion_a(action_probabilities,target)
        else:
            if parser.stack.get_len() < 1:
                # can't left or right
                #tmp = action_probabilities
                #tmp[:,1] = -float('inf')
                #tmp[:,2] = -float('inf')
                best_action = torch.argmax(action_probabilities[:,0],dim=-1).item()
                #print("can be 0 {}".format(best_action))
            elif parser.buffer.get_len() == 1:
                # can't shift
                tmp = action_probabilities.detach().clone().to(device=constants.device)
                tmp[:, 0] = -float('inf')
                #tmp[:, 2] = -float('inf')
                best_action = torch.argmax(tmp, dim=-1).item()
                #print("can be 1,2 {}".format(best_action))

            else:
                best_action = torch.argmax(action_probabilities.clone().detach(), dim=-1).item()
                #print("can be 0,1,2 {}".format(best_action))

        # do the action
        if best_action == 0:
            # shift
            stack.push(buffer.pop())
            action.push(self.shift_embedding)
            parser.shift(self.shift_embedding)
            #print(constants.shift)
        elif best_action == 1:
            # reduce-l
            stack.pop()
            action.push(self.reduce_l_embedding)
            #tmp = parser.head_probs.clone().detach().to(device=constants.device)
            #tmp[:,parser.buffer.left()[1],parser.stack.top()[1]]= action_probabilities[:,1]
            #parser.head_probs = nn.Softmax(dim=1)(torch.mul(parser.head_probs,tmp))
            parser.reduce_l(self.reduce_l_embedding)
            #print(constants.reduce_l)

        else:
            # reduce-r
            buffer.push(stack.pop()) #not sure, should replace in buffer actually...
            action.push(self.reduce_r_embedding)
            #tmp = parser.head_probs.clone().detach().to(device=constants.device)
            #tmp[:, parser.stack.top()[1], parser.buffer.left()[1]] = action_probabilities[:, 2]
            #parser.head_probs = nn.Softmax(dim=1)(torch.mul(parser.head_probs, tmp))
            parser.reduce_r(self.reduce_r_embedding)

            #print(constants.reduce_r)

        return parser, action_probabilities, (stack, buffer, action), l

    def forward(self, x, transitions=None):
        if transitions is None:
            mode = "predict"
        else:
            mode = "train"
        stack = StackRNN(self.stack_lstm, self.lstm_init_state, self.lstm_init_state,self.dropout, self.empty_initial)
        buffer = StackRNN(self.buffer_lstm, self.lstm_init_state, self.lstm_init_state,self.dropout, self.empty_initial)
        action = StackRNN(self.action_lsttm, self.lstm_init_state_actions, self.lstm_init_state_actions,self.dropout, self.empty_initial)
        sent_lens = (x[0] != 0).sum(-1).to(device=constants.device)
        #print(sent_lens)

        x_emb = self.get_embeddings(x)
        #print(x_emb.shape)
        sent_len = x_emb.shape[1]
        max_num_actions_taken = 0
        heads_batch = torch.ones((x_emb.shape[0],sent_len),requires_grad=False).to(device=constants.device)
        heads_batch *= -1
        actions_batch = []
        stack.push(self.gaurd)
        buffer.push(self.gaurd)
        head_probs_batch = torch.zeros((x_emb.shape[0],x_emb.shape[1],x_emb.shape[1])).to(device=constants.device)

        act_loss = 0
        # parse every sentence in batch
        #print(transitions)
        for i, sentence in enumerate(x_emb):
            # initialize a parser
            curr_sentence_length = sent_lens[i]
            sentence = sentence[:curr_sentence_length,:]
            parser = ShiftReduceParser(sentence, self.embedding_size, self.transition_system)
            # initialize buffer first
            for word in sentence:
                buffer.push(word.reshape(1,1,word.shape[0]))
            # push first word to stack
            #item = buffer.pop()
            #print("item {}".format(len(item)))
            stack.push(buffer.pop())
            action.push(self.shift_embedding)
            ##print((parser.buffer.get_len(), parser.stack.get_len()))
            parser.shift(self.shift_embedding)


            #actions_probs = torch.zeros((1, self.num_actions)).to(device=constants.device)
            #actions_probs[:, 0] = 1
            #print((parser.buffer.get_len(), parser.stack.get_len()))

            # collect action_history and head probabilities
            oracle_actions_redundant = transitions[i]
            oracle_actions_ind = torch.where(oracle_actions_redundant!=-1)[0]
            oracle_actions = oracle_actions_redundant[oracle_actions_ind]
            #print(len(oracle_actions))
            oracle_actions = oracle_actions[1:]
            #print(len(oracle_actions))
            if oracle_actions[-1] != -2:
                oracle_actions = torch.cat([oracle_actions,torch.tensor([-2])],dim=0)
            #print(oracle_actions)
            try:
                for step in range(len(oracle_actions)):
                    #print((parser.stack.get_len(),parser.buffer.get_len()))
                    parser, probs, configuration,l = self.parse_step(parser,stack,buffer,action,oracle_actions[step])
                    (stack,buffer,action) = configuration
                    act_loss += l
            except:
                act_loss += 0.1
            #print((parser.stack.get_len(), parser.buffer.get_len()))
            #actions_probs = torch.cat([actions_probs,probs],dim=0)
            #while not parser.is_parse_complete():
            #    parser, probs, configuration = self.parse_step(parser, stack, buffer, action)
            #    (stack, buffer, action) = configuration
            #    actions_probs = torch.cat([actions_probs, probs], dim=0)
                #print((parser.buffer.get_len(),parser.stack.get_len()))

            #print(parser.head_list.shape)
            #head_probs_batch[i, :, :] = parser.head_probs
            #heads_batch[i,:] = parser.head_list
            #actions_batch.append(actions_probs)
            #max_num_actions_taken = max(actions_probs.shape[0], max_num_actions_taken)

        #max_num_actions_taken = max(max_num_actions_taken, transitions.shape[1])
        #actions_taken = torch.zeros((x_emb.shape[0], max_num_actions_taken, self.num_actions)).to(
        #    device=constants.device)
        #actions_oracle = torch.ones((x_emb.shape[0], max_num_actions_taken), dtype=torch.long).to(
        #    device=constants.device)#*-1
        #for i in range(x_emb.shape[0]):
        #    actions_taken[i, :actions_batch[i].shape[0], :] = actions_batch[i].unsqueeze(0).clone()
        #    actions_oracle[i, :transitions.shape[1]] = transitions[i, :]

        #sent_lens = (x[0] != 0).sum(-1).to(device=constants.device)
        #h_logits = self.get_head_logits(head_probs_batch, sent_lens)

        return act_loss#, h_logits#actions_taken, actions_oracle, h_logits


    def get_head_logits(self, h_logits, sent_lens):
        #h_dep = self.dropout(F.relu(self.linear_arc_dep(h_t)))
        #h_arc = self.dropout(F.relu(self.linear_arc_head(h_t)))#

        #h_logits = self.biaffine(h_arc, h_dep)

        # Zero logits for items after sentence length
        for i, sent_len in enumerate(sent_lens):
            h_logits[i, sent_len:, :] = 0
            h_logits[i, :, sent_len:] = 0

        return h_logits

    @staticmethod

    def loss(parser_actions, oracle_actions, h_logits, heads):
        #criterion_a = nn.CrossEntropyLoss().to(device=constants.device)
        criterion_h = nn.CrossEntropyLoss(ignore_index=-1).to(device=constants.device)
        loss = criterion_h(h_logits.reshape(-1, h_logits.shape[-1]), heads.reshape(-1))

        #print("----------------------")
        #print(parser_actions.shape)
        #print(oracle_actions.shape)
        #batch_size = parser_actions.shape[0]
        #loss = 0
        #true_oracle_lengths = (oracle_actions == -1).nonzero(as_tuple=True)
        #tolx = true_oracle_lengths[0]
        #toly = true_oracle_lengths[1]

        #min_inds = torch.zeros((batch_size))
        ## dumb way of doing this
        #for i in range(batch_size):
        #    for ij, j in enumerate(tolx):
        #        if j == i:
        #            min_inds[i] = toly[ij]
        #            break

        #print("parser actions {}".format(parser_actions))
        #print("oracle actions {}".format(oracle_actions))
        ##print(true_oracle_lengths)
        #print(min_inds)
        #max_action_length = torch.ones((batch_size))*parser_actions.shape[1]
        #extra_actions = max_action_length - (min_inds+1)
        #print(extra_actions)
        #print(extra_actions.sum(dim=-1)/(parser_actions.shape[1]*batch_size))
        #loss += extra_actions.sum(dim=-1)/(parser_actions.shape[1]*batch_size)
        #indicies = [int(i) for i in min_inds.clone().detach()]

        #for i in range(batch_size):
        #    #print(parser_actions[i,:indicies[i],:])
        #    #print(oracle_actions[i,:indicies[i]])
        #    #print(criterion_a(parser_actions[i,:indicies[i],:],oracle_actions[i,:indicies[i]]))
        #    #loss += criterion_a(parser_actions[i,:indicies[i],:],oracle_actions[i,:indicies[i]])
        #    loss += criterion_a(parser_actions[i],oracle_actions[i])
        #parser_actions = parser_actions.reshape(-1, parser_actions.shape[-1])
        #oracle_actions = oracle_actions.reshape(-1)
        #print(parser_actions.shape)
        #print(oracle_actions.shape)
        #print("----------------------")
        #loss = criterion_a(parser_actions, oracle_actions)
        return loss
