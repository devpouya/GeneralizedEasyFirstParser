import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import random
import numpy as np

from utils import constants
from .base import BaseParser
from .modules import Biaffine, Bilinear, StackLSTM
from .word_embedding import WordEmbedding, ActionEmbedding
from ..algorithm.transition_parsers import ShiftReduceParser
from .modules import StackRNN, StackCell, SoftmaxLegal, SoftmaxActions
from transformers import BertModel, AutoModel



class NeuralTransitionParser(BaseParser):
    def __init__(self, language, vocabs, embedding_size, rel_embedding_size, batch_size,
                 dropout=0.33,transition_system=None):
        super().__init__()
        # basic parameters
        self.vocabs = vocabs
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.dropout_prob = dropout
        #self.bert = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True).to(device=constants.device)
        if language == "en":
            self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).to(
                device=constants.device)
        elif language == "de":
            self.bert = BertModel.from_pretrained('bert-base-german-cased', output_hidden_states=True).to(
                device=constants.device)
        elif language == "cs":
            self.bert = AutoModel.from_pretrained("DeepPavlov/bert-base-bg-cs-pl-ru-cased",
                                                  output_hidden_states=True).to(device=constants.device)
        elif language == "eu":
            self.bert = AutoModel.from_pretrained("ixa-ehu/berteus-base-cased", output_hidden_states=True).to(
                device=constants.device)
        elif language == "hu":
            self.bert = AutoModel.from_pretrained("SZTAKI-HLT/hubert-base-cc", output_hidden_states=True).to(
                device=constants.device)
        elif language == "tr":
            self.bert = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased",output_hidden_states=True).to(device=constants.device)
        else:
            self.bert = AutoModel.from_pretrained("nlpaueb/bert-base-greek-uncased-v1",output_hidden_states=True).to(device=constants.device)



        self.bert.eval()
        for param in self.bert.parameters():
            param.requires_grad = True

        # transition system
        self.transition_system = transition_system
        # print(self.transition_system)©
        self.actions = transition_system[0]  # [shift, reduce_l, reduce_r]
        self.num_actions = len(self.actions)
        non_labeling_actions = 0
        for act in self.actions:
            if act == constants.shift or act == constants.reduce:
                non_labeling_actions += 1

        self.action2id = {act: i for i, act in enumerate(self.actions)}
        if self.transition_system == constants.arc_standard:
            self.parse_step = self.parse_step_arc_standard
            self.arc_actions = [1, 2]
            self.non_arc_actions = [0]
        elif self.transition_system == constants.arc_eager:
            self.parse_step = self.parse_step_arc_eager
            self.arc_actions = [1, 2]
            self.non_arc_actions = [0, 3]
        elif self.transition_system == constants.hybrid:
            self.parse_step = self.parse_step_hybrid
            self.arc_actions = [1, 2]
            self.non_arc_actions = [0]
        elif self.transition_system == constants.mh4:
            self.parse_step = self.parse_step_mh4
            self.arc_actions = [1, 2, 3, 4, 5, 6]
            self.non_arc_actions = [0]
        else:
            raise Exception("A transition system needs to be satisfied")
        _, _, rels = vocabs

        self.num_rels = rels.size
        self.num_total_actions = non_labeling_actions + (self.num_actions - non_labeling_actions) * self.num_rels
        self.action_embeddings_size = self.embedding_size
        self.rel_embedding_size = rel_embedding_size

        self.tag_embeddings, self.action_embeddings, self.rel_embeddings = self.create_embeddings(vocabs)

        """
            MLP[stack_lstm_output,buffer_lstm_output,action_lstm_output] ---> decide next transition
        """

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

    def create_embeddings(self, vocabs):
        words, tags, rels = vocabs
        # word_embeddings = WordEmbedding(words, self.embedding_size, pretrained=pretrained)
        tag_embeddings = nn.Embedding(tags.size, self.rel_embedding_size)
        rel_embeddings = nn.Embedding(self.num_rels, self.rel_embedding_size, scale_grad_by_freq=True)

        # learned_embeddings = nn.Embedding(words.size, self.rel_embedding_size)
        action_embedding = nn.Embedding(self.num_actions, self.action_embeddings_size, scale_grad_by_freq=True)
        return tag_embeddings, action_embedding, rel_embeddings

    def pairwise(self, iterable):
        a = iter(iterable)
        return zip(a, a)

    def get_bert_embeddings(self, mapping, sentence, tags):
        s = []  # torch.zeros((mapping.shape[0]+1, sentence.shape[1])).to(device=constants.device)
        for start, end in self.pairwise(mapping):
            m = torch.mean(sentence[start:end + 1, :], dim=0)
            s.append(m)
        s = torch.stack(s, dim=0).to(device=constants.device)

        # self.tag_embeddings()
        return torch.cat([s, tags], dim=-1).to(device=constants.device)

    def labeled_action_pairs(self, actions, relations):
        labeled_acts = []
        tmp_rels = relations.clone().detach().tolist()
        for act in actions:
            if act in self.arc_actions:
                labeled_acts.append((act, tmp_rels[0]))
                tmp_rels.pop(0)
            elif act in self.non_arc_actions:
                labeled_acts.append((act, 0))

        return labeled_acts

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
            action_probabilities = SoftmaxActions(dim=-1, parser=parser, transition_system=self.transition_system,temperature=2)(state2)
        else:
            action_probabilities = nn.Softmax(dim=-1)(state2).squeeze(0)
        state2 = self.dropout(F.relu(self.mlp_rel(state1)))
        rel_probabilities = nn.Softmax(dim=-1)(state2).squeeze(0)

        if not self.training:# == 'eval':
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

        #if mode == 'eval':
        #    probs = torch.distributions.Categorical(action_probabilities)
        #    #inds = parser.legal_indices_mh4()
        #    #probs = torch.distributions.Uniform()
        #    #action_samples = []
        #    #for _ in range(self.beam_size):
        #    #    action_samples.append(probs.sample().item())
        #    best_action = probs.sample().item()#random.choice(action_samples)

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
            #self.stack.pop()
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

    def forward(self, x, transitions, relations, map):

        # sent_lens = (x[0] != 0).sum(-1).to(device=constants.device)
        transit_lens = (transitions != -1).sum(-1).to(device=constants.device)
        # print(x[1])
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

        return batch_loss, heads_batch, rels_batch

    def loss(self, probs, targets, probs_rel, targets_rel):
        criterion1 = nn.CrossEntropyLoss().to(device=constants.device)

        criterion2 = nn.CrossEntropyLoss().to(device=constants.device)

        #num_batches = probs.shape[0]
        #l1, l2 = 0, 0
        #for i in range(num_batches):
        #    p = probs[i]
        #    t = targets[i]
        #    t = t[p[:, 0] != -1]
        #    p = p[p[:, 0] != -1, :]
        #    l1 += criterion1(p, t.squeeze(1))
        #    pr = probs_rel[i]
        #    tr = targets_rel[i].squeeze(1)
        #    pr = pr[tr != 0, :]
        #    tr = tr[tr != 0]
        #    tr = tr[pr[:, 0] != -1]
        #    pr = pr[pr[:, 0] != -1]
        #    l2 += criterion2(pr, tr)
        #l1 /= num_batches
        #l2 /= num_batches

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

        loss = criterion1(probs, targets)
        loss +=criterion2(probs_rel, targets_rel)
        return loss#l1 + l2

    def get_args(self):
        return {
            'vocabs': self.vocabs,
            'embedding_size': self.embedding_size,
            'rel_embedding_size': self.rel_embedding_size,
            'dropout': self.dropout_prob,
            'batch_size': self.batch_size,
            'transition_system': self.transition_system
        }
