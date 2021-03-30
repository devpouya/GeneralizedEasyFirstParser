import copy
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from utils import constants
from utils import utils
from transformers import BertModel



class BaseParser(nn.Module, ABC):
    # pylint: disable=abstract-method
    name = 'base'

    def __init__(self):
        super().__init__()

        self.best_state_dict = None#self.load_state_dict(self.state_dict())#.state_dict()

    def set_best(self):
        with torch.no_grad():
            #state_dict = {k: v.detach().cpu() for k, v in self.state_dict().items()}
            #self.best_state_dict = copy.deepcopy(state_dict)
            self.best_state_dict = copy.deepcopy(self.state_dict())

    def recover_best(self):
        with torch.no_grad():
            #state_dict = {k: v.to(device=constants.device).detach()
            #               for k, v in self.best_state_dict.items()}
            #self.load_state_dict(state_dict)
            self.load_state_dict(self.best_state_dict) if self.best_state_dict is not None else self.load_state_dict(self.state_dict())
        # torch.cuda.empty_cache()

    def save(self, path):
        utils.mkdir(path)
        fname = self.get_name(path)
        torch.save({
            'kwargs': self.get_args(),
            'model_state_dict': self.state_dict(),
        }, fname)

    @abstractmethod
    def get_args(self):
        pass

    @classmethod
    def load(cls, path):
        checkpoints = cls.load_checkpoint(path)
        model = cls(**checkpoints['kwargs'])
        model.load_state_dict(checkpoints['model_state_dict'])
        del checkpoints
        return model

    @classmethod
    def load_checkpoint(cls, path):
        fname = cls.get_name(path)
        return torch.load(fname, map_location=constants.device)

    @classmethod
    def get_name(cls, path):
        return '%s/model.tch' % (path)


class BertParser(BaseParser):
    def __init__(self, vocabs, embedding_size, rel_embedding_size, batch_size,
                 dropout=0.33, beam_size=10, transition_system=None):
        super().__init__()
        # basic parameters
        self.vocabs = vocabs
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.dropout_prob = dropout
        self.bert = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True).to(device=constants.device)
        self.bert.eval()
        self.beam_size = beam_size
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
        elif self.transition_system == constants.easy_first:
            self.parse_step = self.parse_step_easy_first
            self.arc_actions = [0,1]
            self.non_arc_actions = []
        else:
            raise Exception("A transition system needs to be satisfied")
        _, _, rels = vocabs

        self.num_rels = rels.size
        self.num_total_actions = non_labeling_actions + (self.num_actions - non_labeling_actions) * self.num_rels
        self.action_embeddings_size = self.embedding_size
        self.rel_embedding_size = rel_embedding_size

        self.tag_embeddings, self.action_embeddings, self.rel_embeddings = self.create_embeddings(vocabs)

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

    def loss(self, probs, targets, probs_rel, targets_rel):
        criterion1 = nn.CrossEntropyLoss().to(device=constants.device)

        criterion2 = nn.CrossEntropyLoss().to(device=constants.device)

        num_batches = probs.shape[0]
        l1, l2 = 0, 0
        for i in range(num_batches):
            p = probs[i]
            t = targets[i]
            t = t[p[:, 0] != -1]
            p = p[p[:, 0] != -1, :]
            l1 += criterion1(p, t.squeeze(1))
            pr = probs_rel[i]
            tr = targets_rel[i].squeeze(1)
            pr = pr[tr != 0, :]
            tr = tr[tr != 0]
            tr = tr[pr[:, 0] != -1]
            pr = pr[pr[:, 0] != -1]
            l2 += criterion2(pr, tr)
        l1 /= num_batches
        l2 /= num_batches

        #probs = probs.reshape(-1, probs.shape[-1])
        #targets = targets.reshape(-1)
        #targets = targets[probs[:, 0] != -1]
        #probs = probs[probs[:, 0] != -1, :]
        #probs_rel = probs_rel.reshape(-1, probs_rel.shape[-1])
        #targets_rel = targets_rel.reshape(-1)
        #probs_rel = probs_rel[targets_rel != 0, :]
        #targets_rel = targets_rel[targets_rel != 0]
        #targets_rel = targets_rel[probs_rel[:, 0] != -1]
        #probs_rel = probs_rel[probs_rel[:, 0] != -1, :]

        #loss = criterion1(probs, targets)
        #loss +=criterion2(probs_rel, targets_rel)
        return l1 + l2

    def get_args(self):
        return {
            'vocabs': self.vocabs,
            'embedding_size': self.embedding_size,
            'rel_embedding_size': self.rel_embedding_size,
            'dropout': self.dropout_prob,
            'batch_size': self.batch_size,
            'transition_system': self.transition_system
        }

