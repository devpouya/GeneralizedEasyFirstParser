import copy
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from utils import constants
from utils import utils
from transformers import BertModel, AutoModel
from transformers import RobertaTokenizer, RobertaModel



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
    def __init__(self, language, vocabs, embedding_size, rel_embedding_size, batch_size,
                 dropout=0.33, transition_system=None):
        super().__init__()
        self.language=language
        # basic parameters
        self.vocabs = vocabs
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.dropout_prob = dropout
        if language == "en":
            self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).to(device=constants.device).train()
        elif language == "de":
            self.bert = BertModel.from_pretrained('bert-base-german-cased', output_hidden_states=True).to(device=constants.device).train()
        elif language == "cs":
            self.bert = AutoModel.from_pretrained("DeepPavlov/bert-base-bg-cs-pl-ru-cased", output_hidden_states=True).to(device=constants.device).train()
        elif language == "eu":
            self.bert = AutoModel.from_pretrained("ixa-ehu/berteus-base-cased", output_hidden_states=True).to(device=constants.device).train()
        elif language == "tr":
            self.bert = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased",output_hidden_states=True).to(device=constants.device).train()
        #self.bert = RobertaModel.from_pretrained("roberta-base").to(device=constants.device)#.train()
        self.bert.eval()
        for param in self.bert.parameters():
            param.requires_grad = True

        # transition system
        self.transition_system = transition_system
        # print(self.transition_system)
        if transition_system is not None:
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
            print("Using a chart parser")
        _, _, rels = vocabs

        self.num_rels = rels.size
        self.rel_embedding_size = rel_embedding_size
        if transition_system is not None:
            self.num_total_actions = non_labeling_actions + (self.num_actions - non_labeling_actions) * self.num_rels
            self.action_embeddings_size = self.embedding_size
            self.tag_embeddings, self.action_embeddings, self.rel_embeddings = self.create_embeddings(vocabs)
        else:
            self.tag_embeddings, self.rel_embeddings = self.create_embeddings(vocabs,no_action=True)

    def create_embeddings(self, vocabs,no_action=False):
        words, tags, rels = vocabs
        # word_embeddings = WordEmbedding(words, self.embedding_size, pretrained=pretrained)
        tag_embeddings = nn.Embedding(tags.size, self.rel_embedding_size)
        rel_embeddings = nn.Embedding(self.num_rels, self.rel_embedding_size, scale_grad_by_freq=True)

        # learned_embeddings = nn.Embedding(words.size, self.rel_embedding_size)
        if not no_action:
            action_embedding = nn.Embedding(self.num_actions, self.action_embeddings_size, scale_grad_by_freq=True)
            return tag_embeddings, action_embedding, rel_embeddings
        else:
            return tag_embeddings, rel_embeddings

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
        #return torch.cat([s, tags], dim=-1).to(device=constants.device)
        return s


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

