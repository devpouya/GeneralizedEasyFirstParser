import copy
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from utils import constants
from utils import utils
from transformers import BertModel, AutoModel
from transformers import AutoTokenizer, AutoModelForMaskedLM



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
    def __init__(self, l, num_rels, batch_size,
                 dropout=0.33):
        super().__init__()
        # basic parameters
        self.batch_size = batch_size
        self.dropout_prob = dropout
        #if language == "en":
        #    self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).to(device=constants.device).train()
        #elif language == "de":
        #    self.bert = BertModel.from_pretrained('bert-base-german-cased', output_hidden_states=True).to(device=constants.device).train()
        #elif language == "cs":
        #    self.bert = AutoModel.from_pretrained("DeepPavlov/bert-base-bg-cs-pl-ru-cased", output_hidden_states=True).to(device=constants.device).train()
        #elif language == "eu":
        #    self.bert = AutoModel.from_pretrained("ixa-ehu/berteus-base-cased", output_hidden_states=True).to(device=constants.device).train()
        #elif language == "tr":
        #    self.bert = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased",output_hidden_states=True).to(device=constants.device).train()
        #self.bert = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased", output_hidden_states=True).to(device=constants.device)
        self.bert = AutoModel.from_pretrained("ixa-ehu/berteus-base-cased", output_hidden_states=True).to(
            device=constants.device).train()

        if l == "eu":
            self.bert = AutoModel.from_pretrained("ixa-ehu/berteus-base-cased", output_hidden_states=True).to(
                device=constants.device).train()
        elif l == "ko":
            self.bert = AutoModel.from_pretrained("kykim/bert-kor-base", output_hidden_states=True).to(
                device=constants.device).train()
        elif l == "hu":
            self.bert = AutoModel.from_pretrained("SZTAKI-HLT/hubert-base-cc", output_hidden_states=True).to(
                device=constants.device).train()
        elif l == "af":
            #self.bert = AutoModel.from_pretrained("jannesg/takalane_afr_roberta", output_hidden_states=True).to(
            #    device=constants.device).train()
            self.bert = AutoModel.from_pretrained("Geotrend/bert-base-nl-cased", output_hidden_states=True).to(
                device=constants.device).train()
        elif l == "la":
            self.bert = AutoModel.from_pretrained("cook/cicero-similis", output_hidden_states=True).to(
                device=constants.device).train()
        elif l == "ur":
            self.bert = AutoModel.from_pretrained("Geotrend/bert-base-ur-cased", output_hidden_states=True).to(
                device=constants.device).train()
        elif l == "da":
            self.bert = AutoModel.from_pretrained("Maltehb/danish-bert-botxo", output_hidden_states=True).to(
                device=constants.device).train()
        elif l == "ga":
            self.bert = AutoModel.from_pretrained("DCU-NLP/bert-base-irish-cased-v1", output_hidden_states=True).to(
                device=constants.device).train()
        elif l == "lt":
            self.bert = AutoModel.from_pretrained("Geotrend/bert-base-lt-cased", output_hidden_states=True).to(
                device=constants.device).train()
        else:
            self.bert = AutoModel.from_pretrained("Geotrend/bert-base-nl-cased", output_hidden_states=True).to(
                device=constants.device).train()

        #self.bert = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base", output_hidden_states=True).to(device=constants.device)
        self.bert.eval()
        for param in self.bert.parameters():
            param.requires_grad = True


        self.num_rels = num_rels


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
            'dropout': self.dropout_prob,
            'batch_size': self.batch_size,
        }

