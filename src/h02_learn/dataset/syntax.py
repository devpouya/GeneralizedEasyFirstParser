import json
import torch
from torch.utils.data import Dataset

from utils import constants


class SyntaxDataset(Dataset):
    def __init__(self, fname):
        self.fname = fname
        self.load_data(fname)
        self.n_instances = len(self.words)

    def load_data(self, fname):
        self.words, self.pos, self.heads, self.rels = [], [], [], []
        with open(fname, 'r') as file:
            for line in file.readlines():
                sentence = json.loads(line)
                self.words += [self.list2tensor([word['word_id'] for word in sentence])]
                self.pos += [self.list2tensor([word['tag1_id'] for word in sentence])]
                self.heads += [self.list2tensor([word['head'] for word in sentence])]
                self.rels += [self.list2tensor([word['rel_id'] for word in sentence])]

    @staticmethod
    def list2tensor(data):
        return torch.LongTensor(data).to(device=constants.device)


    @staticmethod
    def get_n_instances(fname):
        with open(fname, 'r') as file:
            count = 0
            for _ in file.readlines():
                count += 1
        return count

    def __len__(self):
        return self.n_instances

    def __getitem__(self, index):
        return (self.words[index], self.pos[index]), \
            (self.heads[index], self.rels[index])
