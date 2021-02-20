import json
import torch
from torch.utils.data import Dataset

from utils import constants


class SyntaxDataset(Dataset):
    def __init__(self, fname, transition_file, transition_system):
        self.fname = fname
        self.transition_file = transition_file
        self.transition_system = {act: i for (act, i) in zip(transition_system[0], transition_system[1])}
        self.load_data(fname, transition_file)
        self.n_instances = len(self.words)

    def load_data(self, fname, transition_file):
        self.words, self.pos, self.heads, self.rels = [], [], [], []
        self.actions = []
        with open(fname, 'r') as file, open(transition_file, 'r') as file2:
            for line, action in zip(file.readlines(), file2.readlines()):
                sentence = json.loads(line)
                tranisiton = json.loads(action)
                self.words += [self.list2tensor([word['word_id'] for word in sentence])]
                self.pos += [self.list2tensor([word['tag1_id'] for word in sentence])]
                self.heads += [self.list2tensor([word['head'] for word in sentence])]
                self.rels += [self.list2tensor([word['rel_id'] for word in sentence])]
                self.actions += [self.actionsequence2tensor(tranisiton['transition'])]

    def actionsequence2tensor(self, actions):
        ids = [self.transition_system[act] for act in actions]
        return torch.LongTensor(ids).to(device=constants.device)

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
        #print(self.actions[index])
        #return (self.words[index], self.pos[index]), \
        #       (self.heads[index], self.rels[index]), self.actions[index]
        return (self.words[index], self.pos[index]), (self.heads[index], self.rels[index]), (self.actions[index],)
