import json
import torch
from torch.utils.data import Dataset
import re
from utils import constants


class SyntaxDataset(Dataset):
    def __init__(self, fname, transition_file, transition_system, tokenizer):
        self.fname = fname
        self.max_rel = 0

        self.transition_file = transition_file
        self.transition_system = {act: i for (act, i) in zip(transition_system[0], transition_system[1])}
        # self.transition_system[None] = -2
        self.tokenizer = tokenizer
        self.act_counts = {self.transition_system[act]:0 for act in self.transition_system.keys()}

        self.load_data(fname, transition_file)
        self.n_instances = len(self.words)

    def load_data(self, fname, transition_file):
        self.words, self.word_st, self.pos, self.heads, self.rels = [], [], [], [], []
        self.actions = []
        self.mappings = []
        self.relations_in_order = []
        # self.labeled_actions = []
        with open(fname, 'r') as file, open(transition_file, 'r') as file2:
            for line, action in zip(file.readlines(), file2.readlines()):
                sentence = json.loads(line)
                tranisiton = json.loads(action)
                self.words += [self.list2tensor([word['word_id'] for word in sentence])]
                tokens, mapping = self.tokenize([word['word'] for word in sentence])
                self.word_st += [tokens]
                self.mappings += [mapping]
                # self.check_lens([word['word_id'] for word in sentence],tranisiton['transition'])
                self.pos += [self.list2tensor([word['tag1_id'] for word in sentence])]
                self.heads += [self.list2tensor([word['head'] for word in sentence])]
                self.rels += [self.list2tensor([word['rel_id'] for word in sentence])]
                self.actions += [self.actionsequence2tensor(tranisiton['transition'])]
                self.relations_in_order += [self.list2tensor(tranisiton['relations'])]
                # self.labeled_actions += [self.labeled_act2tensor(tranisiton['labeled_actions'])]

    def check_lens(self, words, actions):
        n = len(words)
        ids = [self.transition_system[act] for act in actions]

        # should be 2n-1, there's one extra "null" action for implementation purposes
        print(len(ids) == 2 * n - 1)

        return len(ids) == 2 * n - 1

    def tokenize(self, wordlist):
        encoded = self.tokenizer(wordlist, is_split_into_words=True, return_tensors="pt",
                                 return_attention_mask=False,
                                 return_token_type_ids=False,
                                 add_special_tokens=False)

        enc = [self.tokenizer.encode(x, add_special_tokens=False) for x in wordlist]

        idx = 0
        token_mapping = []
        token_mapping2 = []
        for token in enc:
            tokenout = []
            if len(token)>=1:
                for ids in token:
                    tokenout.append(idx)
                    idx+=1
                token_mapping.append(tokenout[0])
                token_mapping.append(tokenout[-1])
                token_mapping2.append(tokenout)



        return encoded['input_ids'].squeeze(0), torch.LongTensor(token_mapping).to(device=constants.device)

    def actionsequence2tensor(self, actions):
        ids = [self.transition_system[act] for act in actions]
        for id in ids:
            self.act_counts[id] += 1
        return torch.LongTensor(ids).to(device=constants.device)

    def labeled_act2tensor(self, labeled_actions):
        ret = []
        for (act, lab) in labeled_actions:
            act_id = torch.LongTensor(self.transition_system[act]).to(device=constants.device)
            lab_tens = torch.LongTensor(lab).to(device=constants.device)
            ret.append((act_id, lab_tens))
        return ret

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
        # print(self.actions[index])
        # return (self.words[index], self.pos[index]), \
        #       (self.heads[index], self.rels[index]), self.actions[index]
        return (self.word_st[index], self.pos[index]), (self.heads[index], self.rels[index]), \
               (self.actions[index], self.relations_in_order[index]), (self.mappings[index],self.words[index])
