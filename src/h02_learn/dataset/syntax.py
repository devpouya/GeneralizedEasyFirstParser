import json
import torch
from torch.utils.data import Dataset
import re
from utils import constants

class SyntaxDataset(Dataset):
    def __init__(self, fname, transition_file=None, transition_system=None, tokenizer=None):
        self.fname = fname
        self.max_rel = 0
        self.model = transition_system
        self.transition_file = transition_file
        self.transition_system = {act: i for (act, i) in zip(transition_system[0], transition_system[1])}
        # self.transition_system[None] = -2
        self.act_counts = {self.transition_system[act]:0 for act in self.transition_system.keys()}

        self.tokenizer = tokenizer
        self.max_sent_len = 0
        self.load_data(fname, transition_file)
        self.n_instances = len(self.words)


    # save agenda actions in the same format as the rest
    def load_data(self, fname, transition_file):
        self.word_st, self.heads, self.rels = [], [], []
        self.actions = []
        self.mappings = []
        self.words = []
        # self.labeled_actions = []
        with open(fname, 'r') as file, open(transition_file, 'r') as file2:
            for line, action in zip(file.readlines(), file2.readlines()):
                sentence = json.loads(line)
                tranisiton = json.loads(action)
                tokens, mapping = self.tokenize([word['word'] for word in sentence])
                self.word_st += [tokens]
                self.mappings += [mapping]
                length = len([word['word_id'] for word in sentence])
                if length > self.max_sent_len:
                    self.max_sent_len = length
                self.heads += [self.list2tensor([word['head'] for word in sentence])]
                self.rels += [self.list2tensor([word['rel_id'] for word in sentence])]
                self.actions += [self.actionsequence2tensor(tranisiton['transition'])]
                self.words += [self.list2tensor([word['word_id'] for word in sentence])]


    def check_lens(self, words, actions):
        n = len(words)
        ids = [self.transition_system[act] for act in actions]

        # should be 2n-1, there's one extra "null" action for implementation purposes
        print(len(ids) == 2 * n - 1)

        return len(ids) == 2 * n - 1

    def tokenize(self, wordlist):
        wordlist = wordlist #+ ["<EOS>"]
        encoded = self.tokenizer(wordlist, is_split_into_words=True, return_tensors="pt",
                                 return_attention_mask=False,
                                 return_token_type_ids=False,
                                 add_special_tokens=True)
        #print(encoded)
        #kj
        enc = [self.tokenizer.encode(x, add_special_tokens=False) for x in wordlist]
        idx = 0
        token_mapping = []
        token_mapping2 = []
        for token in enc:
            tokenout = []
            for ids in token:
                tokenout.append(idx)
                idx+=1

            token_mapping.append(tokenout[0])
            token_mapping.append(tokenout[-1])
            token_mapping2.append(tokenout)

        #print(self.tokenizer.eos_token_id)
        #jk
        return encoded['input_ids'].squeeze(0), torch.LongTensor(token_mapping).to(device=constants.device)
    def hypergraph2tensor(self, hypergraph):
        all_graphs = []
        for left in hypergraph:
            i1,j1 = left
            tmp = torch.LongTensor([i1,j1]).to(device=constants.device)
            all_graphs.append(tmp)

        return torch.stack(all_graphs).to(device=constants.device)

    def actionsequence2tensor(self, actions):
        acts = []
        if self.model == constants.agenda:
            return self.hypergraph2tensor(actions)
        elif self.model == constants.easy_first:
            for item in actions:
                edge = item[1]
                head = edge[0]
                mod = edge[1]
                acts.append(head)
                acts.append(mod)
            return torch.LongTensor(acts).to(device=constants.device)
        else:
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
        return (self.word_st[index], self.mappings[index]), (self.heads[index], self.rels[index]), self.actions[index]
