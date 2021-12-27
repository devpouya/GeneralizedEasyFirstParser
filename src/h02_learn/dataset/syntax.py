import json
import torch
from torch.utils.data import Dataset
import numpy as np
from utils import constants
import random




class SyntaxDataset(Dataset):
    def __init__(self, fname_all, transition_file=None, transition_system=None, tokenizer=None, rel_size = 20):
        # self.fname = fname
        self.max_rel = rel_size
        self.model = transition_system
        self.transition_file = transition_file
        self.transition_system = {act: i for (act, i) in zip(transition_system[0], transition_system[1])}
        # self.transition_system[None] = -2
        self.act_counts = {self.transition_system[act]: 0 for act in self.transition_system.keys()}

        self.tokenizer = tokenizer
        self.max_sent_len = 0
        self.words, self.word_st, self.pos, self.heads, self.rels = [], [], [], [], []
        self.actions = []
        self.mappings = []
        self.relations_in_order = []
        self.language_starts = [0]
        if isinstance(fname_all,list):
            for fname, tf in zip(fname_all, transition_file):
                self.load_data(fname, tf)
        else:
            self.load_data(fname_all, transition_file)
        #self.n_instances = len(self.words)


    # save agenda actions in the same format as the rest
    def load_data(self, fname, transition_file):

        # self.labeled_actions = []
        index = 0

        with open(fname, 'r') as file, open(transition_file, 'r') as file2:
            for line, action in zip(file.readlines(), file2.readlines()):
                sentence = json.loads(line)
                #print(sentence)
                tranisiton = json.loads(action)
                self.words += [self.list2tensor([word['word_id'] for word in sentence])]
                tokens, mapping = self.tokenize([word['word'] for word in sentence])
                self.word_st += [tokens]
                self.mappings += [mapping]
                length = len([word['word_id'] for word in sentence])
                if length > self.max_sent_len:
                    self.max_sent_len = length
                # self.check_lens([word['word_id'] for word in sentence],tranisiton['transition'])
                self.pos += [self.list2tensor([word['tag1_id'] for word in sentence])]
                self.heads += [self.list2tensor([word['head'] for word in sentence])]
                self.rels += [self.list2tensor([word['rel_id'] for word in sentence])]
                self.actions += [self.actionsequence2tensor(tranisiton['transition'])]
                self.relations_in_order += [self.list2tensor(tranisiton['relations'])]
                # self.labeled_actions += [self.labeled_act2tensor(tranisiton['labeled_actions'])]
                index += 1
        new_start = self.language_starts[-1] + index
        self.language_starts.append(new_start)
        self.n_instances = index

    def check_lens(self, words, actions):
        n = len(words)
        ids = [self.transition_system[act] for act in actions]

        # should be 2n-1, there's one extra "null" action for implementation purposes
        print(len(ids) == 2 * n - 1)

        return len(ids) == 2 * n - 1

    def tokenize(self, wordlist):
        wordlist = wordlist  # + ["<EOS>"]
        #print("wordlen {}".format(len(wordlist)))
        encoded = self.tokenizer(wordlist, is_split_into_words=True, return_tensors="pt",
                                 return_attention_mask=False,
                                 return_token_type_ids=False,
                                 add_special_tokens=True)
        #print("encoded {}".format(len(encoded)))

        # print(encoded)
        # kj
        enc = [self.tokenizer.encode(x, add_special_tokens=False) for x in wordlist]
        idx = 0
        token_mapping = []
        token_mapping2 = []
        for token in enc:
            tokenout = []
            for ids in token:
                tokenout.append(idx)
                idx += 1

            token_mapping.append(tokenout[0])
            token_mapping.append(tokenout[-1])
            token_mapping2.append(tokenout)

        # print(self.tokenizer.eos_token_id)
        # jk
        return encoded['input_ids'].squeeze(0), torch.LongTensor(token_mapping).to(device=constants.device)

    def hypergraph2tensor(self, hypergraph):
        all_graphs = []
        for left in hypergraph:
            i1, j1 = left
            tmp = torch.LongTensor([i1, j1]).to(device=constants.device)
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
        return (self.word_st[index], self.pos[index]), (self.heads[index], self.rels[index]), \
               (self.actions[index], self.relations_in_order[index]), (self.mappings[index], self.words[index])


class LanguageBatchSampler(torch.utils.data.Sampler):

    def __init__(self, language_start_indicies, batch_size, shuffle):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.language_lengths = []


        for i in range(1, len(language_start_indicies)):
            len_lang = language_start_indicies[i] - language_start_indicies[i - 1]
            self.language_lengths.append(len_lang)
        langlengths = np.array(self.language_lengths)
        langlengths = np.power(langlengths, 0.3)
        self.lang_prob = langlengths / np.sum(langlengths)


        lang_indicies = {}
        lang_indicies[0] = list(range(self.language_lengths[0]))
        for i in range(1, len(self.language_lengths)):
            lang_indicies[i] = list(range(self.language_lengths[i - 1], self.language_lengths[i]))
        self.lang_indicies = lang_indicies

    def __iter__(self):
        bins = self.lang_indicies#deepcopy(self.lang_indicies)
        if self.shuffle:
            for key in bins:
                random.shuffle(bins[key])

        # make batched
        batched_lang = {}
        for key in bins:
            l = torch.split(torch.tensor(bins[key]), self.batch_size)#np.array_split(bins[key],self.batch_size)
            batched_lang[key] = list(l)

        self.final_indices = []
        finished = {i:False for i in range(len(bins.keys()))}
        while not all(value is True for value in finished.values()):
            language_sample = np.random.multinomial(1, self.lang_prob, size=1).reshape(len(self.lang_prob))
            ind = np.argwhere(language_sample != 0)[0][0]
            if not finished[ind]:
                language_ind = batched_lang[ind].pop()
                self.final_indices.append(language_ind)
                if len(batched_lang[ind]) == 0:
                    finished[ind] = True
        if self.shuffle:
            random.shuffle(self.final_indices)
        #print(final_indices)
        return iter(self.final_indices)

    def __len__(self):
        return max(self.lang_indicies, key=self.lang_indicies.get)#//self.batch_size