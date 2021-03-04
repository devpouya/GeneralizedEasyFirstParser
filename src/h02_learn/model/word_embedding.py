import numpy as np
import torch
import torch.nn as nn
from utils import constants
import torch

class WordEmbedding(nn.Module):
    # pylint: disable=arguments-differ
    def __init__(self, vocab, embedding_size, pretrained=None):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = vocab.size
        self.embedding_size = embedding_size

        if pretrained is not None:
            #x = torch.matmul(torch.transpose(pretrained),pretrained)/(pretrained.shape[0]-1)
            #U,S,V = torch.svd(x)
            #pretrained = torch.dot(U,S[:embedding_size,:embedding_size])
            pretrained_tensor = self.dict2tensor(self.vocab_size, embedding_size, pretrained)
        else:
            pretrained_tensor = None

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size,
                                      _weight=pretrained_tensor, padding_idx=0)
        self.embedding.weight.requires_grad = True

    def dict2tensor(self, vocab_size, embedding_size, pretrained_dict):
        scale = np.sqrt(3.0 / embedding_size)
        for word, index in self.vocab.items():
            if word in pretrained_dict:
                embedding = pretrained_dict[word]
                pre_shape = embedding.shape[1]
                break


        pretrained = np.empty([vocab_size, pre_shape], dtype=np.float32)
        pretrained[:3, :] = np.random.uniform(
            - scale, scale, [3, pre_shape]).astype(np.float32)  # Special symbols

        oov = 0
        for word, index in self.vocab.items():
            if word in pretrained_dict:
                embedding = pretrained_dict[word]
            elif word.lower() in pretrained_dict:
                embedding = pretrained_dict[word.lower()]
            else:
                embedding = np.random.uniform(-scale, scale, [1, pre_shape]).astype(np.float32)
                oov += 1
            pretrained[index, :] = embedding
        pretrained = torch.from_numpy(pretrained)
        x = torch.matmul(pretrained,pretrained.transpose(1,0))/(pretrained.shape[0]-1)
        U,S,V = torch.svd(x)
        print(pretrained.shape)
        print(U.shape)
        pretrained = torch.dot(U,torch.diag(S)[:,:embedding_size])
        print(pretrained.shape)

        print('# OOV words: %d' % oov)
        return pretrained#torch.from_numpy(pretrained)

    def forward(self, x):
        return self.embedding(x)


class ActionEmbedding(nn.Module):
    # pylint: disable=arguments-differ
    def __init__(self, actions, embedding_size):
        super().__init__()
        self.actions = actions
        self.action_size = len(actions)
        self.embedding_size = embedding_size

        #pretrained_tensor = torch.zeros(size=(self.action_size, self.embedding_size)).to(device=constants.device)
        #for i in range(self.action_size):
        #    pretrained_tensor[i, i] = 1

        self.embedding = nn.Embedding(self.action_size, self.embedding_size, padding_idx=0)
        self.embedding.weight.requires_grad = True

    def forward(self, x):
        return self.embedding(x)
