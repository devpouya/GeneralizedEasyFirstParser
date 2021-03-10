import numpy as np
import torch
import torch.nn as nn
from utils import constants
import torch
from transformers import BertModel

class WordEmbedding(nn.Module):
    # pylint: disable=arguments-differ
    def __init__(self, vocab, embedding_size, pretrained=None):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = vocab.size
        self.embedding_size = embedding_size

        if pretrained is not None:
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
                big_size = pretrained_dict[word].shape[1]
                break


        pretrained = np.empty([vocab_size, big_size], dtype=np.float32)
        pretrained[:3, :] = np.random.uniform(
            - scale, scale, [3, big_size]).astype(np.float32)  # Special symbols
        oov = 0
        for word, index in self.vocab.items():
            if word in pretrained_dict:
                embedding = pretrained_dict[word]
            elif word.lower() in pretrained_dict:
                embedding = pretrained_dict[word.lower()]
            else:
                embedding = np.random.uniform(-scale, scale, [1, big_size]).astype(np.float32)
                oov += 1
            pretrained[index, :] = embedding

        # do PCA on higher dimensional data
        pretrained = torch.from_numpy(pretrained)
        if big_size > embedding_size:
            U,S,V = torch.pca_lowrank(pretrained,q=embedding_size)
            pretrained = torch.matmul(pretrained,V[:,:embedding_size])
            print("Embedding dimensionality bigger than wanted\n PCA'd data down to {}".format(pretrained.shape))
        print('# OOV words: %d' % oov)
        return pretrained#torch.from_numpy(pretrained)

    def forward(self, x):
        return self.embedding(x)


class BertEmbedding(nn.Module):
    # can probably throw away the stupid 32 dimensional "learned" embeddings lol
    def __init__(self,bert_model):
        super().__init__()
        self.model = BertModel.from_pretrained(bert_model)

    def token2tensor(self):

        pass
    def forward(self, x):
        pass


class ActionEmbedding(nn.Module):
    # pylint: disable=arguments-differ
    def __init__(self, actions, embedding_size):
        super().__init__()
        self.actions = actions
        self.action_size = len(actions)
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(self.action_size, self.embedding_size, padding_idx=0)
        self.embedding.weight.requires_grad = True

    def forward(self, x):
        return self.embedding(x)



