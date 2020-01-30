import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from .base import BaseParser
from .modules import Biaffine, Bilinear
from .word_embedding import WordEmbedding


class BiaffineParser(BaseParser):
    # pylint: disable=arguments-differ,too-many-instance-attributes,too-many-arguments
    def __init__(self, vocabs, embedding_size, hidden_size, arc_size, label_size,
                 nlayers=3, dropout=0.33, pretrained_embeddings=None):
        super().__init__()

        self.vocabs = vocabs
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.arc_size = arc_size
        self.label_size = label_size
        self.nlayers = nlayers
        self.dropout_p = dropout

        self.words_embedding, self.tags_embedding = \
            self.create_embeddings(vocabs, pretrained=pretrained_embeddings)

        self.lstm = nn.LSTM(
            embedding_size * 2, hidden_size, nlayers, dropout=(dropout if nlayers > 1 else 0),
            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        self.linear_arc_dep = nn.Linear(hidden_size * 2, arc_size)
        self.linear_arc_head = nn.Linear(hidden_size * 2, arc_size)
        self.biaffine = Biaffine(arc_size, arc_size)

        _, _, rels = vocabs
        self.linear_label_dep = nn.Linear(hidden_size * 2, label_size)
        self.linear_label_head = nn.Linear(hidden_size * 2, label_size)
        self.bilinear_label = Bilinear(label_size, label_size, rels.size)

    def create_embeddings(self, vocabs, pretrained=None):
        words, tags, _ = vocabs
        words_embedding = WordEmbedding(words, self.embedding_size, pretrained=pretrained)
        tags_embedding = nn.Embedding(tags.size, self.embedding_size)
        return words_embedding, tags_embedding

    def forward(self, x, head=None):
        x_emb = self.dropout(self.get_embeddings(x))

        sent_lens = (x[0] != 0).sum(-1)
        h_t = self.run_lstm(x_emb, sent_lens)
        h_logits = self.get_head_logits(h_t, sent_lens)

        if head is None:
            head = h_logits.argmax(-1)
        l_logits = self.get_label_logits(h_t, head)

        return h_logits, l_logits

    def get_embeddings(self, x):
        return torch.cat([self.words_embedding(x[0]), self.tags_embedding(x[1])], dim=-1)

    def run_lstm(self, x, sent_lens):
        lstm_in = pack_padded_sequence(x, sent_lens, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(lstm_in)
        h_t, _ = pad_packed_sequence(lstm_out, batch_first=True)
        h_t = self.dropout(h_t).contiguous()

        return h_t

    def get_head_logits(self, h_t, sent_lens):
        h_dep = self.dropout(F.relu(self.linear_arc_dep(h_t)))
        h_arc = self.dropout(F.relu(self.linear_arc_head(h_t)))

        h_logits = self.biaffine(h_arc, h_dep)

        # Zero logits for items after sentence length
        for i, sent_len in enumerate(sent_lens):
            h_logits[i, sent_len:, :] = 0
            h_logits[i, :, sent_len:] = 0

        return h_logits

    def get_label_logits(self, h_t, head):
        l_dep = self.dropout(F.relu(self.linear_label_dep(h_t)))
        l_head = self.dropout(F.relu(self.linear_label_head(h_t)))

        if self.training:
            assert head is not None, 'During training head should not be None'

        l_head = l_head.gather(dim=1, index=head.unsqueeze(2).expand(l_head.size()))
        l_logits = self.bilinear_label(l_dep, l_head)
        return l_logits

    def get_args(self):
        return {
            'vocabs': self.vocabs,
            'embedding_size': self.embedding_size,
            'hidden_size': self.hidden_size,
            'arc_size': self.arc_size,
            'label_size': self.label_size,
            'nlayers': self.nlayers,
            'dropout': self.dropout_p,
        }
