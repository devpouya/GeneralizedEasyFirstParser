import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import constants
from .biaffine import BiaffineParser


class MSTParser(BiaffineParser):
    @classmethod
    def loss(cls, h_logits, l_logits, heads, rels):
        batch_size, _, _ = h_logits.shape
        sent_lens = (heads != -1).sum(-1)
        label_criterion = nn.CrossEntropyLoss(ignore_index=0).to(device=constants.device)
        label_loss = label_criterion(l_logits.reshape(-1, l_logits.shape[-1]), rels.reshape(-1))
        arc_loss = torch.FloatTensor([0.])
        for i in range(batch_size):
            sent_len = sent_lens[i]
            arc_loss += cls.sentence_loss(h_logits[i, :sent_len, :sent_len], heads[i][:sent_len])
        return label_loss + (arc_loss / sent_lens.sum())

    @staticmethod
    def sentence_loss(h_logit, head):
        loss = torch.FloatTensor([0.])
        logprob = F.log_softmax(h_logit, dim=-1)
        A = F.softmax(h_logit, dim=-1)

        diag_mask = 1.0 - torch.eye(len(head)).type_as(logprob)
        A = A * diag_mask
        logprob = logprob * diag_mask

        D = A.sum(dim=1)
        atol = 1e-6
        D += atol
        D = torch.diag_embed(D)
        L = D - A
        loss += torch.logdet(L[1:, 1:])
        loss -= logprob[torch.arange(0, len(head)), head][1:].sum()
        return loss
