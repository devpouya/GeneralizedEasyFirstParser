import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders
from h02_learn.model import BiaffineParser
from h02_learn.train_info import TrainInfo
from utils import constants
from utils import utils


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--language', type=str, required=True)
    parser.add_argument('--data-path', type=str, default='data/')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--batch-size-eval', type=int, default=128)
    # Model
    parser.add_argument('--nlayers', type=int, default=3)
    parser.add_argument('--embedding-size', type=int, default=100)
    parser.add_argument('--hidden-size', type=int, default=400)
    parser.add_argument('--arc-size', type=int, default=500)
    parser.add_argument('--label-size', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=.33)
    # Optimization
    parser.add_argument('--eval-batches', type=int, default=200)
    parser.add_argument('--wait-epochs', type=int, default=5)
    parser.add_argument('--lr-decay', type=float, default=.5)
    # Save
    parser.add_argument('--checkpoints-path', type=str, default='checkpoints/')
    parser.add_argument('--seed', type=int, default=7)

    args = parser.parse_args()
    args.wait_iterations = args.wait_epochs * args.eval_batches
    args.save_path = '%s/%s/' % (args.checkpoints_path, args.language)
    utils.config(args.seed)
    return args


def get_model(vocabs, embeddings, args):
    return BiaffineParser(
        vocabs, args.embedding_size, args.hidden_size, args.arc_size, args.label_size,
        nlayers=args.nlayers, dropout=args.dropout, pretrained_embeddings=embeddings) \
        .to(device=constants.device)


def calculate_attachment_score(heads_tgt, l_logits, heads, rels):
    acc_h = (heads_tgt == heads)[heads != -1]
    acc_l = (l_logits.argmax(-1) == rels)[heads != -1]

    uas = acc_h.float().mean().item()
    las = (acc_h & acc_l).float().mean().item()

    return las, uas


def get_sentence_mst(logprobs_orig, eps=1e-5):
    logprobs = logprobs_orig.copy() + eps

    # Redue graph to make computation faster
    temp = logprobs[:, 0].copy()
    max_min = logprobs.min(-1).max()
    logprobs[(logprobs > max_min * 1e2)] = 0
    logprobs[:, 0] = temp
    logprobs[0, :] = 0 # Remove edges from head
    np.fill_diagonal(logprobs, 0) # Remove self edges

    graph = nx.DiGraph(logprobs.transpose())
    try:
        mst = nx.minimum_spanning_arborescence(graph)
    except nx.exception.NetworkXException:
        logprobs = logprobs_orig + eps
        graph = nx.DiGraph(logprobs.transpose())
        mst = nx.minimum_spanning_arborescence(graph)

    assert all([mst.in_degree[x] == 1 for x in range(1, logprobs.shape[1])])
    return mst


def get_mst(text, h_logits):
    # pylint: disable=no-member
    sent_lens = (text != 0).sum(-1)

    batch_size, max_len, _ = h_logits.shape
    heads_tgt = np.ones((batch_size, max_len)) * -1
    heads_tgt[:, 0] = 0

    for i in range(len(h_logits)):
        sent_len = sent_lens[i]
        h_logit = h_logits[i, :sent_len, :sent_len].clone()

        # Get logprobs and add epsolon to account for 0 logprob edges
        logprobs = - F.log_softmax(h_logit, dim=-1).cpu().numpy()
        mst = get_sentence_mst(logprobs)

        for src, tgt in mst.edges:
            heads_tgt[i, tgt] = src

    return torch.LongTensor(heads_tgt).to(device=constants.device)


def _evaluate(evalloader, model, use_mst):
    # pylint: disable=too-many-locals
    criterion_h = nn.CrossEntropyLoss(ignore_index=-1) \
        .to(device=constants.device)
    criterion_l = nn.CrossEntropyLoss(ignore_index=0) \
        .to(device=constants.device)

    dev_loss, dev_las, dev_uas, n_instances = 0, 0, 0, 0
    for (text, pos), (heads, rels) in evalloader:
        h_logits, l_logits = model((text, pos))
        loss = criterion_h(h_logits.reshape(-1, h_logits.shape[-1]), heads.reshape(-1)).item()
        loss += criterion_l(l_logits.reshape(-1, l_logits.shape[-1]), rels.reshape(-1)).item()

        if use_mst:
            heads_tgt = get_mst(text, h_logits)
        else:
            heads_tgt = h_logits.argmax(-1)
            heads_tgt[heads == -1] = -1

        las, uas = calculate_attachment_score(heads_tgt, l_logits, heads, rels)

        batch_size = text.shape[0]
        dev_loss += (loss * batch_size)
        dev_las += (las * batch_size)
        dev_uas += (uas * batch_size)
        n_instances += batch_size

    return dev_loss / n_instances, dev_las / n_instances, dev_uas / n_instances


def evaluate(evalloader, model, use_mst=True):
    model.eval()
    with torch.no_grad():
        result = _evaluate(evalloader, model, use_mst=use_mst)
    model.train()
    return result


def train_batch(text, pos, heads, rels, model, optimizer, criterions):
    criterion_h, criterion_l = criterions
    optimizer.zero_grad()
    text, pos = text.to(device=constants.device), pos.to(device=constants.device)
    heads, rels = heads.to(device=constants.device), rels.to(device=constants.device)
    h_logits, l_logits = model((text, pos))
    loss = criterion_h(h_logits.reshape(-1, h_logits.shape[-1]), heads.reshape(-1))
    loss += criterion_l(l_logits.reshape(-1, l_logits.shape[-1]), rels.reshape(-1))
    loss.backward()
    optimizer.step()

    return loss.item()


def train(trainloader, devloader, model, eval_batches, wait_iterations, lr_decay):
    # pylint: disable=too-many-locals
    optimizer = optim.AdamW(model.parameters(), betas=(.9, .9))
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)
    criterions = (nn.CrossEntropyLoss(ignore_index=-1) \
        .to(device=constants.device), \
        nn.CrossEntropyLoss(ignore_index=0) \
        .to(device=constants.device))
    train_info = TrainInfo(wait_iterations, eval_batches)

    while not train_info.finish:
        for (text, pos), (heads, rels) in trainloader:
            loss = train_batch(text, pos, heads, rels, model, optimizer, criterions)
            train_info.new_batch(loss)

            if train_info.eval:
                dev_results = evaluate(devloader, model, use_mst=False)

                if train_info.is_best(dev_results):
                    model.set_best()
                elif train_info.reduce_lr:
                    lr_scheduler.step()
                    optimizer.state.clear()
                    model.recover_best()
                    print('\tReduced lr')
                elif train_info.finish:
                    train_info.print_progress(dev_results)
                    break
                train_info.print_progress(dev_results)

    model.recover_best()


def main():
    # pylint: disable=too-many-locals
    args = get_args()

    trainloader, devloader, testloader, vocabs, embeddings = \
        get_data_loaders(args.data_path, args.language, args.batch_size, args.batch_size_eval)
    print('Train size: %d Dev size: %d Test size: %d' %
          (len(trainloader.dataset), len(devloader.dataset), len(testloader.dataset)))

    model = get_model(vocabs, embeddings, args)
    train(trainloader, devloader, model, args.eval_batches, args.wait_iterations, args.lr_decay)

    model.save(args.save_path)

    # train_loss, train_las, train_uas = evaluate(trainloader, model, use_mst=False)
    train_loss, train_las, train_uas = 0, 0, 0
    dev_loss, dev_las, dev_uas = evaluate(devloader, model, use_mst=False)
    test_loss, test_las, test_uas = evaluate(testloader, model, use_mst=False)

    print('Final Training loss: %.4f Dev loss: %.4f Test loss: %.4f' %
          (train_loss, dev_loss, test_loss))
    print('Final Training las: %.4f Dev las: %.4f Test las: %.4f' %
          (train_las, dev_las, test_las))
    print('Final Training uas: %.4f Dev uas: %.4f Test uas: %.4f' %
          (train_uas, dev_uas, test_uas))

    # train_loss, train_las, train_uas = evaluate(trainloader, model)
    train_loss, train_las, train_uas = 0, 0, 0
    dev_loss, dev_las, dev_uas = evaluate(devloader, model)
    test_loss, test_las, test_uas = evaluate(testloader, model)

    print('Final Training loss: %.4f Dev loss: %.4f Test loss: %.4f' %
          (train_loss, dev_loss, test_loss))
    print('Final Training las: %.4f Dev las: %.4f Test las: %.4f' %
          (train_las, dev_las, test_las))
    print('Final Training uas: %.4f Dev uas: %.4f Test uas: %.4f' %
          (train_uas, dev_uas, test_uas))

if __name__ == '__main__':
    main()
