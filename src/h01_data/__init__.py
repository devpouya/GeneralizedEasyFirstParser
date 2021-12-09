import sys
from os import path

from h01_data.vocab import Vocab
from utils import utils
from utils import constants


def save_vocabs(fpath, words, tags, rels):
    fname = '%s/vocabs.pckl' % (fpath)
    vocabs = (words, tags, rels)
    utils.write_data(fname, vocabs)


def load_vocabs(fpath):
    fname = '%s/vocabs.pckl' % (fpath)
    words, tags, rels = utils.read_data(fname)
    return (words, tags, rels)


def save_embeddings(fpath, embeddings):
    fname = '%s/embeddings.pckl' % (fpath)
    utils.write_data(fname, embeddings)


def load_embeddings(fpath):
    fname = '%s/embeddings.pckl' % (fpath)
    return utils.read_data(fname)


def get_ud_fname(fpath):
    fname_train = '%s/%s.json' % (fpath, 'train')
    fname_dev = '%s/%s.json' % (fpath, 'dev')
    fname_test = '%s/%s.json' % (fpath, 'test')
    return (fname_train, fname_dev, fname_test)

def get_oracle_actions(fpath,is_easy_first=True):
    if not is_easy_first:
        fname_train = '%s/shift_reduce_actions_%s.json' % (fpath,'train')
        fname_dev = '%s/shift_reduce_actions_%s.json' % (fpath,'dev')
        fname_test = '%s/shift_reduce_actions_%s.json' % (fpath,'test')
    else:
        fname_train = '%s/easy_first_actions_%s.json' % (fpath, 'train')
        fname_dev = '%s/easy_first_actions_%s.json' % (fpath, 'dev')
        fname_test = '%s/easy_first_actions_%s.json' % (fpath, 'test')
    return (fname_train, fname_dev, fname_test)
