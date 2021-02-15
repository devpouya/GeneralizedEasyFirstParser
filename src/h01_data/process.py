import sys
from os import path
import argparse
from collections import OrderedDict
import numpy as np

sys.path.append('./src/')
from h01_data import Vocab, save_vocabs, save_embeddings
from h01_data.oracle import arc_standard_oracle, arc_eager_oracle
from utils import utils
from utils import constants


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, required=True)
    parser.add_argument('--data-path', type=str, default='data/')
    parser.add_argument('--glove-file', type=str, required=True)
    parser.add_argument('--min-vocab-count', type=int, default=2)
    parser.add_argument('--transition',type=str,default='arc-standard')
    return parser.parse_args()


def get_sentence(file):
    sentence = []
    for line in file:
        tokens = line.strip().split()
        if tokens and tokens[0].isdigit():
            sentence += [tokens]
        elif tokens:
            pass
        else:
            yield sentence
            sentence = []


def process_sentence(sentence, vocabs, transition_system=None):
    words, tags, rels = vocabs
    processed = [{
        'word': words.ROOT,
        'word_id': words.ROOT_IDX,
        'tag1': tags.ROOT,
        'tag1_id': tags.ROOT_IDX,
        'tag2': tags.ROOT,
        'tag2_id': tags.ROOT_IDX,
        'head': 0,
        'rel': rels.ROOT,
        'rel_id': rels.ROOT_IDX,
    }]
    heads = []
    for token in sentence:
        processed += [{
            'word': token[1],
            'word_id': words.idx(token[1]),
            'tag1': token[3],
            'tag1_id': tags.idx(token[3]),
            'tag2': token[4],
            'tag2_id': tags.idx(token[4]),
            'head': int(token[6]),
            # 'head_id': token[6],
            'rel': token[7],
            'rel_id': rels.idx(token[7]),
        }]

        heads.append(int(token[6]))

    return processed


def process_data(in_fname_base, out_path, mode, vocabs, transition_system=None,transition_name=None):
    in_fname = in_fname_base % mode
    out_fname = '%s/%s.json' % (out_path, mode)
    #if transition_system is not None:
    #    out_fname_history = '%s/%s_actions_%s.json' % (out_path, transition_name,mode)
    #    utils.remove_if_exists(out_fname_history)

    utils.remove_if_exists(out_fname)
    print('Processing: %s' % in_fname)
    with open(in_fname, 'r') as file:
        for sentence in get_sentence(file):
            sent_processed = process_sentence(sentence, vocabs)
            utils.append_json(out_fname, sent_processed)


def add_sentence_vocab(sentence, words, tags, rels):
    for token in sentence:
        words.count_up(token[1])
        tags.count_up(token[3])
        tags.count_up(token[4])
        rels.count_up(token[7])


def process_vocabs(words, tags, rels):
    words.process_vocab()
    tags.process_vocab()
    rels.process_vocab()


def add_embedding_vocab(embeddings, words):
    for word in embeddings.keys():
        words.add_pretrained(word)


def get_vocabs(in_fname_base, out_path, min_count, embeddings=None):
    in_fname = in_fname_base % 'train'
    words, tags, rels = Vocab(min_count), Vocab(min_count), Vocab(min_count)
    print('Getting vocabs: %s' % in_fname)

    with open(in_fname, 'r') as file:
        for sentence in get_sentence(file):
            add_sentence_vocab(sentence, words, tags, rels)

    if embeddings is not None:
        add_embedding_vocab(embeddings, words)

    process_vocabs(words, tags, rels)
    save_vocabs(out_path, words, tags, rels)
    return (words, tags, rels)


def read_embeddings(fname):
    # loading GloVe
    embedd_dim = -1
    embedd_dict = OrderedDict()
    # with gzip.open(fname, 'rt') as file:
    with open(fname, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            if len(line) == 0:
                continue

            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            elif embedd_dim == len(tokens):
                # Skip empty word
                continue
            else:
                assert embedd_dim + 1 == len(tokens), 'Dimension of embeddings should be consistent'

            embedd = np.empty([1, embedd_dim], dtype=np.float32)
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd

    return embedd_dict


def process_embeddings(src_fname, out_path):
    embedding_dict = read_embeddings(src_fname)
    save_embeddings(out_path, embedding_dict)
    return embedding_dict


def main():
    args = get_args()

    in_fname = path.join(args.data_path, constants.UD_LANG_FNAMES[args.language])
    out_path = path.join(args.data_path, constants.UD_PATH_PROCESSED, args.language)
    utils.mkdir(out_path)

    embeddings = process_embeddings(args.glove_file, out_path)

    vocabs = get_vocabs(in_fname, out_path, min_count=args.min_vocab_count, embeddings=embeddings)
    transition_system = None
    if args.transition == 'arc-standard':
        transition_system = arc_standard_oracle
    elif args.transition == 'arc-eager':
        transition_system = arc_eager_oracle

    process_data(in_fname, out_path, 'train', vocabs, transition_system,args.transition)
    process_data(in_fname, out_path, 'dev', vocabs, transition_system,args.transition)
    process_data(in_fname, out_path, 'test', vocabs, transition_system,args.transition)


if __name__ == '__main__':
    main()
