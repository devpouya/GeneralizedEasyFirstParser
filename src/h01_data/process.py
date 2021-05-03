import sys
from os import path
import argparse
from collections import OrderedDict
import numpy as np

sys.path.append('./src/')
from h01_data import Vocab, save_vocabs, save_embeddings
from h01_data.oracle import arc_standard_oracle, arc_eager_oracle, hybrid_oracle, easy_first_arc_standard, \
    easy_first_hybrid, easy_first_arc_eager, easy_first_mh4, mh4_oracle, easy_first_pending, priority_queue_oracle
from h01_data.oracle import is_projective,projectivize,projectivize_mh4,get_arcs
from utils import utils
from utils import constants
from h01_data.item_oracle import item_arc_standard_oracle,build_easy_first,build_easy_first_mh4


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, required=True)
    parser.add_argument('--data-path', type=str, default='data/')
    parser.add_argument('--save-path', type=str, default='data/')
    parser.add_argument('--glove-file', type=str, required=False)
    parser.add_argument('--bert-model', type=str, default='bert-base-cased')
    parser.add_argument('--min-vocab-count', type=int, default=2)
    parser.add_argument('--transition', type=str, choices=['arc-standard', 'arc-eager', 'easy-first',
                                                           'hybrid', 'mh4', 'easy-first-std',
                                                           'easy-first-hybrid',
                                                           'agenda-std','agenda-hybrid','agenda-mh4'],
                        default='agenda-std')
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


def process_sentence(sentence, vocabs):
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
    relations = []
    rel2id = {}
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
        relations.append(token[7])
        rel2id[token[7]] = rels.idx(token[7])

    return processed, heads, relations, rel2id


def labeled_action_pairs(actions, relations):
    labeled_acts = []
    tmp_rels = relations.copy()

    for act in actions:
        if act == constants.shift or act is None:
            labeled_acts.append((act, -1))
        elif act is not None:
            labeled_acts.append((act, tmp_rels[0]))
            tmp_rels.pop(0)

    return labeled_acts


def process_data(in_fname_base, out_path, mode, vocabs, oracle=None, transition_name=None):
    in_fname = in_fname_base % mode
    out_fname = '%s/%s_.json' % (out_path, mode)
    if oracle is not None:
        out_fname_history = '%s/%s_actions_%s.json' % (out_path, transition_name, mode)
        utils.remove_if_exists(out_fname_history)

    utils.remove_if_exists(out_fname)
    print('Processing: %s' % in_fname)
    right = 0
    wrong = 0
    step = 0
    faileds = []
    with open(in_fname, 'r') as file:
        step = 0
        for sentence in get_sentence(file):
            step+=1
            #if step >= 50:
            #    break
            sent_processed, heads, relations,rel2id = process_sentence(sentence, vocabs)
            heads_proper = [0] + heads

            sentence_proper = list(range(len(heads_proper)))

            word2head = {w: h for (w, h) in zip(sentence_proper, heads_proper)}
            if is_projective(word2head) or transition_name == 'mh4' or transition_name == 'agenda-mh4':
                true_arcs = get_arcs(word2head)
            else:
                true_arcs = projectivize(word2head)
            if transition_name == 'mh4' or transition_name == 'agenda-mh4':
                good = False
                while not good:
                    _, _, good = mh4_oracle(sentence_proper, word2head, relations,true_arcs)
                    if good:
                        break
                    else:
                        word2head, true_arcs = projectivize_mh4(word2head,true_arcs)
                actions, _, good = oracle(sentence_proper, word2head, relations, true_arcs)
            else:
                actions, _, good = oracle(sentence_proper, word2head, relations, true_arcs)

            relation_ids = [rel2id[rel] for rel in relations]
            if good:
                right += 1
            else:
                faileds.append(step)
                wrong += 1
            actions_processed = {'transition': actions, 'relations':relation_ids,}
            utils.append_json(out_fname_history, actions_processed)
            utils.append_json(out_fname, sent_processed)

    print("GOOD {}".format(right))
    print("BAD {}".format(wrong))

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


def add_tokenizer_vocab(tokenizer, words):
    pass


def get_vocabs(in_fname_base, out_path, min_count, embeddings=None, tokenizer=None):
    in_fname = in_fname_base % 'train'
    words, tags, rels = Vocab(min_count), Vocab(min_count), Vocab(min_count)
    print('Getting vocabs: %s' % in_fname)

    with open(in_fname, 'r') as file:
        for sentence in get_sentence(file):
            add_sentence_vocab(sentence, words, tags, rels)

    if embeddings is not None:
        add_embedding_vocab(embeddings, words)
    elif tokenizer is not None:
        add_tokenizer_vocab(tokenizer, words)

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
            # print("88888888888")
            # print(embedd_dim)
            # print(len(tokens))
            # print("88888888888")

            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            elif embedd_dim == len(tokens):
                # Skip empty word
                continue
            else:
                continue
                # assert embedd_dim + 1 == len(tokens), 'Dimension of embeddings should be consistent'

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
    out_path = path.join(args.save_path, constants.UD_PATH_PROCESSED, args.language)
    utils.mkdir(out_path)

    # embeddings = process_embeddings(args.glove_file, out_path)
    tokenizer = None  # BertTokenizer.from_pretrained(args.bert_model)
    vocabs = get_vocabs(in_fname, out_path, min_count=args.min_vocab_count, tokenizer=tokenizer)
    oracle = None
    if args.transition == 'arc-standard':
        oracle = arc_standard_oracle
    elif args.transition == 'arc-eager':
        oracle = arc_eager_oracle
    elif args.transition == 'hybrid':
        oracle = hybrid_oracle
    elif args.transition == 'mh4':
        oracle = mh4_oracle
    elif args.transition == 'agenda-std' or args.transition == 'agenda-hybrid':
        #oracle = item_arc_standard_oracle
        oracle = build_easy_first
    elif args.transition == "easy-first-std":
        #oracle = easy_first_arc_standard
        oracle = build_easy_first#easy_first_arc_standard
    elif args.transition == "easy-first-hybrid":
        oracle = easy_first_hybrid
    elif args.transition == "easy-first-eager":
        oracle = easy_first_arc_eager
    elif args.transition == 'agenda-mh4':
        oracle = build_easy_first_mh4
    elif args.transition == 'easy-first':
        oracle = easy_first_pending

    process_data(in_fname, out_path, 'train', vocabs, oracle, args.transition)
    process_data(in_fname, out_path, 'dev', vocabs, oracle, args.transition)
    process_data(in_fname, out_path, 'test', vocabs, oracle, args.transition)


if __name__ == '__main__':
    main()
