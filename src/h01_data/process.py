import sys
from os import path
import argparse


sys.path.append('./src/')
from h01_data import Vocab, save_vocabs
from h01_data.oracle import projectivize_mh4, get_arcs, mh4_oracle
from utils import utils
from utils import constants
from h01_data.item_oracle import build_easy_first_mh4, item_mh4_oracle


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, required=True)
    parser.add_argument('--data-path', type=str, default='data/')
    parser.add_argument('--save-path', type=str, default='data_nonproj/')
    parser.add_argument('--min-vocab-count', type=int, default=2)
    parser.add_argument('--easy-first', type=str, choices=["True", "False"], default="True")
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


def process_data(in_fname_base, out_path, mode, vocabs, is_easy_first=True):
    in_fname = in_fname_base % mode
    out_fname = '%s/%s.json' % (out_path, mode)
    if is_easy_first:
        oracle = build_easy_first_mh4
        out_fname_history = '%s/easy_first_actions_%s.json' % (out_path, mode)
        utils.remove_if_exists(out_fname_history)
    else:
        oracle = item_mh4_oracle
        out_fname_history = '%s/shift_reduce_actions_%s.json' % (out_path, mode)
        print(out_fname_history)
        utils.remove_if_exists(out_fname_history)

    print('Processing: %s' % in_fname)
    right = 0
    wrong = 0
    faileds = []
    with open(in_fname, 'r') as file:
        step = 0
        for sentence in get_sentence(file):
            step += 1
            #print(step)
            sent_processed, heads, relations, rel2id = process_sentence(sentence, vocabs)
            heads_proper = [0] + heads

            sentence_proper = list(range(len(heads_proper)))

            word2head = {w: h for (w, h) in zip(sentence_proper, heads_proper)}
            true_arcs = get_arcs(word2head)

            good = False
            while not good:
                _, _, good = mh4_oracle(sentence_proper, word2head, relations, true_arcs)
                if good:
                    break
                else:
                    word2head, true_arcs = projectivize_mh4(word2head, true_arcs)
            actions, _, good = oracle(sentence_proper, word2head, relations, true_arcs)
            relation_ids = [rel2id[rel] for rel in relations]
            if good:
                right += 1
            else:
                faileds.append(step)
                wrong += 1

            actions_processed = {'transition': actions, 'relations': relation_ids}
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


def get_vocabs(in_fname_base, out_path, min_count):
    in_fname = in_fname_base % 'train'
    words, tags, rels = Vocab(min_count), Vocab(min_count), Vocab(min_count)
    with open(in_fname, 'r') as file:
        for sentence in get_sentence(file):
            add_sentence_vocab(sentence, words, tags, rels)

    process_vocabs(words, tags, rels)
    save_vocabs(out_path, words, tags, rels)
    return (words, tags, rels)


def main():
    args = get_args()
    if args.easy_first == "True":
        ef = True
    else:
        ef = False
    in_fname = path.join(args.data_path, constants.UD_LANG_FNAMES[args.language])
    out_path = path.join(args.save_path, constants.UD_PATH_PROCESSED, args.language)
    utils.mkdir(out_path)

    vocabs = get_vocabs(in_fname, out_path, min_count=args.min_vocab_count)

    process_data(in_fname, out_path, 'train', vocabs, ef)
    process_data(in_fname, out_path, 'dev', vocabs, ef)
    process_data(in_fname, out_path, 'test', vocabs, ef)


if __name__ == '__main__':
    main()
