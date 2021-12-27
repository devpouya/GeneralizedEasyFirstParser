from os import path
from torch.utils.data import DataLoader

from h01_data import load_vocabs, load_embeddings, get_ud_fname, get_oracle_actions#,get_oracle_actions_small,get_ud_fname_small
from utils import constants
from .syntax import SyntaxDataset, LanguageBatchSampler
from transformers import BertTokenizer, BertTokenizerFast
from transformers import AutoTokenizer,RobertaTokenizerFast


def generate_batch(batch,transition_system):
    r"""
    Since the text entries have different lengths, a custom function
    generate_batch() is used to generate data batches and offsets,
    which are compatible with EmbeddingBag. The function is passed
    to 'collate_fn' in torch.utils.data.DataLoader. The input to
    'collate_fn' is a list of tensors with the size of batch_size,
    and the 'collate_fn' function packs them into a mini-batch.[len(entry[0][0]) for entry in batch]
    Pay attention here and make sure that 'collate_fn' is declared
    as a top level def. This ensures that the function is available
    in each worker.
    Output:
        text: the text entries in the data_batch are packed into a list and
            concatenated as a single tensor for the input of nn.EmbeddingBag.
        offsets: the offsets is a tensor of delimiters to represent the beginning
            index of the individual sequence in the text tensor.
        cls: a tensor saving the labels of individual text entries.
    """
    tensor = batch[0][0][0]

    # for entry in batch:
    #    print(entry[2])
    batch_size = len(batch)
    max_length_text = max([len(entry[0][0]) for entry in batch])
    max_length = max([len(entry[0][1]) for entry in batch])
    map_length = max([len(entry[3][0]) for entry in batch])
    max_length_actions = max([len(entry[2][0]) for entry in batch])
    text = tensor.new_zeros(batch_size, max_length_text)
    text_mappings = tensor.new_ones(batch_size, map_length) * -1
    pos = tensor.new_zeros(batch_size, max_length)
    heads = tensor.new_ones(batch_size, max_length) * -1
    rels = tensor.new_zeros(batch_size, max_length)
    if transition_system == constants.agenda:
        transitions = tensor.new_ones(batch_size, max_length_actions, 2) * -1
    else:
        transitions = tensor.new_ones(batch_size, max_length_actions) * -1
    relations_in_order = tensor.new_zeros(batch_size, max_length)

    for i, sentence in enumerate(batch):
        sent_len = len(sentence[0][0])
        pos_len = len(sentence[0][1])
        map_len = len(sentence[3][0])
        text_mappings[i, :map_len] = sentence[3][0]
        text[i, :sent_len] = sentence[0][0]
        pos[i, :pos_len] = sentence[0][1]
        heads[i, :pos_len] = sentence[1][0]
        rels[i, :pos_len] = sentence[1][1]

    for i, sentence in enumerate(batch):
        num_actions = len(sentence[2][0])
        transitions[i, :num_actions] = sentence[2][0]

        num_rels = len(sentence[2][1])
        relations_in_order[i, :num_rels] = sentence[2][1]

    return (text, pos), (heads, rels), (transitions, relations_in_order), text_mappings


def get_data_loader(fname, transitions_file, transition_system, tokenizer, batch_size, rel_size, shuffle):
    dataset = SyntaxDataset(fname, transitions_file, transition_system, tokenizer, rel_size)
    #print(dataset.language_starts)
    sampler = LanguageBatchSampler(batch_size=batch_size,language_start_indicies=dataset.language_starts, shuffle=shuffle)
    #return DataLoader(dataset, collate_fn=lambda batch: generate_batch(batch,transition_system)), dataset.max_rel
    return DataLoader(dataset, collate_fn=lambda batch: generate_batch(batch,transition_system), batch_sampler=sampler), dataset.max_sent_len


def get_data_loaders(data_path, all_languages, batch_size, batch_size_eval, transitions=None, transition_system=None,
                     bert_model=None, is_easy_first=True):
    all_fnames_train = []
    all_fnames_test = {}
    all_fnames_dev = {}
    all_transitions_train = []
    all_transitions_test = {}
    all_transitions_dev = {}
    max_rels_size = 0
    for language in all_languages:
        src_path = path.join(data_path, constants.UD_PATH_PROCESSED, language)
        vocabs = load_vocabs(src_path)
        _,_,rels = vocabs
        if rels.size > max_rels_size:
            max_rels_size = rels.size
        (fname_train, fname_dev, fname_test) = get_ud_fname(src_path)
        all_fnames_train.append(fname_train)
        all_fnames_test[language] = fname_test
        all_fnames_dev[language] = fname_dev
        #all_fnames_test.append(fname_test)
        #all_fnames_dev.append(fname_dev)
        transitions_train, transitions_dev, transitions_test = None, None, None
        if transitions is not None:
            if transition_system == "AGENDA-PARSER":
                is_agenda=True
            else:
                is_agenda=False
            (transitions_train, transitions_dev, transitions_test) = get_oracle_actions(src_path, transitions,is_easy_first)
        all_transitions_train.append(transitions_train)
        all_transitions_test[language] = transitions_test
        all_transitions_dev[language] = transitions_dev
        #all_transitions_test.append(transitions_test)
        #all_transitions_dev.append(transitions_dev)
    if len(all_languages) > 1:
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    else:
        l = all_languages[0]
        if l == "eu":
            tokenizer = AutoTokenizer.from_pretrained("ixa-ehu/berteus-base-cased")
        elif l == "ko":
            tokenizer = AutoTokenizer.from_pretrained("kykim/bert-kor-base")
        elif l == "hu":
            tokenizer = AutoTokenizer.from_pretrained("SZTAKI-HLT/hubert-base-cc")
        elif l == "af":
            tokenizer = RobertaTokenizerFast.from_pretrained("jannesg/takalane_afr_roberta", add_prefix_space=True)
        elif l == "la":
            tokenizer = AutoTokenizer.from_pretrained("cook/cicero-similis")
        elif l == "ur":
            tokenizer = AutoTokenizer.from_pretrained("Geotrend/bert-base-ur-cased")
        elif l == "da":
            tokenizer = BertTokenizer.from_pretrained("Maltehb/danish-bert-botxo")
        elif l == "ga":
            tokenizer = BertTokenizer.from_pretrained("DCU-NLP/bert-base-irish-cased-v1")
        elif l == "lt":
            tokenizer = BertTokenizer.from_pretrained("Geotrend/bert-base-lt-cased")
        elif l == "qhe":
            tokenizer = BertTokenizer.from_pretrained("monsoon-nlp/hindi-bert")
        elif l == "sl":
            tokenizer = BertTokenizer.from_pretrained("EMBEDDIA/sloberta")
        else:
            tokenizer = BertTokenizer.from_pretrained("Geotrend/bert-base-nl-cased")

    #tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    #tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    #tokenizer = AutoTokenizer.from_pretrained("ixa-ehu/berteus-base-cased")
    trainloader, _ = get_data_loader(all_fnames_train, all_transitions_train, transition_system, tokenizer,
                                                      batch_size,max_rels_size,
                                                      shuffle=True)

    devloader_lang_map = {}
    testloader_lang_map = {}
    for language in all_languages:
        devloader, _ = get_data_loader(all_fnames_dev[language], all_transitions_dev[language], transition_system, tokenizer,
                                                      batch_size_eval,max_rels_size,
                                                      shuffle=False)
        testloader, _ = get_data_loader(all_fnames_test[language], all_transitions_test[language], transition_system, tokenizer,
                                                        batch_size_eval,max_rels_size,
                                                        shuffle=False)
        devloader_lang_map[language] = devloader
        testloader_lang_map[language] = testloader

    #max_sent_len = max(max_sent_len_dev, max_sent_len_test, max_sent_len_train)
    return trainloader, devloader_lang_map, testloader_lang_map, max_rels_size
