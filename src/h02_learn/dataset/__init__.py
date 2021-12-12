from os import path
from torch.utils.data import DataLoader

from h01_data import load_vocabs, get_ud_fname, get_oracle_actions
from utils import constants
from .syntax import SyntaxDataset
from transformers import BertTokenizer, BertTokenizerFast
from transformers import AutoTokenizer


def generate_batch(batch):
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
    map_length = max([len(entry[0][1]) for entry in batch])
    max_length_actions = max([len(entry[2]) for entry in batch])
    text = tensor.new_zeros(batch_size, max_length_text)
    text_mappings = tensor.new_ones(batch_size, map_length) * -1
    heads = tensor.new_ones(batch_size, max_length) * -1
    rels = tensor.new_zeros(batch_size, max_length)
    transitions = tensor.new_ones(batch_size, max_length_actions, 2) * -1

    for i, sentence in enumerate(batch):
        sent_len = len(sentence[0][0])
        head_len = len(sentence[1][0])
        map_len = len(sentence[0][1])
        text_mappings[i, :map_len] = sentence[0][1]
        text[i, :sent_len] = sentence[0][0]
        heads[i, :head_len] = sentence[1][0]
        rels[i, :head_len] = sentence[1][1]

    for i, sentence in enumerate(batch):
        num_actions = len(sentence[2])
        transitions[i, :num_actions] = sentence[2]


    return (text, text_mappings), (heads, rels), transitions


def get_data_loader(fname, transitions_file, tokenizer, batch_size, shuffle):
    dataset = SyntaxDataset(fname, transitions_file, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=lambda batch: generate_batch(batch))


def get_data_loaders(data_path, language, batch_size, batch_size_eval, is_easy_first=True):
    src_path = path.join(data_path, constants.UD_PATH_PROCESSED, language)
    (fname_train, fname_dev, fname_test) = get_ud_fname(src_path)

    (transitions_train, transitions_dev, transitions_test) = get_oracle_actions(src_path,is_easy_first)
    vocabs = load_vocabs(src_path)




    if language == "en":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif language == "de":
        tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
    elif language == "cs":
        tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/bert-base-bg-cs-pl-ru-cased")
    elif language == "eu":
        # Basque
        tokenizer = AutoTokenizer.from_pretrained("ixa-ehu/berteus-base-cased")
    elif language == "tr":
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    trainloader = get_data_loader(fname_train, transitions_train, tokenizer,
                                                      batch_size,
                                                      shuffle=True)
    devloader = get_data_loader(fname_dev, transitions_dev, tokenizer,
                                                  batch_size_eval,
                                                  shuffle=False)
    testloader = get_data_loader(fname_test, transitions_test, tokenizer,
                                                    batch_size_eval,
                                                    shuffle=False)

    return trainloader, devloader, testloader, vocabs
