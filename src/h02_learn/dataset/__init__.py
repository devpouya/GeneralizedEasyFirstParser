from os import path
from torch.utils.data import DataLoader

from h01_data import load_vocabs, load_embeddings, get_ud_fname, get_oracle_actions
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
    map_length = max([len(entry[3][0]) for entry in batch])
    max_length_actions = max([len(entry[2][0]) for entry in batch])
    text = tensor.new_zeros(batch_size, max_length_text)
    text_mappings = tensor.new_ones(batch_size,map_length) *-1
    pos = tensor.new_zeros(batch_size, max_length)
    heads = tensor.new_ones(batch_size, max_length) * -1
    rels = tensor.new_zeros(batch_size, max_length)
    transitions = tensor.new_ones(batch_size, max_length_actions) * -1
    relations_in_order = tensor.new_zeros(batch_size, max_length)

    for i, sentence in enumerate(batch):
        sent_len = len(sentence[0][0])
        pos_len = len(sentence[0][1])
        map_len = len(sentence[3][0])
        text_mappings[i,:map_len] = sentence[3][0]
        text[i, :sent_len] = sentence[0][0]
        pos[i, :pos_len] = sentence[0][1]
        heads[i, :pos_len] = sentence[1][0]
        rels[i, :pos_len] = sentence[1][1]

    for i, sentence in enumerate(batch):
        num_actions = len(sentence[2][0])
        # print(num_actions)
        transitions[i, :num_actions] = sentence[2][0]
        num_rels = len(sentence[2][1])
        relations_in_order[i, :num_rels] = sentence[2][1]

    return (text, pos), (heads, rels), (transitions, relations_in_order), text_mappings


def get_data_loader(fname, transitions_file, transition_system, tokenizer, batch_size, shuffle):
    #print(transitions_file)
    dataset = SyntaxDataset(fname, transitions_file, transition_system, tokenizer)
    #print("action distribution {}".format(dataset.act_counts))
    #tot = 0
    #for item in dataset.act_counts.values(): tot+=item
    #print("Total {}".format(tot))
    #print("% {}".format([a/tot for a in dataset.act_counts.values()]))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=generate_batch)


def get_data_loaders(data_path, language, batch_size, batch_size_eval, transitions=None, transition_system=None,
                     bert_model=None):
    src_path = path.join(data_path, constants.UD_PATH_PROCESSED, language)

    (fname_train, fname_dev, fname_test) = get_ud_fname(src_path)
    transitions_train, transitions_dev, transitions_test = None, None, None

    if transitions is not None:
        (transitions_train, transitions_dev, transitions_test) = get_oracle_actions(src_path, transitions)
    vocabs = load_vocabs(src_path)
    #embeddings = load_embeddings(src_path)
    if language == "en":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif language == "de":
        tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
    elif language == "cs":
        # tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/bert-base-bg-cs-pl-ru-cased")
    elif language == "eu":
        # Basque
        tokenizer = AutoTokenizer.from_pretrained("ixa-ehu/berteus-base-cased")
    elif language == "hu":
        tokenizer = AutoTokenizer.from_pretrained("SZTAKI-HLT/hubert-base-cc")
    elif language == "tr":
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    #tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    trainloader = get_data_loader(fname_train, transitions_train, transition_system, tokenizer, batch_size,
                                  shuffle=True)
    devloader = get_data_loader(fname_dev, transitions_dev, transition_system, tokenizer, batch_size_eval,
                                shuffle=False)
    testloader = get_data_loader(fname_test, transitions_test, transition_system, tokenizer, batch_size_eval,
                                 shuffle=False)

    return trainloader, devloader, testloader,vocabs
