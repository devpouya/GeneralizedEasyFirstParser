import os
import pathlib
import json
import pickle
import numpy as np
import torch


def get_filenames(filepath):
    filenames = [os.path.join(filepath, f)
                 for f in os.listdir(filepath)
                 if os.path.isfile(os.path.join(filepath, f))]
    return sorted(filenames)


def append_json(fname, data):
    with open(fname, 'a') as f:
        line = json.dumps(data, ensure_ascii=False)
        f.write("%s\n" % line)


def write_data(filename, embeddings):
    with open(filename, "wb") as f:
        pickle.dump(embeddings, f)


def read_data(filename):
    with open(filename, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


def read_data_if_exists(filename):
    try:
        return read_data(filename)
    except FileNotFoundError:
        return {}


def config(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def remove_if_exists(fname):
    try:
        os.remove(fname)
    except FileNotFoundError:
        pass


def mkdir(folder):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
