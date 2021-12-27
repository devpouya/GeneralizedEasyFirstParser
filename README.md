# Generalized Easy First Parser

[![CircleCI](https://circleci.com/gh/tpimentelms/dep-parser.svg?style=svg&circle-token=e282f0a5450b745a18358ad5b01ae8b1f0f9b02d)](https://circleci.com/gh/tpimentelms/dep-parser)

This repository has been originally forked from tpimentelms/dep-parser but has changed a lot since then. 

It implements a generalized easy first parser currently able to parse according to the MH4 transition system in both a shift-reduce and free order.
It is able to train both a multilingual and a monoligual parser.

## Install

Install the dependencies youself :P

## Data

Get Universal Dependencies data in [https://universaldependencies.org/#download].
```bash
$ make get_ud
```

## Running the code

First preprocess the data for the language you are using:
```bash
$ python src/h01_data/process.py --language <language-code> --easy-first <True/False>
```
Where language is the ISO 639-1 code for the language. You can further specify "multilingual" to prepare a multilingual dataset. easy-first specifies whether the oracle will use an easy-first or shift-reduce order to derive the tree.


Then, train the model with the command:
```bash
$ python src/h02_learn/train.py --language <language-code>  --mode <easy-first/shift-reduce>
```

This code, will by default look for data in the `./data` path. To change it (either during data preprocessing or training) use the argument `--data-path <data-path>`.


