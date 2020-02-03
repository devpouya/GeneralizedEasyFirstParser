# dep-parser

[![CircleCI](https://circleci.com/gh/tpimentelms/dep-parser.svg?style=svg&circle-token=e282f0a5450b745a18358ad5b01ae8b1f0f9b02d)](https://circleci.com/gh/tpimentelms/dep-parser)

Code to train a dependency parser model.


## Install

To install dependencies run:
```bash
$ conda env create -f environment.yml
```

And then install the appropriate version of pytorch, for example:
```bash
$ conda install pytorch torchvision cpuonly -c pytorch
$ # conda install pytorch==1.0.0 torchvision==0.2.1 cuda80 -c pytorch
```

## Data

Get Universal Dependencies data in [https://universaldependencies.org/#download].
```bash
$ make get_ud
```

## Running the code

First preprocess the data for the language you are using:
```bash
$ python src/h01_data/process.py --language <language-code> --glove-file <glove-vectors-filename>
```
Where language is the ISO 639-1 code for the language, and glove file is the path to a txt file containing one word and its embedding per line.
GloVe embeddings for wikipedia can be trained with [this repository](https://github.com/tpimentelms/GloVe).

Then, train the model with the command:
```bash
$ python src/h02_learn/train.py --language <language-code>
```
