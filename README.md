# dep-parser

[![CircleCI](https://circleci.com/gh/tpimentelms/dep-parser.svg?style=svg&circle-token=ee712c838da05c2a7b580214784903b89b85ae44)](https://circleci.com/gh/tpimentelms/dep-parser)
Code to train a dependency parser model


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
$ # conda install pytorch==1.0.0 torchvision==0.2.1 cuda80 -c pytorch
```
