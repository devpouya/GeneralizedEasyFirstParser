
class Vocab:
    # pylint: disable=invalid-name,too-many-instance-attributes
    ROOT = '<ROOT>'
    SPECIAL_TOKENS = ('<PAD>', '<ROOT>', '<UNK>')

    def __init__(self, min_count=None):
        self._counts = {}
        self._pretrained = set([])
        self.min_count = min_count
        self.size = 3

    def idx(self, token):
        if token not in self._vocab:
            return self._vocab['<UNK>']
        return self._vocab[token]

    def count_up(self, token):
        self._counts[token] = self._counts.get(token, 0) + 1

    def add_pretrained(self, token):
        self._pretrained |= set([token])

    def process_vocab(self):
        self._counts_raw = self._counts
        if self.min_count:
            self._counts = {k: count for k, count in self._counts.items()
                            if count > self.min_count}

        words = [x[0] for x in sorted(self._counts.items(), key=lambda x: x[1], reverse=True)]
        pretrained = [x for x in self._pretrained if x not in self._counts]
        types = list(self.SPECIAL_TOKENS) + words + pretrained

        self._vocab = {x: i for i, x in enumerate(types)}
        self._idx2word = {idx: word for word, idx in self._vocab.items()}
        self.ROOT_IDX = self._vocab[self.ROOT]
        self.size = len(self._vocab)

    def items(self):
        for word, idx in self._vocab.items():
            yield word, idx
