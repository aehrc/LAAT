# -*- coding: utf-8 -*-
"""
    This is to create the vocabularies which are used to convert the text data into tensor format
    Author: Thanh Vu <thanh.vu@csiro.au>
    Date created: 01/03/2019
    Date last modified: 19/03/2019
"""
import os
import torch
from collections import Counter
import numpy as np
from gensim.models.fasttext import FastText
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Vocab(object):
    def __init__(self,
                 training_data: list,
                 training_labels: list,
                 min_word_frequency: int = -1,
                 max_vocab_size: int = -1,
                 word_embedding_mode: str = "word2vec",
                 word_embedding_file: str = None,
                 use_gpu: bool = True
                 ):
        """

        :param training_data:
        :param min_word_frequency:
        :param max_vocab_size:
        :param word_embedding_mode: str
            "word2vec": using word embeddings like word2vec or glove
            "fasttext": using subword embeddings like fastText
        :param word_embedding_file:
        :param use_gpu:
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.word_embedding_mode = word_embedding_mode
        self.word_embedding_file = word_embedding_file
        self.word_embedding_size = None
        self.word_embeddings = None

        self.training_data = training_data

        self.PAD_TOKEN = '_PAD'
        self.UNK_TOKEN = '_UNK'
        self.word2index = None
        self.index2word = None

        self.label2index = []
        self.index2label = []

        self.vocab_words = [self.PAD_TOKEN, self.UNK_TOKEN]
        self.all_labels = []

        self.min_word_frequency = min_word_frequency
        self.max_vocab_size = max_vocab_size

        self.logger = None
        self.update_labels(training_labels)

    def prepare_vocab(self):
        self._build_vocab()

        # load pretrain word embeddings
        if self.word_embedding_file is not None:
            self.word_embeddings = torch.FloatTensor(self._load_embeddings())

    def index_of_word(self,
                      word: str) -> int:
        try:
            return self.word2index[word]
        except:
            return self.word2index[self.UNK_TOKEN]

    def index_of_label(self,
                       label: str, level: int) -> int:
        try:
            return self.label2index[level][label]
        except:
            return 0

    def _build_vocab(self):
        all_words = []

        for text, labels, _id in self.training_data:
            filtered_words = text.split()
            all_words.extend(filtered_words)

        counter = Counter(all_words)
        if self.max_vocab_size > 0:
            counter = {word: freq for word, freq in counter.most_common(self.max_vocab_size)}
        if self.min_word_frequency > 0:
            counter = {word: freq for word, freq in counter.items() if freq >= self.min_word_frequency}

        self.vocab_words += list(sorted(counter.keys()))

        self.word2index = {word: idx for idx, word in enumerate(self.vocab_words)}
        self.index2word = {idx: word for idx, word in enumerate(self.vocab_words)}

    def update_labels(self, labels):
        self.all_labels = []
        self.index2label = []
        self.label2index = []
        for level_labels in labels:
            all_labels = list(sorted(level_labels))
            self.label2index.append({label: idx for idx, label in enumerate(all_labels)})
            self.index2label.append({idx: label for idx, label in enumerate(all_labels)})
            self.all_labels.append(all_labels)

    def update_hierarchy(self, hierarchy, poincare=None):
        self.hierarchy = hierarchy
        if poincare is not None:
            self.poincare_embeddings = poincare

    def _load_embeddings(self):
        if self.word_embedding_file is None:
            return None

        gensim_format = False
        if self.word_embedding_file.endswith(".model") or self.word_embedding_file.endswith(".bin"):
            gensim_format = True

        if not gensim_format:
            if self.word_embedding_mode.lower() == "fasttext":
                return self._load_subword_embeddings()
            return self._load_word_embeddings()
        else:
            return self._load_gensim_format_embeddings()

    def _load_word_embeddings(self):
        embeddings = None
        embedding_size = None

        count = 0
        if not os.path.exists(self.word_embedding_file):
            raise Exception("{} is not found!".format(self.word_embedding_file))

        for line in open(self.word_embedding_file, "rt"):
            if count >= 0:

                split = line.rstrip().split(" ")
                word = split[0]
                vector = np.array([float(num) for num in split[1:]]).astype(np.float32)
                if len(vector) > 0:
                    if embedding_size is None:
                        embedding_size = len(vector)

                        unknown_vec = np.random.uniform(-0.25, 0.25, embedding_size)
                        embeddings = [unknown_vec] * (self.n_words())

                        embeddings[0] = np.zeros(embedding_size)
                    if word in self.word2index:
                        embeddings[self.word2index[word]] = vector
            count += 1

        self.word_embedding_size = len(embeddings[0])
        embeddings = np.array(embeddings, dtype=np.float32)

        return embeddings

    def _load_subword_embeddings(self):
        if not os.path.exists(self.word_embedding_file):
            raise Exception("{} is not found!".format(self.word_embedding_file))

        model = FastText.load_fasttext_format(self.word_embedding_file)
        embedding_size = model["and"].size
        unknown_vec = np.random.uniform(-0.25, 0.25, embedding_size)

        embeddings = [unknown_vec] * (self.n_words())
        embeddings[0] = np.zeros(embedding_size)
        for word in self.word2index:
            try:
                embeddings[self.word2index[word]] = model[word]
            except:
                # self.word2index[word] = self.word2index[self.UNK_TOKEN]
                pass

        self.word_embedding_size = len(embeddings[0])
        embeddings = np.array(embeddings, dtype=np.float32)

        return embeddings

    def _load_gensim_format_embeddings(self):
        if not os.path.exists(self.word_embedding_file):
            raise Exception("{} is not found!".format(self.word_embedding_file))

        if self.word_embedding_mode.lower() == "fasttext":

            if self.word_embedding_file.endswith(".model"):
                model = FastText.load(self.word_embedding_file)
            else:
                model = FastText.load_fasttext_format(self.word_embedding_file)

        elif self.word_embedding_file.endswith(".bin"):
            model = KeyedVectors.load_word2vec_format(self.word_embedding_file, binary=True)
        else:
            model = Word2Vec.load(self.word_embedding_file)

        embedding_size = model.wv["and"].size
        unknown_vec = np.random.uniform(-0.25, 0.25, embedding_size)

        embeddings = [unknown_vec] * (self.n_words())
        embeddings[0] = np.zeros(embedding_size)
        for word in self.word2index:
            try:
                embeddings[self.word2index[word]] = model.wv[word]
            except:
                # self.word2index[word] = self.word2index[self.UNK_TOKEN]
                pass

        self.word_embedding_size = len(embeddings[0])
        embeddings = np.array(embeddings, dtype=np.float32)

        return embeddings

    def n_words(self):
        return len(self.vocab_words)

    def n_labels(self, level):
        return len(self.all_labels[level])

    def n_level(self):
        return len(self.all_labels)

    def all_n_labels(self):
        output = []
        for level in range(len(self.all_labels)):
            output.append(len(self.all_labels[level]))
        return output
