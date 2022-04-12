# -*- coding: utf-8 -*-
"""
    This provides the functions to load the data for training and testing the model (e.g., batch)
    Author: Thanh Vu <thanh.vu@csiro.au>
    Date created: 01/03/2019
    Date last modified: 19/08/2020
"""
import torch
from torch.utils.data import DataLoader, Dataset
from src.util.preprocessing import SENTENCE_SEPARATOR, RECORD_SEPARATOR
from torch.nn.utils.rnn import pad_sequence
import random
from tqdm import tqdm
import numpy as np


class TextDataset(Dataset):

    def __init__(self,
                 text_data: list,
                 vocab,
                 sort: bool = False,
                 max_seq_length: int = -1,
                 min_seq_length: int = -1,
                 multilabel: bool = False):
        super(TextDataset, self).__init__()
        self.vocab = vocab
        self.multilabel = multilabel
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.PAD_ID = self.vocab.index_of_word(self.vocab.PAD_TOKEN)
        indexed_data = []
        self.n_instances = len(text_data)
        self.n_total_tokens = 0

        n_label_level = len(text_data[0][1])
        self.label_count = [dict() for _ in range(n_label_level)]
        self.labels = [set() for _ in range(n_label_level)]
        for text, labels, _id in tqdm(text_data, unit="samples", desc="Processing data"):
            label_list = [[] for _ in range(n_label_level)]
            if type(labels) == list:
                for label_lvl in range(len(labels)):
                    for label in labels[label_lvl]:
                        if label in self.vocab.label2index[label_lvl]:

                            label = self.vocab.index_of_label(label, label_lvl)
                            if label not in self.label_count[label_lvl]:
                                self.label_count[label_lvl][label] = 1
                            else:
                                self.label_count[label_lvl][label] += 1
                            self.labels[label_lvl].add(label)
                            label_list[label_lvl].append(label)
                        else:
                            continue

            if len(label_list) == 0:
                continue

            is_skipped = False
            for level_label in label_list:
                if len(level_label) == 0:
                    is_skipped = True
                    break
            if is_skipped:
                continue

            word_seq = []

            records = text.split(RECORD_SEPARATOR)
            should_break = False
            for i in range(len(records)):
                record = records[i]
                sentences = record.split(SENTENCE_SEPARATOR)

                for sentence in sentences:
                    sent_words = sentence.strip().split()
                    if len(sent_words) == 0:
                        continue
                    # self.n_total_tokens += len(sent_words)
                    for word in sent_words:
                        word_idx = vocab.index_of_word(word)
                        word_seq.append(word_idx)
                        self.n_total_tokens += 1
                        if len(word_seq) >= self.max_seq_length > 0:
                            should_break = True
                            break
                    if should_break:
                        break

                if len(word_seq) < self.min_seq_length:
                    continue

                if should_break:
                    break

            # after processing all records
            if len(word_seq) > 0:
                indexed_data.append((word_seq, label_list, _id))

        if sort:
            self.indexed_data = sorted(indexed_data, key=lambda x: -len(x[0]))

        else:
            self.indexed_data = indexed_data
            self.shuffle_data()

        self.labels = sorted(list(self.labels))
        self.size = len(self.indexed_data)

    def shuffle_data(self):
        random.shuffle(self.indexed_data)

    def __getitem__(self, index):
        word_seq, label_list, _id = self.indexed_data[index]

        if len(word_seq) > self.max_seq_length > 0:
            word_seq = word_seq[:self.max_seq_length]

        if not self.multilabel:
            label_out = [None for _ in range(len(label_list))]
            for idx in range(len(label_list)):
                label_out[idx] = label_list[idx][0]
            return word_seq, label_out, _id
        else:
            all_one_hot_label_list = []
            for label_lvl in range(len(label_list)):
                one_hot_label_list = [0] * self.vocab.n_labels(label_lvl)
                for label in label_list[label_lvl]:
                    one_hot_label_list[label] = 1
                all_one_hot_label_list.append(np.asarray(one_hot_label_list).astype(np.int32))
            return word_seq, all_one_hot_label_list, _id

    def __len__(self):
        return len(self.indexed_data)



class HiCuTextDataset(Dataset):

    def __init__(self,
                 text_data: list,
                 vocab,
                 sort: bool = False,
                 max_seq_length: int = -1,
                 min_seq_length: int = -1,
                 multilabel: bool = False):
        super(HiCuTextDataset, self).__init__()
        self.vocab = vocab
        self.multilabel = multilabel
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.PAD_ID = self.vocab.index_of_word(self.vocab.PAD_TOKEN)
        indexed_data = []
        self.n_instances = len(text_data)
        self.n_total_tokens = 0

        n_label_level = vocab.n_level()
        self.label_count = [dict() for _ in range(n_label_level)]
        self.labels = [set() for _ in range(n_label_level)]
        for text, labels, _id in tqdm(text_data, unit="samples", desc="Processing data"):
            label_list = [[] for _ in range(n_label_level)]
            if type(labels) == list:
                for label in labels[0]:
                    if label in self.vocab.label2index[-1]:

                        label_path = vocab.hierarchy[4][label]

                        for level in range(5):
                            label_idx = self.vocab.index_of_label(label_path[level], level)

                            if label_idx not in self.label_count[level]:
                                self.label_count[level][label_idx] = 1
                            else:
                                self.label_count[level][label_idx] += 1
                            self.labels[level].add(label_idx)
                            label_list[level].append(label_idx)
                    else:
                        continue

            if len(label_list) == 0:
                continue

            is_skipped = False
            for level_label in label_list:
                if len(level_label) == 0:
                    is_skipped = True
                    break
            if is_skipped:
                continue

            word_seq = []

            records = text.split(RECORD_SEPARATOR)
            should_break = False
            for i in range(len(records)):
                record = records[i]
                sentences = record.split(SENTENCE_SEPARATOR)

                for sentence in sentences:
                    sent_words = sentence.strip().split()
                    if len(sent_words) == 0:
                        continue
                    # self.n_total_tokens += len(sent_words)
                    for word in sent_words:
                        word_idx = vocab.index_of_word(word)
                        word_seq.append(word_idx)
                        self.n_total_tokens += 1
                        if len(word_seq) >= self.max_seq_length > 0:
                            should_break = True
                            break
                    if should_break:
                        break

                if len(word_seq) < self.min_seq_length:
                    continue

                if should_break:
                    break

            # after processing all records
            if len(word_seq) > 0:
                indexed_data.append((word_seq, label_list, _id))

        if sort:
            self.indexed_data = sorted(indexed_data, key=lambda x: -len(x[0]))

        else:
            self.indexed_data = indexed_data
            self.shuffle_data()

        self.labels = sorted(list(self.labels))
        self.size = len(self.indexed_data)

    def shuffle_data(self):
        random.shuffle(self.indexed_data)

    def __getitem__(self, index):
        word_seq, label_list, _id = self.indexed_data[index]

        if len(word_seq) > self.max_seq_length > 0:
            word_seq = word_seq[:self.max_seq_length]

        if not self.multilabel:
            label_out = [None for _ in range(len(label_list))]
            for idx in range(len(label_list)):
                label_out[idx] = label_list[idx][0]
            return word_seq, label_out, _id
        else:
            all_one_hot_label_list = []
            for label_lvl in range(len(label_list)):
                one_hot_label_list = [0] * self.vocab.n_labels(label_lvl)
                for label in label_list[label_lvl]:
                    one_hot_label_list[label] = 1
                all_one_hot_label_list.append(np.asarray(one_hot_label_list).astype(np.int32))
            return word_seq, all_one_hot_label_list, _id

    def __len__(self):
        return len(self.indexed_data)



class TextDataLoader(DataLoader):
    def __init__(self,
                 vocab,
                 *args,
                 **kwargs):
        super(TextDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.PAD_ID = vocab.index_of_word(vocab.PAD_TOKEN)
        self.vocab = vocab

    def _collate_fn(self, batch):
        length_batch = []

        feature_batch = []
        label_batch = []

        id_batch = []
        multilabel = True
        for features, labels, _id in batch:
            if type(labels[0]) == int:
                multilabel = False

            feature_length = len(features)
            feature_batch.append(torch.LongTensor(features))

            length_batch.append(feature_length)
            label_batch.append(labels)
            id_batch.append(_id)

        feature_batch, label_batch, length_batch, id_batch = \
            self.sort_batch(feature_batch, label_batch, length_batch, id_batch)

        padded_batch = pad_sequence(feature_batch, batch_first=True)
        feature_batch = torch.LongTensor(padded_batch)

        label_batch = np.stack(label_batch, axis=0)
        norm_label_batch = []
        if not multilabel:
            for label_lvl in range(label_batch.shape[1]):
                norm_label_batch.append(torch.LongTensor(label_batch[:, label_lvl].tolist()))
        else:
            for label_lvl in range(label_batch.shape[1]):
                norm_label_batch.append(torch.FloatTensor(label_batch[:, label_lvl].tolist()))

        label_batch = norm_label_batch
        length_batch = torch.LongTensor(length_batch)
        return feature_batch, label_batch, length_batch, id_batch

    @staticmethod
    def sort_batch(features, labels, lengths, id_batch):
        sorted_indices = sorted(range(len(features)), key=lambda i: features[i].size(0), reverse=True)
        sorted_features = []
        sorted_labels = []
        sorted_lengths = []
        sorted_ids = []

        for index in sorted_indices:
            sorted_features.append(features[index])

            sorted_labels.append(labels[index])
            sorted_lengths.append(lengths[index])

            sorted_ids.append(id_batch[index])

        return sorted_features, sorted_labels, sorted_lengths, sorted_ids



