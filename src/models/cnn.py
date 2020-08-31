# -*- coding: utf-8 -*-
"""
    Word-based CNN model for text classification from the paper
    Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014)
    https://arxiv.org/abs/1408.5882
    Add the support of char features learned from RNN or CNN to enrich word embeddings

    @author: Thanh Vu <thanh.vu@csiro.au>
    @date created: 07/03/2019
    @date last modified: 19/08/2020

"""
from src.models.tcn import *
from src.data_helpers.vocab import Vocab
from src.models.attentions.util import *
from src.models.embeddings.util import *
from math import floor


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class WordCNN(nn.Module):

    def __init__(self,
                 vocab: Vocab,
                 args):
        """
        :param vocab: Vocab
            The vocabulary built using the training data
        :param args:
            mode: (str) the mode of initialising embeddings
            kernel_sizes: (list) the kernel sizes
            out_channels: int the number of channels
            dropout: float
        """
        super(WordCNN, self).__init__()
        self.vocab = vocab
        self.args = args
        self.vocab_size = vocab.n_words()

        self.kernel_size = args.kernel_size
        self.out_channels = args.out_channels
        self.dropout = args.dropout
        self.attention_mode = args.attention_mode

        self.output_size = self.out_channels

        self.embedding = init_embedding_layer(args, vocab)

        # build the model architecture
        self.cnn_model = args.cnn_model

        self.flatten = Flatten()

        if self.cnn_model == "CONV1D":
            self.conv = nn.Conv1d(in_channels=self.embedding.output_size, out_channels=self.out_channels,
                                  kernel_size=self.kernel_size, padding=int(floor(self.kernel_size/2)))

        elif self.cnn_model == "TCN":
            self.conv = TemporalConvNet(self.embedding.output_size, num_channels=[self.out_channels] * args.n_layers,
                                        kernel_size=self.kernel_size, dropout=self.dropout)
        else:
            raise NotImplementedError

        self.use_dropout = self.dropout > 0
        self.dropout = nn.Dropout(self.dropout)
        init_attention_layer(self)

    def forward(self, batch_data: torch.LongTensor,
                lengths: torch.LongTensor):
        """
        :param batch_data: torch.LongTensor
            [batch_size x max_seq_len]
        :param lengths: torch.LongTensor
            [batch_size x 1]
        :return: [batch_size x n_classes]
        """

        embeds = self.embedding(batch_data)
        if type(embeds) == tuple:
            embeds = embeds[0]

        if embeds.size(1) < self.kernel_size:
            embeds = self.pad_input(embeds)

        if self.use_dropout:
            embeds = self.dropout(embeds)

        # [batch_size x max_seq_size x embedding_size] --> [batch_size x embedding_size x max_seq_size]
        embeds = embeds.transpose(1, 2)   # ~ embeds.permute(0, 1, 2)

        feature_map = self.conv(embeds)
        feature_map = F.relu(feature_map)

        weighted_outputs, attention_weights = perform_attention(self, feature_map.permute(0, 2, 1),
                                                                self.get_last_hidden_output(feature_map))
        return weighted_outputs, attention_weights

    def pad_input(self, input_batch):
        sizes = input_batch.size()
        padded_input = torch.zeros(sizes[0], self.kernel_size, sizes[2]).to(self.vocab.device)
        padded_input[:, : input_batch.size(1), :] = input_batch
        return padded_input

    def get_last_hidden_output(self, feature_map):
        features = F.max_pool1d(feature_map, feature_map.size(-1)).squeeze(-1)
        return features





