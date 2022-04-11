# -*- coding: utf-8 -*-
"""
    Word-based RNN model for text classification
    @author: Thanh Vu <thanh.vu@csiro.au>
    @date created: 07/03/2019
    @date last modified: 19/08/2020
"""

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.models.attentions.util import *
from src.models.embeddings.util import *
from src.data_helpers.vocab import Vocab, device


class RNN(nn.Module):
    def __init__(self, vocab: Vocab,
                 args):
        """

        :param vocab: Vocab
            The vocabulary normally built on the training data
        :param args:
            mode: rand/static/non-static/multichannel the mode of initialising embeddings
            hidden_size: (int) The size of the hidden layer
            n_layers: (int) The number of hidden layers
            bidirectional: (bool) Whether or not using bidirectional connection
            dropout: (float) The dropout parameter for RNN (GRU or LSTM)
        """

        super(RNN, self).__init__()
        self.vocab_size = vocab.n_words()
        self.vocab = vocab
        self.args = args
        self.use_last_hidden_state = args.use_last_hidden_state
        self.mode = args.mode
        self.n_layers = args.n_layers
        self.hidden_size = args.hidden_size
        self.bidirectional = bool(args.bidirectional)
        self.n_directions = int(self.bidirectional) + 1
        self.attention_mode = args.attention_mode
        self.output_size = self.hidden_size * self.n_directions
        self.rnn_model = args.rnn_model

        self.dropout = args.dropout
        self.embedding = init_embedding_layer(args, vocab)

        if self.rnn_model.lower() == "gru":
            self.rnn = nn.GRU(self.embedding.output_size, self.hidden_size, num_layers=self.n_layers,
                              bidirectional=self.bidirectional, dropout=self.dropout if self.n_layers > 1 else 0)
        else:
            self.rnn = nn.LSTM(self.embedding.output_size, self.hidden_size, num_layers=self.n_layers,
                               bidirectional=self.bidirectional, dropout=self.dropout if self.n_layers > 1 else 0)

        self.use_dropout = args.dropout > 0
        self.dropout = nn.Dropout(args.dropout)
        init_attention_layer(self)

    def init_hidden(self,
                    batch_size: int = 1) -> Variable:
        """
        Initialise the hidden layer
        :param batch_size: int
            The batch size
        :return: Variable
            The initialised hidden layer
        """
        # [(n_layers x n_directions) x batch_size x hidden_size]
        h = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to(device)
        c = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to(device)
        if self.rnn_model.lower() == "gru":
            return h
        return h, c

    def forward(self,
                batch_data: torch.LongTensor,
                lengths: torch.LongTensor) -> tuple:
        """

        :param batch_data: torch.LongTensor
            [batch_size x max_seq_len]
        :param lengths: torch.LongTensor
            [batch_size x 1]
        :return: output [batch_size x n_classes]
            attention_weights
        """

        batch_size = batch_data.size()[0]
        hidden = self.init_hidden(batch_size)

        embeds = self.embedding(batch_data)

        if self.use_dropout:
            embeds = self.dropout(embeds)

        self.rnn.flatten_parameters()
        embeds = pack_padded_sequence(embeds, lengths.cpu(), batch_first=True)

        rnn_output, hidden = self.rnn(embeds, hidden)
        if self.rnn_model.lower() == "lstm":
            hidden = hidden[0]

        rnn_output = pad_packed_sequence(rnn_output)[0]

        rnn_output = rnn_output.permute(1, 0, 2)

        weighted_outputs, attention_weights = perform_attention(self, rnn_output,
                                                                self.get_last_hidden_output(hidden)
                                                                )
        return weighted_outputs, attention_weights

    def get_last_hidden_output(self, hidden):
        if self.bidirectional:
            hidden_forward = hidden[-1]
            hidden_backward = hidden[0]
            if len(hidden_backward.shape) > 2:
                hidden_forward = hidden_forward.squeeze(0)
                hidden_backward = hidden_backward.squeeze(0)
            last_rnn_output = torch.cat((hidden_forward, hidden_backward), 1)
        else:

            last_rnn_output = hidden[-1]
            if len(hidden.shape) > 2:
                last_rnn_output = last_rnn_output.squeeze(0)

        return last_rnn_output
