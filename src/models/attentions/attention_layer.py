"""
    This class is to implement the attention layer which supports hard attention, self-structured attention
        and self attention

    @author: Thanh Vu <thanh.vu@csiro.au>
    @date created: 20/03/2019
    @date last modified: 19/08/2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):

    def __init__(self,
                 args,
                 size: int,
                 level_projection_size: int = 0,
                 n_labels=None,
                 n_level: int = 1
                 ):
        """
        The init function
        :param args: the input parameters from commandline
        :param size: the input size of the layer, it is normally the output size of other DNN models,
            such as CNN, RNN
        """
        super(AttentionLayer, self).__init__()
        self.attention_mode = args.attention_mode

        self.size = size
        # For self-attention: d_a and r are the dimension of the dense layer and the number of attention-hops
        # d_a is the output size of the first linear layer
        self.d_a = args.d_a if args.d_a > 0 else self.size

        # r is the number of attention heads

        self.n_labels = n_labels
        self.n_level = n_level
        self.r = [args.r if args.r > 0 else n_labels[label_lvl] for label_lvl in range(n_level)]

        self.level_projection_size = level_projection_size

        self.linear = nn.Linear(self.size, self.size, bias=False)
        if self.attention_mode == "hard":
            self.first_linears = nn.ModuleList([nn.Linear(self.size, self.size, bias=True) for _ in range(self.n_level)])
            self.second_linears = nn.ModuleList([nn.Linear(self.size, 1, bias=False) for _ in range(self.n_level)])

        elif self.attention_mode == "self":

            self.first_linears = nn.ModuleList([nn.Linear(self.size, self.d_a, bias=False) for _ in range(self.n_level)])
            self.second_linears = nn.ModuleList([nn.Linear(self.d_a, self.r[label_lvl], bias=False) for label_lvl in range(self.n_level)])

        elif self.attention_mode == "label" or self.attention_mode == "caml":
            if self.attention_mode == "caml":
                self.d_a = self.size

            self.first_linears = nn.ModuleList([nn.Linear(self.size, self.d_a, bias=False) for _ in range(self.n_level)])
            self.second_linears = nn.ModuleList([nn.Linear(self.d_a, self.n_labels[label_lvl], bias=False) for label_lvl in range(self.n_level)])
            self.third_linears = nn.ModuleList([nn.Linear(self.size +
                                               (self.level_projection_size if label_lvl > 0 else 0),
                                               self.n_labels[label_lvl], bias=True) for label_lvl in range(self.n_level)])
        else:
            raise NotImplementedError
        self._init_weights(mean=0.0, std=0.03)

    def _init_weights(self, mean=0.0, std=0.03) -> None:
        """
        Initialise the weights
        :param mean:
        :param std:
        :return: None
        """
        for first_linear in self.first_linears:
            torch.nn.init.normal(first_linear.weight, mean, std)
            if first_linear.bias is not None:
                first_linear.bias.data.fill_(0)

        for linear in self.second_linears:
            torch.nn.init.normal(linear.weight, mean, std)
            if linear.bias is not None:
                linear.bias.data.fill_(0)
        if self.attention_mode == "label" or self.attention_mode == "caml":
            for linear in self.third_linears:
                torch.nn.init.normal(linear.weight, mean, std)

    def forward(self, x, previous_level_projection=None, label_level=0):
        """
        :param x: [batch_size x max_len x dim (i.e., self.size)]

        :param previous_level_projection: the embeddings for the previous level output
        :param label_level: the current label level
        :return:
            Weighted average output: [batch_size x dim (i.e., self.size)]
            Attention weights
        """
        if self.attention_mode == "caml":
            weights = F.tanh(x)
        else:
            weights = F.tanh(self.first_linears[label_level](x))

        att_weights = self.second_linears[label_level](weights)
        att_weights = F.softmax(att_weights, 1).transpose(1, 2)
        if len(att_weights.size()) != len(x.size()):
            att_weights = att_weights.squeeze()
        weighted_output = att_weights @ x

        if self.attention_mode == "label" or self.attention_mode == "caml":
            batch_size = weighted_output.size(0)

            if previous_level_projection is not None:
                temp = [weighted_output,
                        previous_level_projection.repeat(1, self.n_labels[label_level]).view(batch_size, self.n_labels[label_level], -1)]
                weighted_output = torch.cat(temp, dim=2)

            weighted_output = self.third_linears[label_level].weight.mul(weighted_output).sum(dim=2).add(
                self.third_linears[label_level].bias)

        else:
            weighted_output = torch.sum(weighted_output, 1) / self.r[label_level]
            if previous_level_projection is not None:
                temp = [weighted_output, previous_level_projection]
                weighted_output = torch.cat(temp, dim=1)

        return weighted_output, att_weights

    # Using when use_regularisation = True
    @staticmethod
    def l2_matrix_norm(m):
        """
        Frobenius norm calculation
        :param m: {Variable} ||AAT - I||
        :return: regularized value
        """
        return torch.sum(torch.sum(torch.sum(m ** 2, 1), 1) ** 0.5)
