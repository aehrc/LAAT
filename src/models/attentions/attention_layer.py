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


from torch.nn.init import xavier_uniform_ as xavier_uniform
import numpy as np
# hicu decoder
class Decoder(nn.Module):
    """
    HiCu Decoder: knowledge transfer initialization and hyperbolic embedding correction
    """
    def __init__(self, args, input_size, Y, dicts):
        super(Decoder, self).__init__()

        self.dicts = dicts
        self.args = args

        if not args.disable_attention_linear:
            self.d_a = args.d_a
        else:
            self.d_a = input_size

        self.decoder_dict = nn.ModuleDict()
        for i in range(len(Y)):
            y = Y[i]
            self.decoder_dict[str(i) + '_' + '0'] = nn.Linear(self.d_a, y)
            self.decoder_dict[str(i) + '_' + '1'] = nn.Linear(self.d_a, y)
            xavier_uniform(self.decoder_dict[str(i) + '_' + '0'].weight)
            xavier_uniform(self.decoder_dict[str(i) + '_' + '1'].weight)
        
        self.use_hyperbolic =  args.decoder.find("Hyperbolic") != -1
        if self.use_hyperbolic:
            self.cat_hyperbolic = args.cat_hyperbolic
            if not self.cat_hyperbolic:
                self.hyperbolic_fc_dict = nn.ModuleDict()
                for i in range(len(Y)):
                    self.hyperbolic_fc_dict[str(i)] = nn.Linear(args.hyperbolic_dim, self.d_a)
            else:
                self.query_fc_dict = nn.ModuleDict()
                for i in range(len(Y)):
                    self.query_fc_dict[str(i)] = nn.Linear(self.d_a + args.hyperbolic_dim, self.d_a)
            
            # build hyperbolic embedding matrix
            self.hyperbolic_emb_dict = {}
            for i in range(len(Y)):
                self.hyperbolic_emb_dict[i] = np.zeros((Y[i], args.hyperbolic_dim))
                for idx, code in dicts.index2label[i].items():
                    self.hyperbolic_emb_dict[i][idx, :] = np.copy(dicts.poincare_embeddings.get_vector(code))
                self.register_buffer(name='hb_emb_' + str(i), tensor=torch.tensor(self.hyperbolic_emb_dict[i], dtype=torch.float32))

        self.cur_depth = 5 - args.depth
        self.is_init = False
        self.change_depth(self.cur_depth)

        if not args.disable_attention_linear:
            self.W = nn.Linear(input_size, self.d_a)
            xavier_uniform(self.W.weight)

        # if args.loss == 'BCE':
        #     self.loss_function = nn.BCEWithLogitsLoss()
        # elif args.loss == 'ASL':
        #     asl_config = [float(c) for c in args.asl_config.split(',')]
        #     self.loss_function = AsymmetricLoss(gamma_neg=asl_config[0], gamma_pos=asl_config[1],
        #                                         clip=asl_config[2], reduction=args.asl_reduction)
        # elif args.loss == 'ASLO':
        #     asl_config = [float(c) for c in args.asl_config.split(',')]
        #     self.loss_function = AsymmetricLossOptimized(gamma_neg=asl_config[0], gamma_pos=asl_config[1],
        #                                                  clip=asl_config[2], reduction=args.asl_reduction)
    
    def change_depth(self, depth=0):
        if self.is_init:
            # copy previous attention weights to current attention network based on ICD hierarchy
            ind2c = self.dicts.index2label
            c2ind = self.dicts.label2index
            hierarchy_dist = self.dicts.hierarchy
            for i, code in ind2c[depth].items():
                tree = hierarchy_dist[depth][code]
                pre_idx = c2ind[depth - 1][tree[depth - 1]]

                self.decoder_dict[str(depth) + '_' + '0'].weight.data[i, :] = self.decoder_dict[str(depth - 1) + '_' + '0'].weight.data[pre_idx, :].clone()
                self.decoder_dict[str(depth) + '_' + '1'].weight.data[i, :] = self.decoder_dict[str(depth - 1) + '_' + '1'].weight.data[pre_idx, :].clone()

        if not self.is_init:
            self.is_init = True

        self.cur_depth = depth
        
    def forward(self, x):
        if not self.args.disable_attention_linear:
            z = torch.tanh(self.W(x))
        else:
            z = x
        # attention
        if self.use_hyperbolic:
            if not self.cat_hyperbolic:
                query = self.decoder_dict[str(self.cur_depth) + '_' + '0'].weight + self.hyperbolic_fc_dict[str(self.cur_depth)](self._buffers['hb_emb_' + str(self.cur_depth)])
            else:
                query = torch.cat([self.decoder_dict[str(self.cur_depth) + '_' + '0'].weight, self._buffers['hb_emb_' + str(self.cur_depth)]], dim=1)
                query = self.query_fc_dict[str(self.cur_depth)](query)
        else:
            query = self.decoder_dict[str(self.cur_depth) + '_' + '0'].weight

        alpha = F.softmax(query.matmul(z.transpose(1, 2)), dim=2)
        m = alpha.matmul(z)

        y = self.decoder_dict[str(self.cur_depth) + '_' + '1'].weight.mul(m).sum(dim=2).add(self.decoder_dict[str(self.cur_depth) + '_' + '1'].bias)

        # loss = self.loss_function(y, target)
        
        return y, alpha