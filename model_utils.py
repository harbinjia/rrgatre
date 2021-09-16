import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False):
        super(GraphConvolution, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(0.5)
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        result = []
        for i in range(len(adj)):
            support = torch.matmul(input[i], self.weight)
            output = torch.matmul(adj[i], support)
            # print(output.shape)
            if self.bias is not None:
                output =  output + self.bias
            activation = torch.nn.LeakyReLU()
            result.append(activation(self.dropout(output)))
        # result = np.array(result)
        return result


class GraphAttentionLayer(nn.Module):
  """
  Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
  """

  def __init__(self, in_features, out_features, dropout, alpha, concat=True):
    super(GraphAttentionLayer, self).__init__()
    self.dropout = dropout
    self.in_features = in_features
    self.out_features = out_features
    self.alpha = alpha
    self.concat = concat

    self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
    nn.init.xavier_uniform_(self.W.data, gain=1.414)
    self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
    nn.init.xavier_uniform_(self.a.data, gain=1.414)

    self.leakyrelu = nn.LeakyReLU(self.alpha)

  def forward(self, input, adj):
    h = torch.mm(input, self.W)
    N = h.size()[0]

    a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
    e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

    zero_vec = -9e15 * torch.ones_like(e)
    attention = torch.where(adj > 0, e, zero_vec)
    attention = F.softmax(attention, dim=1)
    attention = F.dropout(attention, self.dropout, training=self.training)
    h_prime = torch.matmul(attention, h)

    if self.concat:
        return F.elu(h_prime)
    else:
        return h_prime

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
  def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
    """Dense version of GAT."""
    super(GAT, self).__init__()
    self.dropout = dropout

    self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
    for i, attention in enumerate(self.attentions):
        self.add_module('attention_{}'.format(i), attention)

    self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

  def forward(self, x, adj):
    result = []
    for i in range(len(adj)):
        x_tensor = x[i]
        adj_tensor = adj[i]
        x_tensor = F.dropout(x_tensor, self.dropout, training=self.training)
        x_tensor = torch.cat([att(x_tensor, adj_tensor) for att in self.attentions], dim=1)
        x_tensor = F.dropout(x_tensor, self.dropout, training=self.training)
        x_tensor = F.leaky_relu(self.out_att(x_tensor, adj_tensor))
        result.append(x_tensor)
    return result

class MultiNonLinearClassifier(nn.Module):
    """
    linear->ReLU->droupout->linear 
    return features_output
    """
    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        features_output = self.hidden2tag(features_tmp)
        return features_output

def masked_avgpool(sent, mask):
    mask_ = mask.masked_fill(mask == 0, -1e9).float()
    score = torch.softmax(mask_, -1)
    return torch.matmul(score.unsqueeze(1), sent).squeeze(1)

def gen_adj(A):
    # print(A)
    # print(A.sum(1))
    A = A.float()
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(D, A), D)
    return adj
