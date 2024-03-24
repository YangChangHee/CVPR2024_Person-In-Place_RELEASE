import torch
import torch.nn as nn
import torch.nn.functional as F

coco_skeleton = ( (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 6), (11, 12),(5,11),(6,12),(3,5),(4,6) )
coco_joint_num = 17
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

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        batch_size,_,_=h.shape
        Wh = torch.bmm(h, self.W.expand(batch_size,self.in_features,self.out_features)) # h.shape: (N, in_features), Wh.shape: (N, out_features)

        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        batch_size,_,_=Wh.shape
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.bmm(Wh, self.a[:self.out_features, :].expand(batch_size,self.out_features,1))
        Wh2 = torch.bmm(Wh, self.a[self.out_features:, :].expand(batch_size,self.out_features,1))
        # broadcast add
        e = Wh1 + Wh2
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.adj_matrix=self.make_adj_matrix(coco_joint_num,coco_skeleton)

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid* 8, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x):
        batch_size,_,_=x.shape
        adj=self.adj_matrix.expand(batch_size,coco_joint_num,coco_joint_num).cuda()
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x.reshape(batch_size,34)


    def make_adj_matrix(self,skt_num,skt_index):
        zero_mat=torch.zeros((skt_num,skt_num))
        for i in range(skt_num):
            zero_mat[i,i]=1
        for i in skt_index:
            zero_mat[i[0],i[1]]=1
            zero_mat[i[1],i[0]]=1
        return zero_mat