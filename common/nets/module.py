
import torch
import torch.nn as nn
import math
from einops import rearrange
from config import cfg
from torch.nn import functional as F
coco_skeleton = ( (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 6), (11, 12),(5,11),(6,12),(3,5),(4,6) )
coco_joint_num = 17

def make_linear_layers(feat_dims, relu_final=True, use_bn=False):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

class GraphConvBlock(nn.Module):
    def __init__(self, adj, dim_in, dim_out):
        super(GraphConvBlock, self).__init__()
        self.adj = adj
        self.vertex_num = adj.shape[0]
        self.fcbn_list = nn.ModuleList([nn.Sequential(*[nn.Linear(dim_in, dim_out), nn.BatchNorm1d(dim_out)]) for _ in range(self.vertex_num)])

    def forward(self, feat):
        batch_size = feat.shape[0]

        # apply kernel for each vertex

        feat = torch.stack([fcbn(feat[:,i,:]) for i,fcbn in enumerate(self.fcbn_list)],1)

        # apply adj
        adj = self.adj.cuda()[None,:,:].repeat(batch_size,1,1)
        feat = torch.bmm(adj, feat)

        # apply activation function
        out = F.relu(feat)
        return out
    
class GraphResBlock(nn.Module):
    def __init__(self, adj, dim):
        super(GraphResBlock, self).__init__()
        self.adj = adj
        self.graph_block1 = GraphConvBlock(adj, dim, dim)
        self.graph_block2 = GraphConvBlock(adj, dim, dim)

    def forward(self, feat):
        feat_out = self.graph_block1(feat)
        feat_out = self.graph_block2(feat_out)
        out = feat_out + feat
        return out

class GraphConvNet(nn.Module):
    def __init__(self):
        super(GraphConvNet, self).__init__()
        self.adj_matrix=self.make_adj_matrix(coco_joint_num,coco_skeleton)
        self.graph_block = nn.Sequential(*[\
            GraphConvBlock(self.adj_matrix, 2048, 17),
            GraphResBlock(self.adj_matrix, 17),
            GraphResBlock(self.adj_matrix, 17),
            GraphResBlock(self.adj_matrix, 17),
            GraphResBlock(self.adj_matrix, 17)])

    def make_adj_matrix(self,skt_num,skt_index):
        zero_mat=torch.zeros((skt_num,skt_num))
        for i in range(skt_num):
            zero_mat[i,i]=1
        for i in skt_index:
            zero_mat[i[0],i[1]]=1
            zero_mat[i[1],i[0]]=1
        return zero_mat
    
    def forward(self, img_feat):
        img_feat=self.graph_block(img_feat)
        return img_feat

class GraphConvNet_Linear(nn.Module):
    def __init__(self):
        super(GraphConvNet_Linear,self).__init__()
        self.Linear=nn.Linear(17*17,17*2)
    def forward(self,x):
        x = torch.flatten(x, 1)
        x = self.Linear(x)
        return x
    

class AGGREGATION(nn.Module):
    def __init__(self):
        super(AGGREGATION,self).__init__()
        self.Linear=nn.Linear(cfg.agg_num*2*17,17*10)
        self.Linear1=nn.Linear(10*17,17*2)
    def forward(self,x):
        x = torch.flatten(x, 1)
        x = self.Linear(x)
        x = self.Linear1(x)
        return x

class VIG_Linear(nn.Module):
    def __init__(self):
        super(VIG_Linear,self).__init__()
        self.Linear=nn.Linear(320,17*2)
    def forward(self,x):
        x = torch.flatten(x, 1)
        x = self.Linear(x)
        return x
    

class Object_Attention_module(nn.Module):
    def __init__(self, dim, num_heads):
        super(Object_Attention_module, self).__init__()
        self.k=nn.Linear(dim*9,int(dim*9/8))#,bias=False)
        self.v=nn.Linear(dim*9,int(dim*9/8))#,bias=False)
        self.q=nn.Linear(dim*17,int(dim*17/8))#,bias=False)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.batchnorm1=nn.BatchNorm1d(int(2048*9/8))
        self.batchnorm2=nn.BatchNorm1d(int(2048*9/8))
        self.batchnorm3=nn.BatchNorm1d(int(2048*17/8))
        #self.q_dw=nn.Linear(dim,dim)
        #self.kv_dw=nn.Linear(dim*2,dim*2)
        self.project_out=nn.Linear(int(dim*17/8),int(dim*17/8))#,bias=False)
        self.num_heads=num_heads

    def forward(self,x1,x2):
        b,c = x1.shape
        #kv=self.kv_dw(self.kv(x1))
        #q = self.q_dw(self.q(x2))
        k=self.batchnorm1(self.k(x1))
        v=self.batchnorm2(self.v(x1))
        q = self.batchnorm3(self.q(x2))
        q=q.reshape(b,256,17)
        k=k.reshape(b,256,9)
        v=v.reshape(b,256,9)
        q = rearrange(q, 'b (head c) d -> b head c (d)', head=self.num_heads)
        k = rearrange(k, 'b (head c) d -> b head c (d)', head=self.num_heads)
        v = rearrange(v, 'b (head c) d -> b head c (d)', head=self.num_heads)

        #q = torch.nn.functional.normalize(q, dim=-1)
        #k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q.transpose(-2, -1) @ k ) * self.temperature
        attn = attn.softmax(dim=-1) # attn.softmax(dim=-1)
        out = (attn @ v.transpose(-2, -1))
        out=out.transpose(-2, -1)
        out = rearrange(out, 'b head c (d) -> b (head c) d', head=self.num_heads, d=17)
        out=out.reshape(b,-1)

        out = self.project_out(out)

        return out

class MLP_fusion(nn.Module):
    def __init__(self, dim):
        super(MLP_fusion, self).__init__()
        self.lin1=nn.Linear(dim,dim)#,bias=False)
        self.lin2=nn.Linear(dim,dim)#,bias=False)
        self.fusion=nn.Linear(dim,dim)#,bias=False)
        self.batchnorm1=nn.BatchNorm1d(2048)
        self.batchnorm2=nn.BatchNorm1d(2048)
        self.batchnorm3=nn.BatchNorm1d(2048)
    
    def forward(self,x1,x2):
        x1=self.lin1(x1)
        x1=self.batchnorm1(x1)
        x2=self.lin1(x2)
        x2=self.batchnorm2(x2)
        out = self.fusion(x1+x2) # concat해서 넣기
        out=self.batchnorm3(out)
        return out
    
class TransposedAttention_fusion(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(TransposedAttention_fusion, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        nn.init.normal_(self.kv.weight,std=0.001)
        nn.init.constant_(self.kv.bias, 0)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        nn.init.normal_(self.q.weight,std=0.001)
        nn.init.constant_(self.q.bias, 0)
        self.batchnorm1=nn.BatchNorm2d(2048*2)
        nn.init.normal_(self.batchnorm1.weight,std=0.001)
        nn.init.constant_(self.batchnorm1.bias, 0)
        self.batchnorm2=nn.BatchNorm2d(2048)
        nn.init.normal_(self.batchnorm2.weight,std=0.001)
        nn.init.constant_(self.batchnorm2.bias, 0)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        
        nn.init.normal_(self.kv_dwconv.weight,std=0.001)
        nn.init.constant_(self.kv_dwconv.bias, 0)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)    
        
        nn.init.normal_(self.q_dwconv.weight,std=0.001)
        nn.init.constant_(self.q_dwconv.bias, 0)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        nn.init.normal_(self.project_out.weight,std=0.001)
        nn.init.constant_(self.project_out.bias, 0)

    def forward(self, x1, x2):
        b,c,h,w = x1.shape
        #kv = self.kv_dwconv(self.kv(x1))
        #q = self.q_dwconv(self.q(x2))
        kv=self.batchnorm1(self.kv(x1))
        q=self.batchnorm2(self.q(x2))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        return out

class GCN_Linear(nn.Module):
    def __init__(self):
        super(GCN_Linear,self).__init__()
        self.Linear=nn.Linear(17*128,17*2)
        self.fc2 = nn.Linear(17*128, 17)
    def forward(self,x):
        x = torch.flatten(x, 1)
        x1 = self.fc2(x)
        x = self.Linear(x)
        return x,x1


class skeleton_GCN(nn.Module):
    def __init__(self):
        super(skeleton_GCN,self).__init__()
        self.b_size=cfg.train_batch_size
        self.adj_matrix=self.make_adj_matrix(coco_joint_num,coco_skeleton).cuda()
        #self.lin1=nn.Linear(2048*17,256*17)#,bias=False)
        self.W= nn.Parameter(torch.detach(torch.rand(256,256)))#.requires_grad_(True)).cuda()
        self.W1= nn.Parameter(torch.detach(torch.rand(256,256)))#.requires_grad_(True)).cuda()
        self.W2= nn.Parameter(torch.detach(torch.rand(256,256)))#.requires_grad_(True)).cuda()
        self.W3= nn.Parameter(torch.detach(torch.rand(256,128)))#.requires_grad_(True)).cuda()
        #self.W4= nn.Parameter(torch.detach(torch.rand(128,64)))#.requires_grad_(True)).cuda()
        if cfg.GCN_dim==2:
            self.W1_1= nn.Parameter(torch.detach(torch.rand(512,512)))
            self.W2_1= nn.Parameter(torch.detach(torch.rand(256,256)))
        self.init_param()
        self.act=nn.ReLU()
        self.batchnorm1=nn.BatchNorm1d(256)
        self.batchnorm2=nn.BatchNorm1d(256)
        self.batchnorm3=nn.BatchNorm1d(256)
        self.batchnorm4=nn.BatchNorm1d(128)
        #self.batchnorm5=nn.BatchNorm1d(64)
    def init_param(self):
        nn.init.normal_(self.W,std=0.001)
        nn.init.normal_(self.W1,std=0.001)
        nn.init.normal_(self.W2,std=0.001)
        nn.init.normal_(self.W3,std=0.001)
        #nn.init.normal_(self.W4,std=0.01)
        if cfg.GCN_dim==2:
            nn.init.normal_(self.W1_1,std=0.01)
            nn.init.normal_(self.W2_1,std=0.01)

    def make_adj_matrix(self,skt_num,skt_index):
        zero_mat=torch.zeros((skt_num,skt_num))
        for i in range(skt_num):
            zero_mat[i,i]=1
        for i in skt_index:
            zero_mat[i[0],i[1]]=1
            zero_mat[i[1],i[0]]=1
        return zero_mat

    def forward(self,x):
        batch_size,_,_=x.shape
        adj=self.adj_matrix.expand(batch_size,coco_joint_num,coco_joint_num)
        out = torch.bmm(x,self.W.expand(batch_size,256,256))
        out=out.reshape(batch_size*17,-1)
        out=self.batchnorm1(out)
        out=out.reshape(batch_size,17,-1)
        out=self.act(out)
        out = torch.bmm(adj,out)

        out = torch.bmm(out,self.W1.expand(batch_size,256,256))
        out=out.reshape(batch_size*17,-1)
        out=self.batchnorm2(out)
        out=out.reshape(batch_size,17,-1)
        out=self.act(out)
        out = torch.bmm(adj,out)
        if cfg.GCN_dim==2:
            out = torch.bmm(out,self.W1_1.expand(batch_size,256,256))
            out=self.act(out)
            out = torch.bmm(adj,out)#+out1

        out = torch.bmm(out,self.W2.expand(batch_size,256,256))
        out=out.reshape(batch_size*17,-1)
        out=self.batchnorm3(out)
        out=out.reshape(batch_size,17,-1)
        out=self.act(out)
        out = torch.bmm(adj,out)
        if cfg.GCN_dim==2:
            out = torch.bmm(out,self.W2_1.expand(batch_size,256,256))
            out=self.act(out)
            out = torch.bmm(adj,out)#+out1

        out = torch.bmm(out,self.W3.expand(batch_size,256,128))
        out=out.reshape(batch_size*17,-1)
        out=self.batchnorm4(out)
        out=out.reshape(batch_size,17,-1)
        out=self.act(out)
        out = torch.bmm(adj,out)

        # out = torch.bmm(out,self.W4.expand(batch_size,128,64))
        # out=out.reshape(batch_size*17,-1)
        # out=self.batchnorm5(out)
        # out=out.reshape(batch_size,17,-1)
        # out=self.act(out)
        # out = torch.bmm(adj,out)
        return out

class Feat2pose(nn.Module):
    def __init__(self):
        super(Feat2pose, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 2*17)
        self.fc2 = nn.Linear(2048, 17)
        if cfg.bbox_loss==True:
            self.fc3=nn.Linear(2048,4)
    def forward(self, feature):
        x=self.avgpool(feature)
        x = torch.flatten(x, 1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        if cfg.bbox_loss==True:
            x3 = self.fc3(x)
            return x1,x2,x3
        else:
            return x1,x2
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 2*17)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(2*17, 1)
    def forward(self, feature, feature_map=True):
        if feature_map==True:
            batch=feature.shape[0]
            x=self.avgpool(feature)
            x = torch.flatten(x, 1)
            x1 = self.fc1(x)
            x1 = self.relu(x1)
            x1 = self.fc2(x1)
            return x1.reshape(batch,17,2)
        else:
            feature = torch.flatten(feature, 1)
            cls=self.fc3(feature)
            return cls