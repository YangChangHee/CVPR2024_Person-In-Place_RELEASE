## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg
import time

from math import sqrt

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

coco_skeleton = ( (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 6), (11, 12),(5,11),(6,12),(3,5),(4,6) )
coco_joint_num = 17

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

import numpy as np
import torch
import torch.nn as nn

class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        #self.register_buffer("cached_penc", None)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        #if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
        #    return self.cached_penc

        cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return cached_penc

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., changedim=False, currentdim=0, depth=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class self_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False, vis=False):
        super().__init__()
        self.adj_matrix=self.make_adj_matrix(coco_joint_num,coco_skeleton).cuda()
        self.W= nn.Parameter(torch.detach(torch.rand(8*64,8*64)))
        self.batchnorm1=nn.BatchNorm1d(8*64)
        #self.W1= nn.Parameter(torch.detach(torch.rand(8*64,8*64)))
        #self.batchnorm2=nn.BatchNorm1d(8*64)
        self.act=nn.ReLU()
        nn.init.normal_(self.W,std=0.001)
        #nn.init.normal_(self.W1,std=0.001)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.kv=nn.Linear(dim,dim*2,bias=qkv_bias)
        self.q=nn.Linear(dim,dim,bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) 

        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb
        self.vis = vis
    
    def make_adj_matrix(self,skt_num,skt_index):
        zero_mat=torch.zeros((skt_num,skt_num))
        for i in range(skt_num):
            zero_mat[i,i]=1
        for i in skt_index:
            zero_mat[i[0],i[1]]=1
            zero_mat[i[1],i[0]]=1
        return zero_mat
    
    def forward(self,noise,obj_emb,vis=False):
        B, N, C = noise.shape
        _,oN, C= obj_emb.shape
        kv = self.kv(obj_emb).reshape(B, oN, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = self.q(noise).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Now x shape (3, B, heads, N, C//heads)
        q, k, v = q[0], kv[0], kv[1]   # make torchscript happy (cannot use tensor as tuple)
        if self.comb==True:
            attn = (q.transpose(-2, -1) @ k) * self.scale
        elif self.comb==False:
            attn = (q @ k.transpose(-2, -1)) * self.scale

        adj=self.adj_matrix.expand(B,coco_joint_num,coco_joint_num)
        attn=attn.transpose(1,2)
        attn=attn.reshape(B,coco_joint_num,-1)
        attn = torch.bmm(attn,self.W.expand(B,8*64,8*64))
        attn=attn.reshape(B*17,-1)
        attn=self.batchnorm1(attn)
        attn=attn.reshape(B,17,-1)
        
        attn = torch.bmm(adj,attn)
        #attn = torch.bmm(adj,attn)
        attn=attn.reshape(B,coco_joint_num,self.num_heads,-1)
        attn=attn.transpose(1,2)
        

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        if self.comb==True:
            x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
            x = rearrange(x, 'B H N C -> B N (H C)')
        elif self.comb==False:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class self_Attention_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., attention=self_Attention, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, comb=False, changedim=False, currentdim=0, depth=0, vis=False):
        super().__init__()

        self.changedim = changedim
        self.currentdim = currentdim
        self.depth = depth
        if self.changedim:
            assert self.depth>0

        self.norm1 = norm_layer(dim)
        self.attn = attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, comb=comb, vis=vis)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        if self.changedim and self.currentdim < self.depth//2:
            self.reduction = nn.Conv1d(dim, dim//2, kernel_size=1)
            # self.reduction = nn.Linear(dim, dim//2)
        elif self.changedim and depth > self.currentdim > self.depth//2:
            self.improve = nn.Conv1d(dim, dim*2, kernel_size=1)
            # self.improve = nn.Linear(dim, dim*2)
        self.vis = vis

    def forward(self, noise,obj_emb, vis=False):
        #print(obj_emb.shape)
        #print(self.drop_path(self.attn(self.norm1(noise),self.norm1(obj_emb), vis=vis)).shape)
        x = noise + self.drop_path(self.attn(self.norm1(noise),self.norm1(obj_emb), vis=vis))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        if self.changedim and self.currentdim < self.depth//2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.reduction(x)
            x = rearrange(x, 'b c t -> b t c')
        elif self.changedim and self.depth > self.currentdim > self.depth//2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.improve(x)
            x = rearrange(x, 'b c t -> b t c')
        return x



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False, vis=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) 

        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb
        self.vis = vis

    def forward(self, x, vis=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Now x shape (3, B, heads, N, C//heads)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        if self.comb==True:
            attn = (q.transpose(-2, -1) @ k) * self.scale
        elif self.comb==False:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        if self.comb==True:
            x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
            x = rearrange(x, 'B H N C -> B N (H C)')
        elif self.comb==False:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., attention=Attention, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, comb=False, changedim=False, currentdim=0, depth=0, vis=False):
        super().__init__()

        self.changedim = changedim
        self.currentdim = currentdim
        self.depth = depth
        if self.changedim:
            assert self.depth>0

        self.norm1 = norm_layer(dim)
        self.attn = attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, comb=comb, vis=vis)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        if self.changedim and self.currentdim < self.depth//2:
            self.reduction = nn.Conv1d(dim, dim//2, kernel_size=1)
            # self.reduction = nn.Linear(dim, dim//2)
        elif self.changedim and depth > self.currentdim > self.depth//2:
            self.improve = nn.Conv1d(dim, dim*2, kernel_size=1)
            # self.improve = nn.Linear(dim, dim*2)
        self.vis = vis

    def forward(self, x, vis=False):
        x = x + self.drop_path(self.attn(self.norm1(x), vis=vis))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        if self.changedim and self.currentdim < self.depth//2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.reduction(x)
            x = rearrange(x, 'b c t -> b t c')
        elif self.changedim and self.depth > self.currentdim > self.depth//2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.improve(x)
            x = rearrange(x, 'b c t -> b t c')
        return x

class skeleton_GCN(nn.Module):
    def __init__(self):
        super(skeleton_GCN,self).__init__()
        self.b_size=64
        self.adj_matrix=self.make_adj_matrix(coco_joint_num,coco_skeleton).cuda()
        #self.lin1=nn.Linear(2048*17,256*17)#,bias=False)
        self.W= nn.Parameter(torch.detach(torch.rand(32,32)))#.requires_grad_(True)).cuda()
        self.W1= nn.Parameter(torch.detach(torch.rand(32,32)))#.requires_grad_(True)).cuda()
        self.W2= nn.Parameter(torch.detach(torch.rand(32,32)))#.requires_grad_(True)).cuda()
        self.W3= nn.Parameter(torch.detach(torch.rand(32,32)))#.requires_grad_(True)).cuda()
        self.init_param()
        self.act=nn.ReLU()
        self.batchnorm1=nn.BatchNorm1d(32)
        self.batchnorm2=nn.BatchNorm1d(32)
        self.batchnorm3=nn.BatchNorm1d(32)
        self.batchnorm4=nn.BatchNorm1d(32)
    def init_param(self):
        nn.init.normal_(self.W,std=0.001)
        nn.init.normal_(self.W1,std=0.001)
        nn.init.normal_(self.W2,std=0.001)
        nn.init.normal_(self.W3,std=0.001)
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
        out = torch.bmm(x,self.W.expand(batch_size,32,32))
        out=out.reshape(batch_size*17,-1)
        out=self.batchnorm1(out)
        out=out.reshape(batch_size,17,-1)
        out=self.act(out)
        out = torch.bmm(adj,out)

        out = torch.bmm(out,self.W1.expand(batch_size,32,32))
        out=out.reshape(batch_size*17,-1)
        out=self.batchnorm2(out)
        out=out.reshape(batch_size,17,-1)
        out=self.act(out)
        out = torch.bmm(adj,out)
        return out


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class  MixSTE2(nn.Module):
    def __init__(self, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, is_train=True):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = 2     #### output dimension is num_joints * 3
        self.is_train=is_train
        self.pe_embedding_shape=embed_dim_ratio

        ### spatial patch embedding
        # self.Spatial_patch_to_embedding = nn.Linear(in_chans + 2, embed_dim_ratio)
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        # 1 17 512
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.GCN=skeleton_GCN()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(embed_dim_ratio),
            nn.Linear(embed_dim_ratio, embed_dim_ratio*2),
            nn.GELU(),
            nn.Linear(embed_dim_ratio*2, embed_dim_ratio),
        )


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.block_depth = depth

        self.STEblocks = nn.ModuleList([
            # Block: Attention Block
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.OBJblocks = nn.ModuleList([
            # Block: Attention Block
            self_Attention_Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer)])
        
        self.PE=PositionalEncoding1D(embed_dim_ratio).cuda()
        self.Spatial_norm = norm_layer(embed_dim_ratio)



        self.image_feauter_linear = nn.Sequential(
                    nn.LayerNorm(2048),
                    nn.Linear(2048 , 17*embed_dim_ratio),
        )
        self.obj_feauter_linear = nn.Sequential(
                    nn.LayerNorm(64*32),
                    nn.Linear(64*32 , 17*embed_dim_ratio),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )


    def STE_forward_1(self, x_3d,image_feature,object_feature, t):
        batch_size,_,_=x_3d.shape
        Positional_embedding_objshape=self.PE(torch.zeros(batch_size,64,32).cuda())
        Positional_embedding_objshape=Positional_embedding_objshape.reshape(object_feature.shape)
        feature_embedding=self.image_feauter_linear(image_feature)
        feature_embedding=feature_embedding.reshape(batch_size,17,-1)
        

        if self.is_train:
            x = x_3d
            b, n, c = x.shape
            x = self.Spatial_patch_to_embedding(x)
            x += self.Spatial_pos_embed
            x += feature_embedding
            
            for i in range(0, 1):
                objblock = self.OBJblocks[i]
                x = objblock(x,object_feature)
                x = self.Spatial_norm(x)
            
            time_embed = self.time_mlp(t)[:, None, :].repeat(1,n,1)
            x += time_embed
        else:
            x=x_3d
            b, n, c = x.shape
            x = self.Spatial_patch_to_embedding(x)
            x += self.Spatial_pos_embed
            x += feature_embedding
            for i in range(0, 1):
                objblock = self.OBJblocks[i]
                x = objblock(x,object_feature)
                x = self.Spatial_norm(x)
            
            time_embed = self.time_mlp(t)[:, None, :].repeat(1,n, 1)
            x += time_embed

        x = self.pos_drop(x)
        blk = self.STEblocks[0]
        x = blk(x)

        x = self.Spatial_norm(x)
        return x
    
    def ST_foward(self, x):
        assert len(x.shape)==3, "shape is equal to 4"
        b, n, cw = x.shape
        for i in range(1, self.block_depth):
            steblock = self.STEblocks[i]
            
            x = steblock(x)
            x = self.Spatial_norm(x)
        
        return x

    def forward(self, gt_2d,image_feature,object_feature, t):

        x = self.STE_forward_1(gt_2d,image_feature,object_feature, t)
        x = self.ST_foward(x)

        x = self.head(x)
        return x