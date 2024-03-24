import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os.path as osp
sys.path.insert(0, osp.join('..', 'common'))
from nets.resnet import ResNetBackbone
from config import cfg
from nets.module import Feat2pose, Discriminator,AGGREGATION,Object_Attention_module,MLP_fusion, TransposedAttention_fusion,skeleton_GCN,GCN_Linear,GraphConvNet,GraphConvNet_Linear,VIG_Linear
from nets.GAT import GAT
from nets.loss import distance_loss
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
sys.path.append("putting your root/common/")
from nets.cls_hrnet import HighResolutionNet,get_cls_net
from nets.hrnet_config import cfg as cfg_t
from nets.hrnet_config import update_config
from nets.diffusionpose import D3DP
from nets.vig import vig_s_224_gelu,vig_b_224_gelu
from torchvision.ops import RoIPool
from nets.arguments import parse_args as diffusion_args



class Model(nn.Module):
    def __init__(self, backbone,feat2pose,CrossAttention4,GCN_module,GCN_linear,objattention,model_pos_train,model_pos_test_temp,aggregation):
        super(Model, self).__init__()
        self.backbone = backbone
        self.feat2pose=feat2pose
        if cfg.obj_attention==True:
            self.CrossAttention4=CrossAttention4
        if cfg.GCN_Model=="GNN":
            self.GCN_module=GCN_module
            self.GCN_linear=GCN_linear
        if cfg.aggregation==True:
            self.aggregation=aggregation
        self.distance_loss=distance_loss()
        self.loss =nn.L1Loss()
        self.sigmoid=nn.Sigmoid()
        self.roipool=RoIPool(output_size=(8, 8), spatial_scale=1.)
        self.conv1_3=nn.Conv2d(1024, 512, 3, stride=1,padding=1,bias=False)
        self.conv1_3_bn = nn.BatchNorm2d(1024)
        self.relu=nn.ReLU()
        self.conv2_3=nn.Conv2d(512, 256, 3, stride=1,padding=1,bias=False)
        self.conv2_3_bn = nn.BatchNorm2d(512)
        self.relu=nn.ReLU()
        nn.init.normal_(self.conv1_3.weight,std=0.001)
        if cfg.GCN_Model=="GNN":
            self.conv2048_256=nn.Conv2d(2048, 256, 1, stride=1,bias=False)
            nn.init.normal_(self.conv1_3.weight,std=0.001)
            self.GNN_Linear=nn.Linear(64*256,17*256)
            nn.init.normal_(self.GNN_Linear.weight,std=0.001)
        if cfg.diffusion==True:
            self.dif_train=model_pos_train
            self.dif_test=model_pos_test_temp



    def make_2d_gaussian_heatmap(self, joint_coord_img):
        x = torch.arange(cfg.output_hm_shape[2])
        y = torch.arange(cfg.output_hm_shape[1])
        yy, xx = torch.meshgrid(y, x)
        xx = xx[None, None, :, :].cuda().float();
        yy = yy[None, None, :, :].cuda().float();

        x = joint_coord_img[:, :, 0, None, None];
        y = joint_coord_img[:, :, 1, None, None];
        heatmap = torch.exp(
            -(((xx - x) / cfg.sigma) ** 2) / 2 - (((yy - y) / cfg.sigma) ** 2) / 2)
        return heatmap

    def mix_up_tensor(self,tesnor_torch):
        batch=tesnor_torch.shape[0]
        random_tesnor_torch=tesnor_torch.detach().clone().cpu()
        for i in range(batch):
            index=int(random.random()*batch)
            while i==index:
                index=int(random.random()*batch)
            random_tesnor_torch[i]=tesnor_torch[index]
        return random_tesnor_torch.cuda()
                
    def forward(self, inputs,target,mode):
        batch_size=inputs['img'].shape[0]
        feature_map = self.backbone(inputs['img'],skip_early=True)  # backbone
        bbox_1,bbox_2,bbox_3=inputs['feat_bbox_1'],inputs['feat_bbox_2'],inputs['feat_bbox_3']
        x1,x2,x3,x4=self.backbone(feature_map)
        roi_feature3=self.roipool(x3,[bbox_2])
        roi_feature3=self.relu(self.conv1_3(self.conv1_3_bn(roi_feature3)))
        roi_feature3=self.relu(self.conv2_3(self.conv2_3_bn(roi_feature3)))
        output,confidence=self.feat2pose(x4)
        if mode=='train':
            output1=self.dif_train(target['normalization_joint'],roi_feature3,x4)
        else:
            demo_test=torch.zeros(batch_size,17,2)
            output1=self.dif_test(demo_test,roi_feature3,x4)

        if mode=='train':
            output=output.reshape(batch_size,17,2) * inputs['coco_joint_trunc'] * inputs['coco_joint_valid']
            if cfg.diffusion==True:
                output1=output1.reshape(batch_size,17,2) * inputs['coco_joint_trunc'] * inputs['coco_joint_valid']


            target_pose=target['normalization_joint'] * inputs['coco_joint_trunc'] * inputs['coco_joint_valid']

            hyper_param=target['hyper_param']
            loss = {}
            loss['confidence'] = self.loss(self.sigmoid(confidence.reshape(batch_size,17,1)), inputs['coco_joint_trunc'] * inputs['coco_joint_valid'])
            loss['dif_key_conf']=self.loss(target_pose * hyper_param,output1 * inputs['coco_joint_trunc'] * inputs['coco_joint_valid'] * hyper_param)
            return loss
        else:
            if cfg.demo==False:
                target_pose=target['normalization_joint'] * inputs['coco_joint_trunc'] * inputs['coco_joint_valid']

                out={'output1':output1.reshape(batch_size,17,2),
                    'output':output.reshape(batch_size,17,2),
                    'confidence':self.sigmoid(confidence.reshape(batch_size,17,1)),
                    'gt_valid':inputs['coco_joint_valid'].reshape(batch_size,17,1),
                    'normalization_joint':target['normalization_joint'],
                    'img_id':target['img_id'],
                    "bb2img_trans":target['bb2img_trans']}
            else :
                confidence=self.sigmoid(confidence.reshape(batch_size,17,1))
                output1=output1*confidence
                out={'output':output1.reshape(batch_size,17,2),
                     'confidence':confidence.reshape(batch_size,17,1),
                    'input':inputs['img'],
                    "bb2img_trans":target['bb2img_trans'],
                    "obj_bbox":target['obj_bbox'],
                    "p_bbox":target['p_bbox']}
            return out
    
def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.BatchNorm1d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias,0)

def get_model(mode):
    if cfg.backbone=='resnet':
        backbone = ResNetBackbone(cfg.resnet_type)
    else:
        import os.path as osp
        if cfg.hrnet_type==18:
            config_file = osp.join("./common/nets/models/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml")
            pretrained_model_path=osp.join("./common/nets/hrnet/pretrained_model/HRNet_W18_C_ssld_pretrained.pth")
        if cfg.hrnet_type==32:
            config_file = osp.join("./common/nets/models/cls_hrnet_w32_sgd_lr5e-2_wd1e-4_bs32_x100.yaml")
            pretrained_model_path=osp.join("./common/nets/hrnet/pretrained_model/hrnetv2_w32_imagenet_pretrained.pth")
        if cfg.hrnet_type==48:
            config_file = osp.join("./common/nets/models/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml")
            pretrained_model_path=osp.join("./common/nets/hrnet/pretrained_model/hrnetv2_w48_imagenet_pretrained.pth")
        if cfg.hrnet_type==64:
            config_file = osp.join("./common/nets/models/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml")
            pretrained_model_path=osp.join("./common/nets/hrnet/pretrained_model/hrnetv2_w64_imagenet_pretrained.pth")

        update_config(cfg_t, config_file)
        backbone = get_cls_net(cfg_t,pretrained_model_path)
    feat2pose = Feat2pose()
    CrossAttention4=TransposedAttention_fusion(2048,8,True)
    #CrossAttention4.apply(init_weights)
    if cfg.obj_attention==True:
        objattention=Object_Attention_module(2048,8)
        objattention.apply(init_weights)
    elif cfg.MLP_fusion==True:
        objattention=MLP_fusion(2048)
        objattention.apply(init_weights)

    else:
        objattention=None

    if mode == 'train':
        backbone.init_weights()
        feat2pose.apply(init_weights)
        GCN_module=None
        GCN_linear=None
    else:
        GCN_module=None
        GCN_linear=None
    if cfg.diffusion==True:
        #dif_args=diffusion_args()
        dif_args={
            "timestep":1000,
            "scale":1.0,
            "cs":cfg.dif_cs,
            "dep":8,
        }
        model_pos_train = D3DP(dif_args, is_train=True)
        model_pos_train.apply(init_weights)
        model_pos_test_temp = D3DP(dif_args, is_train=False)
    else:
        model_pos_train=None
        model_pos_test_temp=None
    if cfg.aggregation==False:
        model = Model(backbone,feat2pose,CrossAttention4,GCN_module,GCN_linear,objattention,model_pos_train,model_pos_test_temp,None)
    else:
        #aggregation=AGGREGATION()
        aggregation=None
        model = Model(backbone,feat2pose,CrossAttention4,GCN_module,GCN_linear,objattention,model_pos_train,model_pos_test_temp,aggregation)
        
    return model
