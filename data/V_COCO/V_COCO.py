import os
import os.path as osp
import numpy as np
from config import cfg
import copy
import json
import scipy.io as sio
import cv2
import random
import math
import torch
import transforms3d
from pycocotools.coco import COCO
import torchvision.transforms as transforms
from utils.evaluate import accuracy, accuracy_bbox
from utils.preprocessing import load_img, process_bbox, augmentation, compute_iou
from utils.transforms import denorm_joints
from utils.vis import vis_keypoints_with_skeleton
import torchvision.transforms as T
from scipy.spatial import distance
import math


import torch.nn.functional as F

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)



class V_COCO(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform1 = transforms.Compose([
            transforms.ToTensor(),
            self.normalize])
        self.transform = transform
        self.data_split = 'train' if data_split == 'train' else 'val'
        # putting your path => /home/user/data
        self.background_path = osp.join('putting your path/MSCOCO2014_all')
        self.annot_path = osp.join('putting your path/data/V_COCO')
        # 
        # mscoco skeleton
        self.coco_joint_num = 17 # original: 17, manually added pelvis
        self.LSPET_joints_name = ('R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'R_Wrist', 'R_Elbow', \
            'R_Shoulder', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'Neck', 'Head_top')
        self.LSPET_joint_num=14
        self.coco_joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle')
        self.non_face_coco_index=[5,6,7,8,9,10,11,12,13,14,15,16]
        self.coco_skeleton = ( (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 6), (11, 12),(5,11),(6,12),(3,5),(4,6) )
        
        self.coco_flip_pairs = ( (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16) )

        self.datalist = self.load_data()
        betas = cosine_beta_schedule(1000)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        self.sqrt_alphas_cumprod= torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod= torch.sqrt(1. - alphas_cumprod)
        self.scale=1.0
        print("coco data len: ", len(self.datalist))

    def q_sample(self,x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def matching_bbox(self,obj_bbox,keypoints,bbox_factor):
        index_joint=[]
        distance_j=[]
        for num,i in enumerate(keypoints):
            if i[2]<0.7:
                index_joint.append(-1)
                distance_j.append(1000000)
                continue
            x,y=i[0],i[1]
            person_bbox=[x-bbox_factor/2,y-bbox_factor/2,x+bbox_factor/2,y+bbox_factor/2]
            box_factor_obj=cfg.obj_bbox_factor
            obj_width=obj_bbox[2]-obj_bbox[0]
            obj_height=obj_bbox[3]-obj_bbox[1]
            center_x=(obj_bbox[0]+obj_bbox[2])/2
            center_y=(obj_bbox[1]+obj_bbox[3])/2
            new_obj_bbox=[center_x-obj_width*box_factor_obj/2,center_y-obj_height*box_factor_obj/2,center_x+obj_width*box_factor_obj/2,center_y+obj_height*box_factor_obj/2]
            index_joint.append(self.Object_IoU(new_obj_bbox,person_bbox))
            distance_j.append(self.points_distance([center_x,center_y],[i[0],i[1]]))
        return index_joint,distance_j

    def points_distance(self,point_1,point_2):
        return math.sqrt((point_1[0]-point_2[0])**2 + (point_1[1]-point_2[1])**2)

    def IoU(self,box1, box2):
        # box = (x1, y1, x2, y2)
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # obtain x1, y1, x2, y2 of the intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # compute the width and height of the intersection
        w = max(0, x2 - x1 + 1)
        h = max(0, y2 - y1 + 1)

        inter = w * h
        iou = inter / (box1_area + box2_area - inter)
        if np.isnan(iou):
            iou=-1
        return iou

    def Object_IoU(self,box1, box2):
        # box = (x1, y1, x2, y2)
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # obtain x1, y1, x2, y2 of the intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # compute the width and height of the intersection
        w = max(0, x2 - x1 + 1)
        h = max(0, y2 - y1 + 1)

        inter = w * h
        iou = inter / (box1_area)
        if np.isnan(iou):
            iou=-1
        return iou

    def load_data(self):
        if self.data_split == 'train':
            db = COCO(osp.join(self.annot_path, 'object_keypoint_' + self.data_split + '2014.json'))

            datalist = []
            for iid in db.imgs.keys():
                aids = db.getAnnIds([iid])

                tmplist = []
                for aid in aids:
                    ann = db.anns[aid]
                    img = db.loadImgs(ann['image_id'])[0]
                    file_name=img['file_name'].split("_")[-1]
                    background_path=osp.join(self.background_path,"inpainted_with_"+file_name)
                    width, height = img['width'], img['height']
                    img_id=int(img['file_name'].split("/")[-1].split("_")[-1][:-4])
                    if ann['iscrowd'] or (ann['num_keypoints'] == 0):
                        continue
                    if cfg.bbox_loss==True:
                        if ann['num_keypoints']<10:
                            continue
                    # bbox
                    tight_bbox = np.array(ann['bbox'])
                    # large scale bbox
                    bbox_factor=math.sqrt(tight_bbox[2]*tight_bbox[3])/cfg.bbox_factor_param
                    tight_bbox[0]=tight_bbox[0]-tight_bbox[2]/2
                    tight_bbox[1]=tight_bbox[1]-tight_bbox[3]/2
                    tight_bbox[2]=tight_bbox[2]*2
                    tight_bbox[3]=tight_bbox[3]*2
                    bbox = process_bbox(tight_bbox, width, height)
                    if bbox is None: continue
                    #if tight_bbox is None: continue
                    # joint coordinates
                    joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
                    
                    joint_valid = (joint_img[:, 2].copy().reshape(-1, 1) > 0).astype(np.float32)
                    joint_img[:, 2] = joint_valid[:, 0] # for posefix, only good for 2d datasets
                    if not os.path.isfile(background_path):
                        continue
                    for action,obj_bbox in zip(ann['class'],ann['obj_bbox']):
                        what_keypoints,distance_j=self.matching_bbox(obj_bbox,joint_img,bbox_factor)
                        tmplist.append({
                            'img_path': background_path,
                            'img_shape': (height, width),
                            'bbox': bbox,
                            'joint_img': joint_img,
                            'joint_valid': joint_valid,
                            'image_id': img_id,
                            'action_class':action,
                            'obj_bbox':obj_bbox,
                            'bbox_factor':bbox_factor,
                            'tight_bbox':tight_bbox,
                            'evaluation':what_keypoints,
                            'p_distance':distance_j
                        })

                datalist.extend(tmplist)
            return datalist
        else:
            #db = COCO(osp.join(self.annot_path, 'person_keypoints_' + self.data_split + '2017.json'))
            db = COCO(osp.join(self.annot_path, 'object_keypoint_' + self.data_split + '2014.json'))
            datalist=[]
            for iid in db.imgs.keys():
                aids = db.getAnnIds([iid])

                tmplist = []
                for aid in aids:
                    ann = db.anns[aid]
                    if 437148 == aid:
                        print(img['file_name'].split("_")[-1])
                    img = db.loadImgs(ann['image_id'])[0]
                    file_name=img['file_name'].split("_")[-1]
                    background_path=osp.join(self.background_path,"inpainted_with_"+file_name)
                    width, height = img['width'], img['height']
                    img_id=int(img['file_name'].split("/")[-1].split("_")[-1][:-4])
                    if ann['iscrowd'] or (ann['num_keypoints'] == 0):
                        continue
                    if cfg.bbox_loss==True:
                        if ann['num_keypoints']<10:
                            continue
                    tight_bbox = np.array(ann['bbox'])
                    bbox_factor=math.sqrt(tight_bbox[2]*tight_bbox[3])/cfg.bbox_factor_param
                    tight_bbox[0]=tight_bbox[0]-tight_bbox[2]/2
                    tight_bbox[1]=tight_bbox[1]-tight_bbox[3]/2
                    tight_bbox[2]=tight_bbox[2]*2
                    tight_bbox[3]=tight_bbox[3]*2
                    bbox = process_bbox(tight_bbox, width, height)
                    if bbox is None: continue
                    joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
                    
                    joint_valid = (joint_img[:, 2].copy().reshape(-1, 1) > 0).astype(np.float32)
                    joint_img[:, 2] = joint_valid[:, 0]
                    if not os.path.isfile(background_path):
                        continue
                    for action,obj_bbox in zip(ann['class'],ann['obj_bbox']):
                        what_keypoints,distance_j=self.matching_bbox(obj_bbox,joint_img,bbox_factor)
                        tmplist.append({
                            'img_path': background_path,
                            'img_shape': (height, width),
                            'bbox': bbox,
                            'joint_img': joint_img,
                            'joint_valid': joint_valid,
                            'image_id': aid,
                            'action_class':action,
                            'obj_bbox':obj_bbox,
                            'bbox_factor':bbox_factor,
                            'tight_bbox':tight_bbox,
                            'evaluation':what_keypoints,
                            'p_distance':distance_j
                        })

                datalist.extend(tmplist)
            return datalist

    def prepare_diffusion_concat(self, pose_2d):

        t = torch.randint(0, cfg.augmentation_ratio, (1,), device='cpu').long()
        noise = torch.randn(17, 2, device='cpu')

        x_start = pose_2d

        x_start = x_start * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min= -1.1 * self.scale, max= 1.1*self.scale)
        x = x / self.scale

        return x, noise, t
    
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']
        img=load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip=augmentation(img,bbox,self.data_split)
        img = self.transform(img.astype(np.float32))
        img=img/255.

        coco_joint_img = data['joint_img']
        coco_joint_valid = data['joint_valid']
        if do_flip:
            coco_joint_img[:,0] = img_shape[1] - 1 - coco_joint_img[:,0]
            for pair in self.coco_flip_pairs:
                coco_joint_img[pair[0],:], coco_joint_img[pair[1],:] = coco_joint_img[pair[1],:].copy(), coco_joint_img[pair[0],:].copy()
                coco_joint_valid[pair[0],:], coco_joint_valid[pair[1],:] = coco_joint_valid[pair[1],:].copy(), coco_joint_valid[pair[0],:].copy()

        coco_joint_img_xy1 = np.concatenate((coco_joint_img[:,:2], np.ones_like(coco_joint_img[:,:1])),1)
        coco_joint_img[:,:2] = np.dot(img2bb_trans, coco_joint_img_xy1.transpose(1,0)).transpose(1,0)
        normalization_joint=np.zeros((coco_joint_img.shape))
        normalization_joint=coco_joint_img[:,:2]/256
        coco_joint_trunc = coco_joint_valid * ((coco_joint_img[:,0] >= 0) * (coco_joint_img[:,0] < 256) * \
                    (coco_joint_img[:,1] >= 0) * (coco_joint_img[:,1] < 256)).reshape(-1,1).astype(np.float32)


        action=data['action_class']
        obj_bbox=data['obj_bbox']
        bbox_coord=np.array([[obj_bbox[0],obj_bbox[1]],[obj_bbox[0],obj_bbox[3]],[obj_bbox[2],obj_bbox[1]],[obj_bbox[2],obj_bbox[3]]])
        bbox_coord_img_xy1 = np.concatenate((bbox_coord[:,:2], np.ones_like(bbox_coord[:,:1])),1)
        bbox_coord[:,:2] = np.dot(img2bb_trans, bbox_coord_img_xy1.transpose(1,0)).transpose(1,0)
        bbox_coord=np.where(bbox_coord<0,0,bbox_coord)
        bbox_coord=np.where(bbox_coord>256,256,bbox_coord)
        x_max,x_min,y_max,y_min=np.max(bbox_coord[:,0]),np.min(bbox_coord[:,0]),np.max(bbox_coord[:,1]),np.min(bbox_coord[:,1])

        new_bbox=np.array([x_min,x_max,y_min,y_max])

        center_obj_bbox=np.array([[(x_max+x_min)/2,(y_max+y_min)/2]])
        center_obj_bbox=np.repeat(center_obj_bbox,17,axis=0)
        index_selection=np.array(coco_joint_valid).reshape(-1) * np.array(coco_joint_trunc).reshape(-1)
        index_selection=np.array(index_selection,dtype=np.bool)
        
        distance_joint=coco_joint_img[:,:2]*1
        distance_obj_keypoint=distance_joint-center_obj_bbox # => distance
        dim1_distance=np.sqrt(distance_obj_keypoint[:,0]**2+distance_obj_keypoint[:,1]**2)
        dim1_distance=dim1_distance[index_selection]


        one_npy=np.ones((len(dim1_distance)))*20
        distance_soft=self.softmax(one_npy/dim1_distance)
        hyper_param=np.zeros((17))
        num=0
        for n_j,i in enumerate(index_selection):
            if i==0:
                continue
            else:
                hyper_param[n_j]=distance_soft[num]
                num+=1


        
        
        new_bbox=(new_bbox/4).astype('float32') # 64
        new_bbox_1=(new_bbox/4).astype('float32') # 16
        new_bbox_2=(new_bbox/8).astype('float32') # 8
        new_bbox_index=new_bbox_2.astype('int16')
        mask=np.ones((1024,16,16))
        mask=mask*cfg.filtering_param
        mask[:,new_bbox_index[2]:new_bbox_index[3],new_bbox_index[0]:new_bbox_index[1]]=1
        mask=mask.astype('float32')
        inputs = {'img': img,"coco_joint_trunc":coco_joint_trunc,"coco_joint_valid":coco_joint_valid,"mask":torch.from_numpy(mask),'feat_bbox_1':torch.from_numpy(new_bbox),'feat_bbox_2':torch.from_numpy(new_bbox_1),'feat_bbox_3':torch.from_numpy(new_bbox_2)}
        targets = {'hyper_param':hyper_param.reshape(17,1)*100,'normalization_joint': normalization_joint,"img_id":data['image_id'],"bb2img_trans":bb2img_trans,"action":action,"obj_bbox":obj_bbox}#,"index_selection":np.array(index_selection,dtype=np.bool),"r_loss_term":r_loss_term}
        return inputs, targets
    def softmax(self,overlape):
        exp_overlape=np.exp(overlape)
        sum_overlape=np.sum(exp_overlape)
        y=exp_overlape/sum_overlape
        return y
    
    def sharpen(self,probabilities, T):

        if probabilities.ndim == 1:
            tempered = torch.pow(probabilities, 1 / T)
            tempered = (
                tempered
                / (torch.pow((1 - probabilities), 1 / T) + tempered)
            )

        else:
            tempered = torch.pow(probabilities, 1 / T)
            tempered = tempered / tempered.sum(dim=-1, keepdim=True)

        return tempered

    def KL_divergence(self,gt,pred):
        return distance.jensenshannon(gt,pred,2.0)
    
    def evaluate(self,outs,cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        acc_num=0
        list_id=[]
        zero_num=0
        dif_acc_num=0
        forward_iou={"1":0,"3":0,"5":0}
        dif_iou={"1":0,"3":0,"5":0}
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            bkg_path=annot['img_path']
            bbox_factor=annot['bbox_factor']
            obj_bbox=annot['obj_bbox']
            gt_iou_list=annot['evaluation']
            distance_points=annot['p_distance']
            index_list=[]
            for i in range(0,len(distance_points)):
                min_index=distance_points.index(min(distance_points))
                index_list.append(min_index)
                distance_points[min_index]=1000001


            out = outs[n]
            gt_keypoints=out['normalization_joint']*255
            dif_keypoints=out['output1']*255
            img_id=out['img_id']
            valid=out['confidence']
            gt_valid=out['gt_valid']

            def extract(a, t, x_shape):
                batch_size = t.shape[0]
                out = a.gather(-1, t)
                return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
            def q_sample(x_start, t, noise=None):
                if noise is None:
                    noise = torch.randn_like(x_start)

                sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
                sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

                return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


            dif_pose_out_img = denorm_joints(dif_keypoints.copy(), out['bb2img_trans'])
            dif_pose_out_img = np.concatenate((dif_pose_out_img, gt_valid), axis=1)

            gt_out_img = denorm_joints(gt_keypoints.copy(), out['bb2img_trans'])
            gt_out_img = np.concatenate((gt_out_img, gt_valid), axis=1)
            
            
            
            
            dif_pred_iou_list,_=self.matching_bbox(obj_bbox,dif_pose_out_img,bbox_factor)
            new_gt=[]
            new_pred=[]
            dif_gt=[]
            dif_pred=[]
            for top in [1,3,5]:
                for top_n,i in enumerate(index_list):
                    gt,pred=gt_iou_list[i],dif_pred_iou_list[i]
                    if top_n==top+1:
                        break
                    if pred>0:
                        dif_iou[str(top)]+=1
                        break


            for gt,pred in zip(gt_iou_list,dif_pred_iou_list):
                #if gt==-1 or pred==-1:
                if gt==-1:
                    continue
                dif_gt.append(gt)
                dif_pred.append(pred)
            # Evaluation Softmax & KL Divergence
            if not dif_gt:
                zero_num+=1
                continue
            dif_gt=np.array(dif_gt)
            dif_pred=np.array(dif_pred)
            dif_sotf_gt=self.softmax(dif_gt)
            dif_sotf_pred=self.softmax(dif_pred)
            dif_result=self.KL_divergence(dif_sotf_gt,dif_sotf_pred)
            if np.isnan(dif_result):
                continue
            #acc_num+=result
            dif_acc_num+=dif_result
            list_id.append(img_id)


            if cfg.vis==True:
                key_x=dif_pose_out_img[:,0].copy()
                key_y=dif_pose_out_img[:,1].copy()
                key_z=gt_valid.copy()
                res_key=np.concatenate((key_x.reshape(1,17),key_y.reshape(1,17),key_z.reshape(1,17)),axis=0)
                bkg_image=cv2.imread(bkg_path)
                out_img=vis_keypoints_with_skeleton(bkg_image.copy(),res_key,self.coco_skeleton)

                cv2.imwrite('putting your path/output/'+cfg.save_folder+'/vis/'+str(img_id)+"_"+"test_res.jpg",out_img)

        return dif_acc_num,dif_iou,list_id,zero_num