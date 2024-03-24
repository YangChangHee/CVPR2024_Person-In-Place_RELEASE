import gradio as gr

import os
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import torch
import argparse
from tqdm import tqdm
import numpy as np
import torch.backends.cudnn as cudnn
from torch.nn.parallel.data_parallel import DataParallel
import torchvision.transforms as transforms
import sys
import cv2
import os.path as osp
import math
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from utils.preprocessing import process_bbox,generate_patch_image
from utils.transforms import denorm_joints
from utils.vis import vis_keypoints_with_skeleton
from model import get_model
from config import cfg
from base import Tester
from pycocotools.coco import COCO

import json

coco_joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 
                    'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 
                    'R_Ankle', 'Neck')
openpose=("Nose","Neck","R_Shoulder","R_Elbow","R_Wrist","L_Shoulder","L_Elbow",
          "L_Wrist","R_Hip","R_Knee","R_Ankle",
          "L_Hip","L_Knee","L_Ankle","R_Eye","L_Eye","R_Ear","L_Ear")

def add_neck(joint_coord, joints_name):
    lshoulder_idx = joints_name.index('L_Shoulder')
    rshoulder_idx = joints_name.index('R_Shoulder')
    neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
    neck[2] = joint_coord[lshoulder_idx, 2] * joint_coord[rshoulder_idx, 2]
    neck = neck.reshape(1,3)

    joint_coord = np.concatenate((joint_coord, neck))

    return joint_coord

def transform_joint_to_other_db(src_joint, src_name, dst_name):
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint

def draw_bodypose(canvas, subsets):
    H,W,C=canvas.shape
    stickwidth = 4

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for subset in subsets:
        subset = np.array(subset)
        for i in range(len(limbSeq)):
            index = subset[np.array(limbSeq[i]) - 1]
            if np.any(index[:, 2] < 0.8):
                continue
            Y=index.astype(int)[:,0]
            X=index.astype(int)[:,1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)
    for subset in subsets:
        subset = np.array(subset)
        for i,(x,y,z) in enumerate(subset):
            if z==0:
                continue
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
        canvas=cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB)
    return canvas

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0",dest='gpu_ids')
    parser.add_argument('--model_path', type=str, default='../output/SOTA/checkpoint/snapshot_30.pth.tar',help="this root is best pth root")
    parser.add_argument('--cfg', type=str, default='../assets/yaml/v-coco_diffusion_image_feature_demo.yml', help='experiment configure file name')

    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    return args

coco_skeleton = ( (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 6), (11, 12),(5,11),(6,12),(3,5),(4,6) )
        

args = parse_args()
cfg.set_args(args.gpu_ids, is_test=True)
cfg.render = True
cudnn.benchmark = True
if args.cfg:
    cfg.update(args.cfg)

transform = transforms.ToTensor()

model_path = args.model_path
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model('test')
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()
print("Success load model")

ROI_coordinates = {
    'x_temp': 0,
    'y_temp': 0,
    'x_new': 0,
    'y_new': 0,
    'x_obj_temp': 0,
    'y_obj_temp': 0,
    'x_obj_new': 0,
    'y_obj_new': 0,
    'clicks': 0,
}




def get_select_coordinates(img, evt: gr.SelectData):
    sections = []
    # update new coordinates
    ROI_coordinates['clicks'] += 1
    if ROI_coordinates['clicks'] % 4 == 1 or ROI_coordinates['clicks'] % 4 == 2:
        ROI_coordinates['x_temp'] = ROI_coordinates['x_new']
        ROI_coordinates['y_temp'] = ROI_coordinates['y_new']
        ROI_coordinates['x_new'] = evt.index[0]
        ROI_coordinates['y_new'] = evt.index[1]
        # compare start end coordinates
        x_start = ROI_coordinates['x_new'] if (ROI_coordinates['x_new'] < ROI_coordinates['x_temp']) else ROI_coordinates['x_temp']
        y_start = ROI_coordinates['y_new'] if (ROI_coordinates['y_new'] < ROI_coordinates['y_temp']) else ROI_coordinates['y_temp']
        x_end = ROI_coordinates['x_new'] if (ROI_coordinates['x_new'] > ROI_coordinates['x_temp']) else ROI_coordinates['x_temp']
        y_end = ROI_coordinates['y_new'] if (ROI_coordinates['y_new'] > ROI_coordinates['y_temp']) else ROI_coordinates['y_temp']
    else:
        ROI_coordinates['x_obj_temp'] = ROI_coordinates['x_obj_new']
        ROI_coordinates['y_obj_temp'] = ROI_coordinates['y_obj_new']
        ROI_coordinates['x_obj_new'] = evt.index[0]
        ROI_coordinates['y_obj_new'] = evt.index[1]
        # compare start end coordinates
        x_start = ROI_coordinates['x_new'] if (ROI_coordinates['x_new'] < ROI_coordinates['x_temp']) else ROI_coordinates['x_temp']
        y_start = ROI_coordinates['y_new'] if (ROI_coordinates['y_new'] < ROI_coordinates['y_temp']) else ROI_coordinates['y_temp']
        x_end = ROI_coordinates['x_new'] if (ROI_coordinates['x_new'] > ROI_coordinates['x_temp']) else ROI_coordinates['x_temp']
        y_end = ROI_coordinates['y_new'] if (ROI_coordinates['y_new'] > ROI_coordinates['y_temp']) else ROI_coordinates['y_temp']
        x_obj_start = ROI_coordinates['x_obj_new'] if (ROI_coordinates['x_obj_new'] < ROI_coordinates['x_obj_temp']) else ROI_coordinates['x_obj_temp']
        y_obj_start = ROI_coordinates['y_obj_new'] if (ROI_coordinates['y_obj_new'] < ROI_coordinates['y_obj_temp']) else ROI_coordinates['y_obj_temp']
        x_obj_end = ROI_coordinates['x_obj_new'] if (ROI_coordinates['x_obj_new'] > ROI_coordinates['x_obj_temp']) else ROI_coordinates['x_obj_temp']
        y_obj_end = ROI_coordinates['y_obj_new'] if (ROI_coordinates['y_obj_new'] > ROI_coordinates['y_obj_temp']) else ROI_coordinates['y_obj_temp']

    if ROI_coordinates['clicks'] % 4 == 2:
        # both start and end point get
        sections.append(((x_start, y_start, x_end, y_end), "Person BBOX"))

        return (img, sections)
    
    elif ROI_coordinates['clicks'] % 4 == 1:
        point_width = int(img.shape[0]*0.05)
        sections.append(((ROI_coordinates['x_new'], ROI_coordinates['y_new'], 
                          ROI_coordinates['x_new'] + point_width, ROI_coordinates['y_new'] + point_width),
                        "Select of Person BBOX"))
        return (img, sections)
    
    elif ROI_coordinates['clicks'] % 4 == 0:
        # both start and end point get
        sections.append(((x_start, y_start, x_end, y_end), "Person BBOX"))
        sections.append(((x_obj_start, y_obj_start, x_obj_end, y_obj_end), "Object BBOX"))
        p_bbox=[x_start,y_start,x_end-x_start,y_end-y_start]
        o_bbox=[x_obj_start, y_obj_start, x_obj_end, y_obj_end]
        input=img.copy()
        origin_img=img.copy()
        original_img_height, original_img_width = img.shape[:2]
        bbox_factor=math.sqrt(p_bbox[2]*p_bbox[3])/cfg.bbox_factor_param
        p_bbox[0]=p_bbox[0]-p_bbox[2]/2
        p_bbox[1]=p_bbox[1]-p_bbox[3]/2
        p_bbox[2]=p_bbox[2]*2
        p_bbox[3]=p_bbox[3]*2
        bbox = process_bbox(p_bbox, original_img_width, original_img_height)
        inp_img, img2bb_trans, bb2img_trans = generate_patch_image(input[:,:,::-1], bbox, 1.0, 0.0, False, cfg.input_img_shape)
        inp_img = transform(inp_img.astype(np.float32))/255
        inp_img = inp_img.cuda()[None,:,:,:]
        bbox_coord=np.array([[o_bbox[0],o_bbox[1]],[o_bbox[0],o_bbox[3]],[o_bbox[2],o_bbox[1]],[o_bbox[2],o_bbox[3]]])
        bbox_coord_img_xy1 = np.concatenate((bbox_coord[:,:2], np.ones_like(bbox_coord[:,:1])),1)
        bbox_coord[:,:2] = np.dot(img2bb_trans, bbox_coord_img_xy1.transpose(1,0)).transpose(1,0)
        bbox_coord=np.where(bbox_coord<0,0,bbox_coord)
        bbox_coord=np.where(bbox_coord>256,256,bbox_coord)
        x_max,x_min,y_max,y_min=np.max(bbox_coord[:,0]),np.min(bbox_coord[:,0]),np.max(bbox_coord[:,1]),np.min(bbox_coord[:,1])

        new_bbox=np.array([x_min,x_max,y_min,y_max])

        new_bbox=(new_bbox/4).astype('float32') # 64
        new_bbox_1=(new_bbox/4).astype('float32') # 16
        new_bbox_2=(new_bbox/8).astype('float32') # 8

        inputs = {'img': inp_img,
                    'feat_bbox_1':torch.from_numpy(new_bbox).unsqueeze(0).cuda(),
                    'feat_bbox_2':torch.from_numpy(new_bbox_1).unsqueeze(0).cuda(),
                    'feat_bbox_3':torch.from_numpy(new_bbox_2).unsqueeze(0).cuda()}
        targets = {'normalization_joint':torch.from_numpy(np.ones((17,2))).unsqueeze(0).cuda(),
                    "bb2img_trans":bb2img_trans,
                    "obj_bbox":None,
                    "p_bbox":None}
        meta_info = {}
        with torch.no_grad():
            out = model(inputs, targets, 'test')
        dif_pose_out_img = denorm_joints(out['output'][0].cpu().numpy().copy()*255, bb2img_trans)
        gt_valid=out['confidence'][0].cpu().numpy().copy()
        dif_pose_out_img = np.concatenate((dif_pose_out_img, gt_valid), axis=1)
        joint_img = add_neck(dif_pose_out_img,coco_joints_name)
        joint_img=transform_joint_to_other_db(joint_img,coco_joints_name,openpose)
        joint_img_list=[]
        joint_img_list.append(joint_img)
        #img1=np.zeros((original_img_height, original_img_width,3))
        img1=origin_img
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1=draw_bodypose(img1,joint_img_list)

        return (img1, sections)
    
    elif ROI_coordinates['clicks'] % 4 == 3:
        point_width = int(img.shape[0]*0.05)
        sections.append(((x_start, y_start, x_end, y_end), "Person BBOX"))
        sections.append(((ROI_coordinates['x_obj_new'], ROI_coordinates['y_obj_new'], 
                          ROI_coordinates['x_obj_new'] + point_width, ROI_coordinates['y_obj_new'] + point_width),
                        "Select of Object BBOX"))
        return (img, sections)


with gr.Blocks() as demo:
    with gr.Row():
        input_img = gr.Image(label="Click")
        img_output = gr.AnnotatedImage(label="ROI", 
                                       color_map={"Select of Person BBOX": "#9987FF",
                                                  "Person BBOX": "#f44336",
                                                  "Select of Object BBOX": "#fb87ff",
                                                  "Object BBOX": "#95ff87"})
    input_img.select(get_select_coordinates, input_img, img_output)

if __name__ == '__main__':
    demo.launch(inbrowser=True)
