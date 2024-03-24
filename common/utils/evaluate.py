# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1
    
def IoU(box1, box2):
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

def accuracy_bbox(img_shape,target_bbox, keypoints,list_joint, thr=0.05):
    '''
    img_shape = (height,width)
    target_bbox => obj_bbox [x1,y1,x2,y2]
    keypoints => [17,3] x,y,conf
    '''
    img_shape=np.array(img_shape)
    thr_height,thr_width=img_shape*thr
    eval_joint=[]
    for i in keypoints[list_joint]:
        if i[2]>0.9:
            eval_joint.append(i[:2])
    if not eval_joint:
        return 2
    eval_joint=np.array(eval_joint,dtype=np.uint32)
    list_x1=eval_joint[:,0]-thr_width/2
    list_x2=eval_joint[:,0]+thr_width/2
    list_y1=eval_joint[:,1]-thr_height/2
    list_y2=eval_joint[:,1]+thr_height/2
    for x1,x2,y1,y2 in zip(list_x1,list_x2,list_y1,list_y2):
        if IoU(target_bbox,[x1,x2,y1,y2])>0:
            return 1
    return 0
    


def accuracy(output, target, thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    h = 256
    w = 256
    norm = np.ones((target.shape[0], 2)) * np.array([h, w]) / 10

    dists = calc_dists(output, target, norm)


    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc

