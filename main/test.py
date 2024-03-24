import os
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import torch
import argparse
from tqdm import tqdm
import numpy as np
import torch.backends.cudnn as cudnn
from config import cfg
from base import Tester
import random
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--exp_dir', type=str, default='')
    parser.add_argument('--cfg', type=str, default='', help='experiment configure file name')

    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():
    args = parse_args()
    cfg.set_args(args.gpu_ids, is_test=True, exp_dir=args.exp_dir)
    cudnn.benchmark = True
    if args.cfg:
        cfg.update(args.cfg)

    manualSeed = 1
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    tester = Tester(args.test_epoch)
    tester._make_batch_generator()
    tester._make_model()
    all_acc=0
    dif_all_acc=0
    zero_num=0
    cur_sample_idx = 0
    if cfg.diffusion==True:
        tester.model.module.dif_test.load_state_dict(tester.model.module.dif_train.state_dict(), strict=False)
    tester.model.eval()
    annot_dict={}
    all_fwd_iou={"1":0,"3":0,"5":0}
    all_dif_iou={"1":0,"3":0,"5":0}
    if cfg.demo==True:
        demo_json={}
    for itr, (inputs,targets) in enumerate(tqdm(tester.batch_generator)):
        #for k in joint_error:
        #    oks_error+=k
        # forward
        with torch.no_grad():
            out = tester.model(inputs, targets, 'test')
    
        # save output
        out = {k: v.cpu().numpy() for k,v in out.items()}
        for k,v in out.items(): batch_size = out[k].shape[0]
        out = [{k: v[bid] for k,v in out.items()} for bid in range(batch_size)]
        
        # demo
        if cfg.demo==False:
            if cfg.gan_loss==False:
                dif_b_acc,dif_iou,_,z_num=tester._evaluate(out, cur_sample_idx)
            else:
                b_acc,z_num=tester._evaluate(out, cur_sample_idx)
        else:
            l_pose_out_img,l_img_name,l_input_object_bbox,l_input_person_bbox,l_aid=tester._evaluate(out, cur_sample_idx)
            for (pose_out_img,img_name,obj_bbox,person_bbox,aid) in zip(l_pose_out_img,l_img_name,l_input_object_bbox,l_input_person_bbox,l_aid):
                if img_name in demo_json:
                    demo_json[img_name][aid]={
                        "img_name":img_name,
                        "obj_bbox":obj_bbox,
                        "person_bbox":person_bbox,
                        "pose_out_img":pose_out_img.tolist()
                    }
                else:
                    demo_json[img_name]={}
                    demo_json[img_name][aid]={
                        "img_name":img_name,
                        "obj_bbox":obj_bbox,
                        "person_bbox":person_bbox,
                        "pose_out_img":pose_out_img.tolist()
                    }
        if cfg.demo==False:
            for i in ["1","3","5"]:
                all_dif_iou[i]+=dif_iou[i]
            dif_all_acc+=dif_b_acc
            zero_num+=z_num
        cur_sample_idx += len(out)
    if cfg.demo==False:
        print("pass_image : ",zero_num)
        dif_mean_acc=dif_all_acc/(cur_sample_idx-zero_num)
        dif_current_acc=dif_mean_acc*1
        dif_top_1=all_dif_iou['1']/(cur_sample_idx-zero_num)*1
        dif_top_3=all_dif_iou['3']/(cur_sample_idx-zero_num)*1
        dif_top_5=all_dif_iou['5']/(cur_sample_idx-zero_num)*1
        print("diffusion keypoints : ",dif_current_acc)
        print("=============================================================")
        print("|  Top  |  dif  |  1  |  3  |  5  |")
        print("|       |{}|{}|{}|".format(round(dif_top_1,3),round(dif_top_3,3),round(dif_top_5,3)))
        print("=============================================================")
    if cfg.demo==True:
        import json
        with open("demo_result.json","w") as f:
            json.dump(demo_json,f)
    else:
        import json
        with open("test.json","w") as f:
            json.dump(annot_dict,f)

if __name__ == "__main__":
    main()