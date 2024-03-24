import os
import os.path as osp
import sys
import numpy as np
import datetime
import yaml
import shutil
import glob
from easydict import EasyDict as edict

class Config:
    trainset_2d = ['V_COCO']
    testset = 'V_COCO'
    using_gt=True
    ## model setting
    resnet_type = 152  # 50, 101, 152
    hrnet_type=18 # 18 34 48 64
    ## input, output
    input_img_shape = (256, 256)  #(256, 192)
    end_epoch = 20 #13 if 'FreiHAND' not in trainset_3d + trainset_2d + [testset] else 25
    lr = 1e-4
    pretrianed=False
    lr_backbone = 1e-4
    dis_backbone= 1e-2
    backbone='resnet'
    GCN_feature= False
    lr_dec_factor = 10
    train_batch_size = 64
    lr_dec_epoch = [15]
    ## testing config
    test_batch_size = 64
    filtering_mask=False
    num_thread = 32
    gpu_ids = '0'
    aggregation=False
    agg_num=20
    num_gpus = 1
    continue_train = False
    image_feature=False
    eval_thr=0.1
    gan_loss=False
    bbox_factor_param=1.
    dif_cs=32
    filtering_param=0.5
    obj_bbox_factor=1
    selection_skt='feature'
    filtering_feature=False
    diffusion=False
    obj_attention=False
    demo=False
    vis=False
    roi=False
    GCN_dim=2
    GCN_Model="GNN"
    Top=5
    MLP_fusion=False
    augmentation_ratio=50
    bbox_loss=False
    OCHuman=False
    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    # hongsuk choi style
    # KST = datetime.timezone(datetime.timedelta(hours=9))
    # save_folder = 'exp_' + str(datetime.datetime.now(tz=KST))[5:-16]
    save_folder = 'exp_' + str(datetime.datetime.now())[5:-10]
    save_folder = save_folder.replace(" ", "_")
    output_dir = osp.join(output_dir, save_folder)
    print('output dir: ', output_dir)

    #model_dir = osp.join(pretrianed_model_path, 'checkpoint')
    model_dir1 = osp.join(output_dir, 'checkpoint')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')

    def set_args(self, gpu_ids, continue_train=False,is_test=False,exp_dir=''):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        if not is_test:
            self.continue_train = continue_train
            if self.continue_train:
                if exp_dir:
                    checkpoints = sorted(glob.glob(osp.join(exp_dir, 'checkpoint') + '/*.pth.tar'), key=lambda x: int(x.split('_')[-1][:-8]))
                    shutil.copy(checkpoints[-1], osp.join(cfg.model_dir, checkpoints[-1].split('/')[-1]))

                else:
                    shutil.copy(osp.join(cfg.root_dir, 'tool', 'snapshot_0.pth.tar'), osp.join(cfg.model_dir, 'snapshot_0.pth.tar'))
        elif is_test and exp_dir:
            self.output_dir = exp_dir
            self.model_dir = osp.join(self.output_dir, 'checkpoint')
            self.vis_dir = osp.join(self.output_dir, 'vis')
            self.log_dir = osp.join(self.output_dir, 'log')
            self.result_dir = osp.join(self.output_dir, 'result')
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))
        

    def update(self, config_file):
        with open(config_file) as f:
            exp_config = edict(yaml.load(f))
            for k, v in exp_config.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
                else:
                    raise ValueError("{} not exist in config.py".format(k))

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
dataset_list = ['V_COCO','Object']
for i in range(len(dataset_list)):
    add_pypath(osp.join(cfg.data_dir, dataset_list[i]))
make_folder(cfg.model_dir1)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)