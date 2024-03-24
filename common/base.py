import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from timer import Timer
from logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from config import cfg
from model import get_model
from dataset import MultipleDatasets
dataset_list = ['V_COCO','Object']
for i in range(len(dataset_list)):
    exec('from ' + dataset_list[i] + ' import ' + dataset_list[i])


class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return


class Trainer(Base):
    def __init__(self):
        super(Trainer, self).__init__(log_name = 'train_logs.txt')

    def get_optimizer(self, model):
        if cfg.GCN_feature==True:
            for p in model.module.backbone.parameters():
                p.requires_grad = False
            for p in model.module.feat2pose.parameters():
                p.requires_grad = False
            for p in model.module.CrossAttention.parameters():
                p.requires_grad = False
            for p in model.module.conv1_1.parameters():
                p.requires_grad = False
            for p in model.module.conv1_2.parameters():
                p.requires_grad = False
            for p in model.module.conv1_3.parameters():
                p.requires_grad = False
            if cfg.GCN_Model=="GAT":
                optimizer = torch.optim.Adam([
                    # GCN feature true
                    {'params': model.module.GCN_module.parameters()},
                ],lr=cfg.lr)
            else:
                if cfg.obj_attention==True or cfg.MLP_fusion==True:
                    optimizer = torch.optim.Adam([
                        # GCN feature true
                        {'params': model.module.GCN_module.parameters()},
                        {'params': model.module.GCN_linear.parameters()},
                        {'params': model.module.OBJattention.parameters()},
                        {'params': model.module.conv1_1.parameters()},
                        {'params': model.module.conv1_2.parameters()},
                        {'params': model.module.conv1_3.parameters()},
                    ],lr=cfg.lr)
                else:
                    optimizer = torch.optim.Adam([
                        # GCN feature true
                        {'params': model.module.GCN_module.parameters()},
                        {'params': model.module.GCN_linear.parameters()},
                        #{'params': model.module.OBJattention.parameters()},
                        {'params': model.module.conv1_1.parameters()},
                        {'params': model.module.conv1_2.parameters()},
                        {'params': model.module.conv1_3.parameters()},
                    ],lr=cfg.lr)
        else:
            if cfg.roi==True and cfg.obj_attention==False:
                print("optim_1")
                optimizer = torch.optim.Adam([
                    {'params': model.module.backbone.parameters(), 'lr': cfg.lr_backbone},
                    {'params': model.module.feat2pose.parameters()},
                    {'params': model.module.conv1_3.parameters()},
                    #{'params': model.module.GCN_module.parameters()},
                    #{'params': model.module.GCN_linear.parameters()},
                    #{'params': model.module.GNN_Linear.parameters()},
                    #{'params': model.module.conv2048_256.parameters()},
                ],
                lr=cfg.lr)
            elif cfg.obj_attention==True:
                print("optim_2")
                optimizer = torch.optim.Adam([
                    {'params': model.module.backbone.parameters(), 'lr': cfg.lr_backbone},
                    {'params': model.module.feat2pose.parameters()},
                    {'params': model.module.CrossAttention4.parameters()},
                    {'params': model.module.conv1_3.parameters()},
                ],
                lr=cfg.lr)
            elif cfg.diffusion==True:
                if cfg.aggregation==False:
                    print("optim_4_1")
                    optimizer = torch.optim.Adam([
                        {'params': model.module.backbone.parameters(), 'lr': cfg.lr_backbone},
                        {'params': model.module.feat2pose.parameters(), 'lr': cfg.lr_backbone},
                        {'params': model.module.conv1_3.parameters()},
                        {'params': model.module.dif_train.parameters()},
                    ],
                    lr=cfg.lr)
                else:
                    print("optim_4_2")
                    optimizer = torch.optim.Adam([
                        {'params': model.module.backbone.parameters(), 'lr': cfg.lr_backbone},
                        {'params': model.module.feat2pose.parameters(), 'lr': cfg.lr_backbone},
                        {'params': model.module.conv1_3.parameters()},
                        {'params': model.module.dif_train.parameters()},
                        #{'params': model.module.aggregation.parameters()},
                    ],
                    lr=cfg.lr)

            else:
                print("optim_3")
                optimizer = torch.optim.Adam([
                    {'params': model.module.backbone.parameters(), 'lr': cfg.lr_backbone},
                    {'params': model.module.feat2pose.parameters()},
                ],
                lr=cfg.lr)

        print('The parameters of backbone, pose2feat, position_net, rotation_net, are added to the optimizer.')

        return optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir1,'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(cfg.model_dir,'*.pth.tar'))
        #model_file_list = glob.glob(osp.join(cfg.model_dir))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
        #cur_epoch=0
        print("load {} epoch pretrained model !!".format(cur_epoch))
        ckpt_path = osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
        ckpt = torch.load(ckpt_path) 
        #start_epoch = ckpt['epoch'] + 1
        start_epoch=0
#        print(ckpt['network'].module.backbone.parameters())
        model.load_state_dict(ckpt['network'], strict=False)
        #optimizer.load_state_dict(ckpt['optimizer'])

        self.logger.info('Load checkpoint from {}'.format(ckpt_path))
        return start_epoch, model, optimizer

    def set_lr(self, epoch):
        for e in cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < cfg.lr_dec_epoch[-1]:
            idx = cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr
    
    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        trainset2d_loader = []
        for i in range(len(cfg.trainset_2d)):
            trainset2d_loader.append(eval(cfg.trainset_2d[i])(transforms.ToTensor(), "train"))


        trainset_loader = MultipleDatasets(trainset2d_loader, make_same_len=False)
            
        self.itr_per_epoch = math.ceil(len(trainset_loader) / cfg.num_gpus / cfg.train_batch_size)
        self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.num_gpus*cfg.train_batch_size, shuffle=True, num_workers=cfg.num_thread, pin_memory=True)

    def _make_model(self):
        # prepare network
        if cfg.backbone=='resnet':
            self.logger.info("Creating ResNet Type : {} backone...".format(cfg.resnet_type))
        if cfg.backbone=='hrnet':
            self.logger.info("Creating HRNet Type : {} backone...".format(cfg.hrnet_type))
        model = get_model('train')
        model = DataParallel(model).cuda()
        optimizer = self.get_optimizer(model)
        if cfg.pretrianed:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
            optimizer = self.get_optimizer(model)
        else:
            start_epoch = 0
        #model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer


class Tester(Base):
    def __init__(self, test_epoch):
        self.test_epoch = int(test_epoch)
        super(Tester, self).__init__(log_name = 'test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        testset_loader = eval(cfg.testset)(transforms.ToTensor(), "test")
        batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus*cfg.test_batch_size, shuffle=False, num_workers=cfg.num_thread, pin_memory=True)
        
        self.testset = testset_loader
        self.batch_generator = batch_generator

    def _make_model(self):
        #if cfg.smplify==False:
        model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        #else:
        #    model_path = cfg.distillation_module_path
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))
        
        # prepare network
        self.logger.info("Creating graph...")
        model = get_model('test')
        model = DataParallel(model).cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'], strict=False)
        model.eval()

        self.model = model

    def _evaluate(self, outs, cur_sample_idx):
        eval_result = self.testset.evaluate(outs, cur_sample_idx)
        return eval_result

    def _print_eval_result(self, eval_result):
        self.testset.print_eval_result(eval_result)
