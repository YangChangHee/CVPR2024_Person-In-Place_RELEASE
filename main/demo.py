import os
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import torch
import argparse
from tqdm import tqdm
import numpy as np
import torch.backends.cudnn as cudnn
from config import cfg
from base import Tester

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

    tester = Tester(args.test_epoch)
    tester._make_batch_generator()
    tester._make_model()
    eval_result = {}
    cur_sample_idx = 0
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
        tester._evaluate(out, cur_sample_idx)

if __name__ == "__main__":
    main()
