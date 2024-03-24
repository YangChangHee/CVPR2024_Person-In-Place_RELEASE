import os
import torch
import cv2
import numpy as np
import argparse
from glob import glob
from pathlib import Path
from torchmetrics.image.fid import FID
from tqdm import tqdm
from einops import rearrange, repeat
import torchvision.transforms.functional as vF
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--src_root", type=str, required=True)
MSCOCO_VAL_ROOT = "putting your path/val2014"
BATCH_SIZE = 256

class FID_Dataset(Dataset):
    def __init__(self, fake_root:str, real_root:str, common_names:list) ->None:
        super().__init__()
        self.fake_root = fake_root
        self.real_root = real_root
        self.common_names = common_names
    
    def __len__(self) -> int:
        return len(self.common_names)
    
    def __getitem__(self, index):
        real_path = os.path.join(self.real_root, self.common_names[index]) + ".jpg"
        fake_path = os.path.join(self.fake_root, self.common_names[index]) + ".jpg"

        real_img = cv2.imread(real_path)
        fake_img = cv2.imread(fake_path)

        real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
        fake_img = cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB)
        real_img = torch.from_numpy(real_img)
        fake_img = torch.from_numpy(fake_img)

        real_img = rearrange(real_img, 'w h c -> c w h')
        fake_img = rearrange(fake_img, 'w h c -> c w h')

        real_img = vF.resize(real_img, [512, 512])
        fake_img = vF.resize(fake_img, [512, 512])

        return real_img*255, fake_img*255
        

def compute_fid(fake_root: str, real_root:str, common_names:list):
    fid = FID(feature=2048).cuda()

    fid_dataset = FID_Dataset(fake_root, real_root, common_names)
    fid_loader = DataLoader(fid_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    for rimg, fimg in tqdm(fid_loader):
        rimg = rimg.to('cuda')
        fimg = fimg.to('cuda')

        fid.update(rimg, real=True)
        fid.update(fimg, real=False)

    return fid


if __name__ == "__main__":
    args = parser.parse_args()

    source_root = args.src_root

    src_paths = glob(os.path.join(source_root, "*.jpg"))
    real_paths = glob(os.path.join(MSCOCO_VAL_ROOT, "*.jpg"))

    src_names = set([Path(src_path).stem for src_path in src_paths])
    real_names = set([Path(real_path).stem for real_path in real_paths])

    commons = list(src_names.intersection(real_names))

    print(f'src length : {len(src_names)}, real length : {len(real_names)}, commons length : {len(commons)}')
    values = []

    # compute source metirc
    fid_src = compute_fid(source_root, MSCOCO_VAL_ROOT, commons)
    print(f'Source FID: {fid_src.compute()}')


    

