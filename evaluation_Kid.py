import torch
import os
import cv2
import torchvision.transforms.functional as vF
from glob import glob
from torchmetrics.image.kid import KID
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from einops import rearrange

import argparse


BATCH_SIZE=512

class KidDataset(Dataset):
    def __init__(self, fake_root:str, real_root:str):
        super().__init__()
        self.fake_root = fake_root
        self.real_root = real_root
        self.fake_paths = glob(os.path.join(fake_root, '*.jpg'))
        self.real_paths = glob(os.path.join(real_root, '*.jpg'))
        self.commons = set([Path(fpath).stem for fpath in self.fake_paths]).intersection(set([Path(rpath).stem for rpath in self.real_paths]))
        self.commons = list(self.commons)
    
    
    def __len__(self):
        return len(self.commons)


    def __getitem__(self, index):
        real_path = os.path.join(self.real_root, self.commons[index] + ".jpg")
        fake_path = os.path.join(self.fake_root, self.commons[index] + ".jpg")

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
    

def compute_kid(fake_root:str, real_root:str):
    kid = KID().cuda()
    kid_dataset = KidDataset(fake_root, real_root)
    kid_loader = DataLoader(kid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for rimg, fimg in tqdm(kid_loader):
        rimg = rimg.cuda()
        fimg = fimg.cuda()

        kid.update(rimg, real=True)
        kid.update(fimg, real=False)

    return kid.compute()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake_root",type=str, required=True)
    parser.add_argument("--real_root",type=str, required=True)
    args = parser.parse_args()


    mean_kid, std_kid = compute_kid(args.fake_root, args.real_root)

    print(f'Mean KID : {mean_kid}, STD Kid : {std_kid}')




