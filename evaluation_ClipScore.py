from typing import Any
import torch
import numpy as np
import cv2
import os
import argparse
import torchvision.transforms.functional as vF

from tqdm import tqdm
from pycocotools.coco import COCO
from glob import glob
from torchmetrics.multimodal.clip_score import CLIPScore
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from einops import rearrange


BATCH_SIZE = 1

class CLIPDataSet(Dataset):
    def __init__(self, source_root:str, real_root:str, common_names:list, db):
        self.src_root = source_root
        self.real_root = real_root
        self.common_names = common_names
        self.db = db
    

    def __len__(self) -> int:
        return len(self.common_names)
    
    
    def __getitem__(self, index):
        # get propmt, real image and source image
        iid = int(self.common_names[index].split('_')[-1])
        aids = self.db.getAnnIds(imgIds=iid)
        cap_anns = self.db.loadAnns(aids)
        prompt = cap_anns[1]['caption']
        imgname = Path(self.db.imgs[iid]['file_name']).stem

        real_path = os.path.join(self.real_root, imgname+'.jpg')
        src_path = os.path.join(self.src_root, imgname+'.jpg')

        real_img = cv2.imread(real_path)
        src_img = cv2.imread(src_path)

        real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        real_img=cv2.resize(real_img,(512,512))
        src_img=cv2.resize(src_img,(512,512))
        real_img = vF.to_tensor(real_img)
        src_img = vF.to_tensor(src_img)
        return prompt, real_img*255, src_img*255


def compute_cs(src_root:str, real_root:str, commons:list, db):
    cs_dataset = CLIPDataSet(src_root, real_root, commons, db)
    cs_dataloader = DataLoader(cs_dataset, batch_size=BATCH_SIZE, shuffle=False)

    clip_score = CLIPScore('openai/clip-vit-base-patch16').cuda()
    cs = []

    for prompt, real_img, src_img in tqdm(cs_dataloader):
        src_img = torch.FloatTensor(src_img).cuda()
        score = clip_score(src_img, prompt[0])
        cs.append(score.cpu().detach().numpy())

    cs = np.array(cs)
    cs = np.mean(cs)
    
    return cs



parser = argparse.ArgumentParser()
MSCOCO_VAL_ROOT = "putting your path/MSCOCO/images/val2014"

if __name__ == "__main__":
    parser.add_argument("--src_root", type=str, required=True)
    args = parser.parse_args()
    source_root = args.src_root

    db_cap = COCO('putting your path/annotations/captions_val2014.json')

    real_paths = glob(os.path.join(MSCOCO_VAL_ROOT, "*.jpg"))
    source_paths = glob(os.path.join(source_root, "*.jpg"))

    real_names = set([Path(real_name).stem for real_name in real_paths])
    source_names = set([Path(src_name).stem for src_name in source_paths])

    common_names = list(real_names.intersection(source_names))


    cs = compute_cs(source_root, MSCOCO_VAL_ROOT, common_names, db_cap)

    print(f'CLIP Score : {cs}')


