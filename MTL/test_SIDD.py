from matplotlib.pylab import f
import torch
import torch.nn as nn
import torchvision
import tqdm
from pytorch_msssim import ssim
import datetime
import os
import json
import matplotlib.pyplot as plt
from MTL import model
from model import DIDN, ResBlock, ResNet18, UncertaintyWeightingLoss, DnCNN
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from itertools import cycle
import random
from pcgrad import PCGrad

SIDD_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

class SIDD_Dataset_Resize(Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform
        self.data_path = os.path.join(data_path, 'Data')

        scene_file = os.path.join(data_path, 'Scene_Instances.txt')
        with open(scene_file, 'r') as f:
            self.scene_instances = [line.strip() for line in f]

        self.samples = []
        for folder in self.scene_instances:
            nosiy_path = os.path.join(self.data_path, folder, 'NOISY_SRGB_010.PNG')
            gt_path = os.path.join(self.data_path, folder, 'GT_SRGB_010.PNG')
    
            self.samples.append((nosiy_path, gt_path))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        noisy_path, gt_path = self.samples[idx]
        noisy_image = Image.open(noisy_path).convert('RGB')
        gt_image = Image.open(gt_path).convert('RGB')

        noisy_image = noisy_image.resize((512, 512))
        gt_image = gt_image.resize((512, 512))      

        if self.transform:
            noisy_image = self.transform(noisy_image)
            gt_image = self.transform(gt_image)

        return noisy_image, gt_image
    
class SIDD_Dataset_Crop(Dataset):
    def __init__(self, data_path, transform=None, crop_size=0, train=True):
        self.transform = transform
        self.data_path = os.path.join(data_path, 'Data')
        self.crop_size = crop_size
        self.train = train

        scene_file = os.path.join(data_path, 'Scene_Instances.txt')
        with open(scene_file, 'r') as f:
            self.scene_instances = [line.strip() for line in f]

        self.samples = []
        for folder in self.scene_instances:
            nosiy_path = os.path.join(self.data_path, folder, 'NOISY_SRGB_010.PNG')
            gt_path = os.path.join(self.data_path, folder, 'GT_SRGB_010.PNG')
    
            self.samples.append((nosiy_path, gt_path))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        noisy_path, gt_path = self.samples[idx]
        noisy_image = Image.open(noisy_path).convert('RGB')
        gt_image = Image.open(gt_path).convert('RGB')
        
        if self.crop_size > 0:
            if self.train:
                # 随机裁剪
                w, h = noisy_image.size
                i = random.randint(0, h - self.crop_size)
                j = random.randint(0, w - self.crop_size)
                noisy_image = noisy_image.crop((j, i, j + self.crop_size, i + self.crop_size))
                gt_image = gt_image.crop((j, i, j + self.crop_size, i + self.crop_size))
            else:
                # 中心裁剪
                w, h = noisy_image.size
                i = (h - self.crop_size) // 2
                j = (w - self.crop_size) // 2
                noisy_image = noisy_image.crop((j, i, j + self.crop_size, i + self.crop_size))
                gt_image = gt_image.crop((j, i, j + self.crop_size, i + self.crop_size))

        if self.transform:
            noisy_image = self.transform(noisy_image)
            gt_image = self.transform(gt_image)

        return noisy_image, gt_image
    
SIDD_data_path = "Task1/data/SIDD_Small_sRGB_Only"
SIDD_full_dataset = SIDD_Dataset_Crop(SIDD_data_path, transform=SIDD_transform, crop_size=32, train=True)
SIDD_train_size = int(0.8 * len(SIDD_full_dataset))
SIDD_val_size = len(SIDD_full_dataset) - SIDD_train_size
SIDD_train_dataset, SIDD_val_dataset = torch.utils.data.random_split(SIDD_full_dataset, [SIDD_train_size, SIDD_val_size])
SIDD_train_dataloader = DataLoader(SIDD_train_dataset, batch_size=16, shuffle=True, num_workers=4)
SIDD_val_dataloader = DataLoader(SIDD_val_dataset, batch_size=16, shuffle=False, num_workers=4)
SIDD_full_dataloader = DataLoader(SIDD_full_dataset, batch_size=16, shuffle=False, num_workers=4)

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = './saved_models/'


if __name__ == "__main__":

    DIDN_model = DIDN().to(device)
    DnCNN_model = DnCNN().to(device)

