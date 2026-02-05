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
from model import DIDN, ResBlock, ResNet18, UncertaintyWeightingLoss, DnCNN
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from itertools import count, cycle
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
SIDD_full_dataset = SIDD_Dataset_Crop(SIDD_data_path, transform=SIDD_transform, crop_size=64, train=False)
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
model_path = './saved_models/DIDN_train_freeze_20260205_140936_ep100_loss0.0207_best_psnr31.80dB_best_ssim0.95.pth'
DnCNN_path = './saved_models/DnCNN_train_freeze_20260205_123926_ep100_loss0.0406_best_psnr27.06dB_best_ssim0.87.pth'


if __name__ == "__main__":

    DIDN_model = DIDN().to(device)
    DnCNN_model = DnCNN().to(device)
    DIDN_model.load_state_dict(torch.load(model_path))
    DnCNN_model.load_state_dict(torch.load(DnCNN_path))
    DnCNN_model.eval()
    DIDN_model.eval()

    initialize_psnr = []
    initialize_ssim = []
    psnr_list = []
    ssim_list = []
    with torch.no_grad():
        for noisy_imgs, gt_imgs in tqdm.tqdm(SIDD_full_dataloader):
            noisy_imgs = noisy_imgs.to(device)
            gt_imgs = gt_imgs.to(device)

            batch_initial_psnr = psnr(noisy_imgs, gt_imgs).item()
            batch_initial_ssim = ssim(noisy_imgs, gt_imgs, data_range=1.0).item()
            initialize_psnr.append(batch_initial_psnr)
            initialize_ssim.append(batch_initial_ssim)

            denoised_imgs = DnCNN_model(noisy_imgs)

            batch_psnr = psnr(denoised_imgs, gt_imgs).item()
            batch_ssim = ssim(denoised_imgs, gt_imgs, data_range=1.0).item()

            psnr_list.append(batch_psnr)
            ssim_list.append(batch_ssim)
    avg_initial_psnr = sum(initialize_psnr) / len(initialize_psnr)
    avg_initial_ssim = sum(initialize_ssim) / len(initialize_ssim)
    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)

    print(f"Average PSNR: Initial{avg_initial_psnr:.4f} dB => Denoised{avg_psnr:.2f} dB")
    print(f"Average SSIM: Initial{avg_initial_ssim:.4f} => Denoised{avg_ssim:.4f}")

