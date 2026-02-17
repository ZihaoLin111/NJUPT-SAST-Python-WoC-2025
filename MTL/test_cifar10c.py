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

class PairedCIFAR10(Dataset):
    def __init__(self, clean_path, corrupted_path, transform=None, corruption_type="gaussian_noise", severity=1):
        self.transform = transform

        self.clean_dataset = torchvision.datasets.CIFAR10(
            root=clean_path,
            train=False,
            download=True,
            transform=None
        )

        corrupted_data_path = os.path.join(corrupted_path, corruption_type + ".npy")
        full_corrupted_data = np.load(corrupted_data_path)

        num_per_severity = 10000
        start_idx = (severity - 1) * num_per_severity
        end_idx = severity * num_per_severity
        self.corrupted_data = full_corrupted_data[start_idx:end_idx]

    def __len__(self):
        return len(self.clean_dataset)
    def __getitem__(self, idx):
        clean_img, label = self.clean_dataset[idx]
        corrupted_img = self.corrupted_data[idx]

        if self.transform:
            clean_img = self.transform(clean_img)
            corrupted_img = self.transform(corrupted_img)

        return corrupted_img, clean_img, label
    
CIFAR10_val_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
    
cifar10_testset = torchvision.datasets.CIFAR10(
    root='Task2/data',
    train=False,
    download=True,
    transform=CIFAR10_val_transform
)
    

clean_path = 'Task2/data'
corrupted_path = "MTL/data/CIFAR-10-C"



cifar10c_dataset = PairedCIFAR10(
    clean_path, 
    corrupted_path, 
    transform=CIFAR10_val_transform, 
    corruption_type="gaussian_noise", 
    severity=3
)

cifar10_loader = DataLoader(cifar10_testset, batch_size=128, shuffle=False, num_workers=4)

cifar10c_loader = DataLoader(cifar10c_dataset, batch_size=128, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))



if __name__ == "__main__":
    resnet_model = ResNet18(num_classes=10).to(device)
    DnCNN_model = DnCNN().to(device)
    ResNet_path = './saved_models/ResNet18_20260205_0422_ep150_train_best_acc_100.000val_best_acc88.400.pth'
    resnet_model.load_state_dict(torch.load(ResNet_path))
    DnCNN_path = './saved_models/DnCNN_train_freeze_20260207_003904_ep150_loss0.2808_best_psnr22.92dB_best_ssim0.74.pth'
    DnCNN_model.load_state_dict(torch.load(DnCNN_path))
    DnCNN_model.eval()
    resnet_model.eval()
    correct = 0
    total = 0
    initial_psnr = 0
    initial_ssim = 0
    final_psnr = 0
    final_ssim = 0
    count_ = 0
    with torch.no_grad():
        for corrupted_imgs, clean_imgs, labels in tqdm.tqdm(cifar10c_loader):
            corrupted_imgs = corrupted_imgs.to(device)
            labels = labels.to(device)

            denoised_imgs = DnCNN_model(corrupted_imgs)
            classify_input = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)).to(device)(denoised_imgs)
            outputs = resnet_model(classify_input)

            for i in range(corrupted_imgs.size(0)):
                initial_psnr += psnr(corrupted_imgs[i], clean_imgs[i].to(device)).item()
                initial_ssim += ssim(corrupted_imgs[i].unsqueeze(0), clean_imgs[i].unsqueeze(0).to(device)).item()
                final_psnr += psnr(denoised_imgs[i], clean_imgs[i].to(device)).item()
                final_ssim += ssim(denoised_imgs[i].unsqueeze(0), clean_imgs[i].unsqueeze(0).to(device)).item()
                count_ += 1

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"PSNR: Initial: {initial_psnr / count_:.2f} dB => Final: {final_psnr / count_:.2f} dB")
    print(f"SSIM: Initial: {initial_ssim / count_:.4f} => Final: {final_ssim / count_:.4f}")
    print(f"Accuracy on CIFAR-10-C (Gaussian Noise, Severity 3): {100 * correct / total:.2f}%")

