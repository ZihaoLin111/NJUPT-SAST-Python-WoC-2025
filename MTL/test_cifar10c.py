from unittest import loader

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



cifar10c_dataset_1 = PairedCIFAR10(
    clean_path, 
    corrupted_path, 
    transform=CIFAR10_val_transform, 
    corruption_type="gaussian_noise", 
    severity=1
)
cifar10c_dataset_2 = PairedCIFAR10(
    clean_path, 
    corrupted_path, 
    transform=CIFAR10_val_transform, 
    corruption_type="gaussian_noise", 
    severity=2
)
cifar10c_dataset_3 = PairedCIFAR10(
    clean_path, 
    corrupted_path, 
    transform=CIFAR10_val_transform, 
    corruption_type="gaussian_noise", 
    severity=3
)
cifar10c_dataset_4 = PairedCIFAR10(
    clean_path, 
    corrupted_path, 
    transform=CIFAR10_val_transform, 
    corruption_type="gaussian_noise", 
    severity=4
)
cifar10c_dataset_5 = PairedCIFAR10(
    clean_path, 
    corrupted_path, 
    transform=CIFAR10_val_transform, 
    corruption_type="gaussian_noise", 
    severity=5
)


CIFAR10_train_size = int(0.8 * len(cifar10_testset))
CIFAR10_val_size = len(cifar10_testset) - CIFAR10_train_size
_,cifar10_test_0 = torch.utils.data.random_split(cifar10_testset, [CIFAR10_train_size, CIFAR10_val_size],generator=torch.Generator().manual_seed(42))
_,cifar10_test_1 = torch.utils.data.random_split(cifar10c_dataset_1, [CIFAR10_train_size, CIFAR10_val_size],generator=torch.Generator().manual_seed(42))
_,cifar10_test_2 = torch.utils.data.random_split(cifar10c_dataset_2, [CIFAR10_train_size, CIFAR10_val_size],generator=torch.Generator().manual_seed(42))
_,cifar10_test_3 = torch.utils.data.random_split(cifar10c_dataset_3, [CIFAR10_train_size, CIFAR10_val_size],generator=torch.Generator().manual_seed(42))
_,cifar10_test_4 = torch.utils.data.random_split(cifar10c_dataset_4, [CIFAR10_train_size, CIFAR10_val_size],generator=torch.Generator().manual_seed(42))
_,cifar10_test_5 = torch.utils.data.random_split(cifar10c_dataset_5, [CIFAR10_train_size, CIFAR10_val_size],generator=torch.Generator().manual_seed(42))


cifar10_loader = DataLoader(cifar10_test_0, batch_size=128, shuffle=False, num_workers=4)
cifar10c1_loader = DataLoader(cifar10_test_1, batch_size=128, shuffle=False, num_workers=4)
cifar10c2_loader = DataLoader(cifar10_test_2, batch_size=128, shuffle=False, num_workers=4)
cifar10c3_loader = DataLoader(cifar10_test_3, batch_size=128, shuffle=False, num_workers=4)
cifar10c4_loader = DataLoader(cifar10_test_4, batch_size=128, shuffle=False, num_workers=4)
cifar10c5_loader = DataLoader(cifar10_test_5, batch_size=128, shuffle=False, num_workers=4)

loaders = [cifar10c1_loader, cifar10c2_loader, cifar10c3_loader, cifar10c4_loader, cifar10c5_loader]

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
    DnCNN_path = './saved_models/DnCNN_train_freeze_ResNet18_uncertain_loss_20260221_233117_ep150_loss-2.3083_best_psnr22.51dB_best_ssim0.73.pth'
    DnCNN_model.load_state_dict(torch.load(DnCNN_path))
    DnCNN_model.eval()
    resnet_model.eval()

    for i in range(5):
        print(f"Evaluating on Dataset {i+1}...")
        Loader = loaders[i]
        correct = 0
        total = 0
        initial_psnr = 0
        initial_ssim = 0
        final_psnr = 0
        final_ssim = 0
        count_ = 0
        with torch.no_grad():
            for corrupted_imgs, clean_imgs, labels in tqdm.tqdm(Loader):
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
        print(f"Accuracy on CIFAR-10-C (Gaussian Noise, Severity {i+1}): {100 * correct / total:.2f}%")

