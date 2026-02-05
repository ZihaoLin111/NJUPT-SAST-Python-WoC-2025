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
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
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
    severity=5
)

cifar10_loader = DataLoader(cifar10_testset, batch_size=128, shuffle=False, num_workers=4)

cifar10c_loader = DataLoader(cifar10c_dataset, batch_size=128, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    resnet_model = ResNet18(num_classes=10).to(device)
    ResNet_path = './saved_models/ResNet18_20260205_0422_ep150_train_best_acc_100.000val_best_acc88.400.pth'
    resnet_model.load_state_dict(torch.load(ResNet_path))
    resnet_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for corrupted_imgs, clean_imgs, labels in tqdm.tqdm(cifar10c_loader):
            corrupted_imgs = corrupted_imgs.to(device)
            labels = labels.to(device)
            outputs = resnet_model(corrupted_imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on CIFAR-10-C (Gaussian Noise, Severity 5): {100 * correct / total:.2f}%")

