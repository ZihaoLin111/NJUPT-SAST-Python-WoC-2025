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
from model import DUB, ResBlock
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from itertools import cycle
import random

class Basetest(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, num_classes=10):
        super(Basetest, self).__init__()

        # Shared Layers
        self.entry = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        self.dublayer1 = nn.Sequential(
            DUB(out_channels, out_channels),
            DUB(out_channels, out_channels),
            DUB(out_channels, out_channels)
        )

        # # For Task 1
        # self.dublayer2 = nn.Sequential(
        #     DUB(out_channels, out_channels)
        # )
        # self.exit = nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1)

        # For Task 2
        self.reslayer1 = nn.Sequential(
            ResBlock(out_channels, out_channels, stride=1),
            ResBlock(out_channels, out_channels, stride=1)
        )
        self.reslayer2 = nn.Sequential(
            ResBlock(out_channels, out_channels*2, stride=2),
            ResBlock(out_channels*2, out_channels*2, stride=1)
        )
        self.reslayer3 = nn.Sequential(
            ResBlock(out_channels*2, out_channels*4, stride=2),
            ResBlock(out_channels*4, out_channels*4, stride=1)
        )
        self.reslayer4 = nn.Sequential(
            ResBlock(out_channels*4, out_channels*8, stride=2),
            ResBlock(out_channels*8, out_channels*8, stride=1)
        )

        # # For Task ALL
        # self.reslayer11 = nn.Sequential(
        #     ResBlock(in_channels, out_channels, stride=1),
        #     ResBlock(out_channels, out_channels, stride=1)
        # )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flaten = nn.Flatten()
        self.fc = nn.Linear(out_channels*8, num_classes)

    def forward(self, x, task='all'):

        shared_feat = self.entry(x)
        # shared_feat = self.bn1(shared_feat)
        shared_feat = self.relu(shared_feat)
        shared_feat = self.dublayer1(shared_feat)

        out1, out2, out3 = None, None, None

        # if task == 'task1' or task == 'all':
        #     task1_feat = self.dublayer2(shared_feat)
        #     out1 = self.exit(task1_feat)

        if task == 'task2' or task == 'all':
            task2_feat = self.reslayer1(shared_feat)
            task2_feat = self.reslayer2(task2_feat)
            task2_feat = self.reslayer3(task2_feat)
            task2_feat = self.reslayer4(task2_feat)
            task2_feat = self.avgpool(task2_feat)
            task2_feat = self.flaten(task2_feat)
            out2 = self.fc(task2_feat)

        # if task == 'all':
        #     out3 = self.reslayer11(out1)
        #     out3 = self.reslayer2(out3)
        #     out3 = self.reslayer3(out3)
        #     out3 = self.reslayer4(out3)
        #     out3 = self.avgpool(out3)
        #     out3 = self.flaten(out3)
        #     out3 = self.fc(out3)
        
        return out2
    
CIFAR10C_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

CIFAR10_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

class PairedCIFAR10(Dataset):
    def __init__(self, clean_path, corrupted_path, transform=None, transformc=None, corruption_type="gaussian_noise", severity=1):
        self.transform = transform
        self.transformc = transformc

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

        if self.transformc:
            corrupted_img = self.transformc(corrupted_img)

        return corrupted_img, clean_img, label
    
clean_path = 'Task2/data'
corrupted_path = "MTL/data/CIFAR-10-C"

full_cifar10_dataset = PairedCIFAR10(
    clean_path=clean_path,
    corrupted_path=corrupted_path,
    transform=CIFAR10_transform,
    transformc=CIFAR10C_transform,
    corruption_type="gaussian_noise",
    severity=1
)

CIFAR10_train_size = int(0.9 * len(full_cifar10_dataset))
CIFAR10_val_size = len(full_cifar10_dataset) - CIFAR10_train_size
CIFAR10_train_dataset, CIFAR10_val_dataset = torch.utils.data.random_split(full_cifar10_dataset, [CIFAR10_train_size, CIFAR10_val_size])
CIFAR10_train_dataloader = DataLoader(CIFAR10_train_dataset, batch_size=128, shuffle=True, num_workers=4)
CIFAR10_val_dataloader = DataLoader(CIFAR10_val_dataset, batch_size=128, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Basetest().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

epochs = 20

for epoch in range(epochs):
    net.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for corrupted_imgs, clean_imgs, labels in tqdm.tqdm(CIFAR10_train_dataloader):
        corrupted_imgs, clean_imgs, labels = corrupted_imgs.to(device), clean_imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(corrupted_imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * corrupted_imgs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss /= len(CIFAR10_train_dataloader)
    train_acc = 100. * correct / total

    net.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for corrupted_imgs, clean_imgs, labels in tqdm.tqdm(CIFAR10_val_dataloader):
            corrupted_imgs, clean_imgs, labels = corrupted_imgs.to(device), clean_imgs.to(device), labels.to(device)

            outputs = net(corrupted_imgs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * corrupted_imgs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_acc = 100. * correct / total

    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

