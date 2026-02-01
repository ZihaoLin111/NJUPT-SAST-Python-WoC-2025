from calendar import c
from matplotlib.pylab import f
import torch
import torch.nn as nn
import torchvision
from model import DUB, DIDN
import tqdm
from pytorch_msssim import ssim
import datetime
import os
import json
import csv
import matplotlib.pyplot as plt
from model import DUB
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from itertools import count, cycle

class Basetest(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, dub_num=4):
        super(Basetest, self).__init__()

        self.entry = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU(num_parameters=out_channels, init=0.25)
        self.dub_blocks = nn.ModuleList([DUB(out_channels, out_channels) for _ in range(dub_num)])

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.prelu_21 = nn.PReLU(num_parameters=out_channels, init=0.25)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear((out_channels) * 8 * 8, 512) 
        self.prelu_22 = nn.PReLU(num_parameters=512, init=0.25)
        self.fc2 = nn.Linear(512, 128)
        self.prelu_23 = nn.PReLU(num_parameters=128, init=0.25)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):

        shared_feat = self.entry(x)
        shared_feat = self.prelu(shared_feat)
        for dub in self.dub_blocks:
            shared_feat = dub(shared_feat)

        out = self.pool1(shared_feat)
        out = self.conv(out)
        out = self.prelu_21(out)
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.prelu_22(out)
        out = self.fc2(out)
        out = self.prelu_23(out)
        out = self.classifier(out)
        return out
    
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

