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

class Res_DUB_MTL(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, num_classes=10):
        super(Res_DUB_MTL, self).__init__()

        # Shared Layers
        self.entry = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # For Task 1
        self.dublayer = nn.Sequential(
            DUB(out_channels, out_channels),
            DUB(out_channels, out_channels),
            DUB(out_channels, out_channels),
            DUB(out_channels, out_channels)
        )
        self.exit = nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1)

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

        # For Task ALL
        self.reslayer11 = nn.Sequential(
            ResBlock(in_channels, out_channels, stride=1),
            ResBlock(out_channels, out_channels, stride=1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flaten = nn.Flatten()
        self.fc = nn.Linear(out_channels*8, num_classes)

    def forward(self, x, task='all'):

        shared_feat = self.entry(x)
        # shared_feat = self.bn1(shared_feat)
        shared_feat = self.relu(shared_feat)

        out1, out2, out3 = None, None, None

        if task == 'task1' or task == 'all':
            task1_feat = self.dublayer(shared_feat)
            out1 = self.exit(task1_feat)

        if task == 'task2' or task == 'all':
            task2_feat = self.reslayer1(shared_feat)
            task2_feat = self.reslayer2(task2_feat)
            task2_feat = self.reslayer3(task2_feat)
            task2_feat = self.reslayer4(task2_feat)
            task2_feat = self.avgpool(task2_feat)
            task2_feat = self.flaten(task2_feat)
            out2 = self.fc(task2_feat)

        if task == 'all':
            out3 = self.reslayer11(out1)
            out3 = self.reslayer2(out3)
            out3 = self.reslayer3(out3)
            out3 = self.reslayer4(out3)
            out3 = self.avgpool(out3)
            out3 = self.flaten(out3)
            out3 = self.fc(out3)
        
        return out1, out2, out3
    
class UncertaintyWeightingLoss(nn.Module):
    def __init__(self, task_num):
        super(UncertaintyWeightingLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(task_num))

    def forward(self, losses):
        dtype = losses[0].dtype
        device = losses[0].device
        log_vars = self.log_vars.to(dtype).to(device)
        vars = torch.exp(log_vars) 
        total_loss = 0
        for i, loss in enumerate(losses):
            L = loss / (2 * vars[i]) + 0.5 * log_vars[i]
            total_loss += L
        return total_loss

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
SIDD_train_dataloader = DataLoader(SIDD_train_dataset, batch_size=32, shuffle=True, num_workers=4)
SIDD_val_dataloader = DataLoader(SIDD_val_dataset, batch_size=32, shuffle=False, num_workers=4)

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
CIFAR10_train_dataloader = DataLoader(CIFAR10_train_dataset, batch_size=256, shuffle=True, num_workers=4)
CIFAR10_val_dataloader = DataLoader(CIFAR10_val_dataset, batch_size=256, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Res_DUB_MTL().to(device)

criterion_task1 = nn.L1Loss()
criterion_task2 = nn.CrossEntropyLoss()
mtl_loss = UncertaintyWeightingLoss(task_num=4)
optimizer = torch.optim.Adam([
    {'params':net.parameters()}, 
    {'params':mtl_loss.parameters(), 'lr':1e-3}
    ],lr=1e-4)

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


epochs = 20
# task1_weight = 5.0
# task2_weight = 0.5

loader_len = max(len(SIDD_train_dataloader), len(CIFAR10_train_dataloader))
SIDD_Iter = cycle(SIDD_train_dataloader)
CIFAR10_Iter = cycle(CIFAR10_train_dataloader)

loss_list = []
psnr_1_list = []
psnr_2_list = []
accuracy_a_list = []
accuracy_b_list = []

for epoch in range(epochs):
    mtl_loss.train()
    net.train()
    running_loss = 0.0
    pbar = tqdm.tqdm(range(loader_len),desc=f"Epoch {epoch+1}/{epochs} Training",unit="batch")

    for step in pbar:
        noisy_imgs, clean_imgs = next(SIDD_Iter)
        noisy_class_imgs, clean_class_imgs, class_labels = next(CIFAR10_Iter)

        noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
        noisy_class_imgs, clean_class_imgs, class_labels = noisy_class_imgs.to(device), clean_class_imgs.to(device), class_labels.to(device)

        optimizer.zero_grad()

        denoised_imgs, _, _ = net(noisy_imgs, task='task1')
        loss_task1_a = criterion_task1(denoised_imgs, clean_imgs)
        # (loss_task1_a*task1_weight).backward()

        denoised_class_imgs, class_outputs_a, class_outputs_b = net(noisy_class_imgs, task='all')
        loss_task1_b = criterion_task1(denoised_class_imgs, clean_class_imgs)
        # (loss_task1_b*task1_weight).backward()
        loss_task2_a = criterion_task2(class_outputs_a, class_labels)
        loss_task2_b = criterion_task2(class_outputs_b, class_labels)
        # (loss_task2*task2_weight).backward()

        total_loss = mtl_loss([loss_task1_a, loss_task1_b, loss_task2_a, loss_task2_b])
        total_loss.backward()

        optimizer.step()
        running_loss += total_loss.item()
        avg_loss = running_loss / (step + 1)
        pbar.set_description(f"Epoch {epoch+1}/{epochs} Training - Loss: {avg_loss:.4f}")
        pbar.update(1)

    pbar.close()

    mtl_loss.eval()
    net.eval()
    initial_psnr_1 = 0.0
    initial_ssim_1 = 0.0
    initial_psnr_2 = 0.0
    initial_ssim_2 = 0.0
    total_psnr_1 = 0.0
    total_ssim_1 = 0.0
    total_psnr_2 = 0.0
    total_ssim_2 = 0.0
    correct_count_a = 0
    correct_count_b = 0
    count_1 = 0
    count_2 = 0
    with torch.no_grad():
        for noisy_imgs, clean_imgs in SIDD_val_dataloader:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            denoised_imgs, _, _ = net(noisy_imgs, task='task1')

            for i in range(noisy_imgs.size(0)):
                initial_psnr_1 += psnr(noisy_imgs[i], clean_imgs[i]).item()
                initial_ssim_1 += ssim(noisy_imgs[i].unsqueeze(0), clean_imgs[i].unsqueeze(0), data_range=1.0, size_average=True).item()
                total_psnr_1 += psnr(denoised_imgs[i], clean_imgs[i]).item()
                total_ssim_1 += ssim(denoised_imgs[i].unsqueeze(0), clean_imgs[i].unsqueeze(0), data_range=1.0, size_average=True).item()
                count_1 += 1

        for noisy_class_imgs, clean_class_imgs, class_labels in CIFAR10_val_dataloader:
            noisy_class_imgs, clean_class_imgs, class_labels = noisy_class_imgs.to(device), clean_class_imgs.to(device), class_labels.to(device)
            denoised_class_imgs, class_outputs_a, class_outputs_b = net(noisy_class_imgs, task='all')

            for i in range(noisy_class_imgs.size(0)):
                initial_psnr_2 += psnr(noisy_class_imgs[i], clean_class_imgs[i]).item()
                initial_ssim_2 += ssim(noisy_class_imgs[i].unsqueeze(0), clean_class_imgs[i].unsqueeze(0), data_range=1.0, size_average=True).item()
                total_psnr_2 += psnr(denoised_class_imgs[i], clean_class_imgs[i]).item()
                total_ssim_2 += ssim(denoised_class_imgs[i].unsqueeze(0), clean_class_imgs[i].unsqueeze(0), data_range=1.0, size_average=True).item()
                _, predicted_a = torch.max(class_outputs_a[i], 0)
                _, predicted_b = torch.max(class_outputs_b[i], 0)
                if predicted_a.item() == class_labels[i].item():
                    correct_count_a += 1
                
                if predicted_b.item() == class_labels[i].item():
                    correct_count_b += 1
                
                count_2 += 1
    
    avg_initial_psnr_1 = initial_psnr_1 / count_1
    avg_initial_ssim_1 = initial_ssim_1 / count_1
    avg_total_psnr_1 = total_psnr_1 / count_1
    avg_total_ssim_1 = total_ssim_1 / count_1

    avg_initial_psnr_2 = initial_psnr_2 / count_2
    avg_initial_ssim_2 = initial_ssim_2 / count_2
    avg_total_psnr_2 = total_psnr_2 / count_2
    avg_total_ssim_2 = total_ssim_2 / count_2

    accuracy_a = correct_count_a / count_2 * 100.0
    accuracy_b = correct_count_b / count_2 * 100.0

    avg_loss = running_loss / loader_len
    loss_list.append(avg_loss)
    psnr_1_list.append(avg_total_psnr_1)
    psnr_2_list.append(avg_total_psnr_2)
    accuracy_a_list.append(accuracy_a)
    accuracy_b_list.append(accuracy_b)
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}")
    print(f"Task 1 - Initial PSNR: {avg_initial_psnr_1:.4f}, SSIM: {avg_initial_ssim_1:.4f} => Denoised PSNR: {avg_total_psnr_1:.4f}, SSIM: {avg_total_ssim_1:.4f}")
    print(f"Task 2 - Initial PSNR: {avg_initial_psnr_2:.4f}, SSIM: {avg_initial_ssim_2:.4f} => Denoised PSNR: {avg_total_psnr_2:.4f}, SSIM: {avg_total_ssim_2:.4f}")
    print(f"Task 2 - Classification Accuracy: {accuracy_a:.2f}%")
    print(f"Task 2 - Classification Accuracy: {accuracy_b:.2f}% (from denoised output)")


# 保存模型
save_folder = './saved_models'
model_name = 'Res_DUB_MTL'
epoch = epochs
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

os.makedirs(save_folder, exist_ok=True)
filename = f"{model_name}_{timestamp}_ep{epoch}_loss{loss_list[-1]:.4f}_psnr1{psnr_1_list[-1]:.2f}dB_acc{accuracy_b_list[-1]:.2f}.pth"
save_path = os.path.join(save_folder, filename)
torch.save(net.state_dict(), save_path)

print(f"模型参数以保存至: {save_path}")

run_log = {
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "MTL",
    "epochs": epochs,
    "optimizer": "Adam",
    "lr": 1e-4,
    "train_loss_last": loss_list[-1],
    "val_psnr_last": psnr_1_list[-1],
}

# 保存JSONL
jsonl_path = 'runs.jsonl'
with open(jsonl_path, 'a', encoding="utf-8") as f:
    f.write(json.dumps(run_log) + '\n')

# 绘制并保存曲线
# 损失曲线
plt.figure()
plt.plot(range(1, epochs + 1), loss_list, marker='o')
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.savefig(f'loss_curve_{timestamp}.png')
plt.close()

# psnr曲线
plt.figure()
plt.plot(range(1, epochs + 1), psnr_1_list, marker='o', label='Task 1 PSNR')
plt.plot(range(1, epochs + 1), psnr_2_list, marker='o', label='Task 2 PSNR')
plt.title('Validation PSNR Curve')
plt.xlabel('Epochs')
plt.ylabel('PSNR (dB)')
plt.legend()
plt.grid()
plt.savefig(f'psnr_curve_{timestamp}.png')
plt.close()

# 准确率曲线
plt.figure()
plt.plot(range(1, epochs + 1), accuracy_a_list, marker='o', label='Accuracy from noisy input')
plt.plot(range(1, epochs + 1), accuracy_b_list, marker='o', label='Accuracy from denoised output')
plt.title('Validation Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid()
plt.savefig(f'accuracy_curve_{timestamp}.png')
plt.close()