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

class MTL(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, dub_num=4):
        super(MTL, self).__init__()

        # Shared Layers
        self.entry = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU(num_parameters=out_channels, init=0.25)
        self.dub_blocks = nn.ModuleList([DUB(out_channels, out_channels) for _ in range(dub_num)])

        # For Task 1
        self.exit = nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1)

        # For Task 2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.prelu_21 = nn.PReLU(num_parameters=out_channels, init=0.25)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear((out_channels) * 8 * 8, 512) 
        self.prelu_22 = nn.PReLU(num_parameters=512, init=0.25)
        self.fc2 = nn.Linear(512, 128)
        self.prelu_23 = nn.PReLU(num_parameters=128, init=0.25)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x, task='all'):

        shared_feat = self.entry(x)
        shared_feat = self.prelu(shared_feat)
        for dub in self.dub_blocks:
            shared_feat = dub(shared_feat)

        out1, out2 = None, None

        if task == 'task1' or task == 'all':
            out1 = self.exit(shared_feat)

        if task == 'task2' or task == 'all':
            out2 = self.pool1(shared_feat)
            out2 = self.prelu_21(self.conv(out2))
            out2 = self.pool2(out2)
            out2 = out2.view(out2.size(0), -1)
            out2 = self.prelu_22(self.fc1(out2))
            out2 = self.prelu_23(self.fc2(out2))
            out2 = self.classifier(out2)
        
        return out1, out2
    
SIDD_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

class SIDD_Dataset(Dataset):
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

SIDD_data_path = "Task1/data/SIDD_Small_sRGB_Only"
SIDD_full_dataset = SIDD_Dataset(SIDD_data_path, transform=SIDD_transform)
SIDD_train_size = int(0.8 * len(SIDD_full_dataset))
SIDD_val_size = len(SIDD_full_dataset) - SIDD_train_size
SIDD_train_dataset, SIDD_val_dataset = torch.utils.data.random_split(SIDD_full_dataset, [SIDD_train_size, SIDD_val_size])
SIDD_train_dataloader = DataLoader(SIDD_train_dataset, batch_size=2, shuffle=True)
SIDD_val_dataloader = DataLoader(SIDD_val_dataset, batch_size=2, shuffle=False)

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

# 以下请忽略
# class CIFAR10C(Dataset):
#     def __init__(self, data_path, transform=None, corruption_type="gaussian_noise", severity=1):
#         self.transform = transform

#         path = data_path
#         data_path = os.path.join(data_path, corruption_type + ".npy")
#         full_data = torch.from_numpy(np.load(data_path))

#         num_per_severity = 10000
#         start_idx = (severity - 1) * num_per_severity
#         end_idx = severity * num_per_severity
#         self.data = full_data[start_idx:end_idx]

#         labels_path = os.path.join(path, "labels.npy")
#         self.labels = torch.from_numpy(np.load(labels_path))
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         img, label = self.data[idx], self.labels[idx]
#         if self.transform:
#             img = self.transform(img)
#         return img, label
    

# CIFAR10_full_trainset = torchvision.datasets.CIFAR10(
#     root='Task2/data',
#     train=True,
#     download=True,
#     transform=CIFAR10_transform
# )
# CIFAR10_train_size = int(0.9 * len(CIFAR10_full_trainset))
# CIFAR10_val_size = len(CIFAR10_full_trainset) - CIFAR10_train_size
# CIFAR10_train_dataset, CIFAR10_val_dataset = torch.utils.data.random_split(CIFAR10_full_trainset, [CIFAR10_train_size, CIFAR10_val_size])
# CIFAR10_train_dataloader = DataLoader(CIFAR10_train_dataset, batch_size=16, shuffle=True)
# CIFAR10_val_dataloader = DataLoader(CIFAR10_val_dataset, batch_size=16, shuffle=False)

# CIFAR10C_data_path = "MTL/data/CIFAR-10-C"
# CIFAR10C_full_dataset = CIFAR10C(CIFAR10C_data_path, transform=CIFAR10C_transform, corruption_type="gaussian_noise", severity=1)
# CIFAR10C_train_size = int(0.8 * len(CIFAR10C_full_dataset))
# CIFAR10C_val_size = len(CIFAR10C_full_dataset) - CIFAR10C_train_size
# CIFAR10C_train_dataset, CIFAR10C_val_dataset = torch.utils.data.random_split(CIFAR10C_full_dataset, [CIFAR10C_train_size, CIFAR10C_val_size])
# CIFAR10C_train_dataloader = DataLoader(CIFAR10C_train_dataset, batch_size=16, shuffle=True)
# CIFAR10C_test_dataloader = DataLoader(CIFAR10C_val_dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = MTL().to(device)

criterion_task1 = nn.L1Loss()
criterion_task2 = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


epochs = 20
task1_weight = 50.0
task2_weight = 0.5

loader_len = max(len(SIDD_train_dataloader), len(CIFAR10_train_dataloader))
SIDD_Iter = cycle(SIDD_train_dataloader)
CIFAR10_Iter = cycle(CIFAR10_train_dataloader)

loss_list = []
psnr_1_list = []
psnr_2_list = []
accuracy_list = []

for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    pbar = tqdm.tqdm(total=loader_len, desc=f"Epoch {epoch+1}/{epochs} Training", unit="batch")
    for step in tqdm.tqdm(range(loader_len)):
        noisy_imgs, clean_imgs = next(SIDD_Iter)
        noisy_class_imgs, clean_class_imgs, class_labels = next(CIFAR10_Iter)

        noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
        noisy_class_imgs, clean_class_imgs, class_labels = noisy_class_imgs.to(device), clean_class_imgs.to(device), class_labels.to(device)

        optimizer.zero_grad()

        denoised_imgs, _ = net(noisy_imgs, task='task1')
        loss_task1_a = criterion_task1(denoised_imgs, clean_imgs)
        # (loss_task1_a*task1_weight).backward()

        denoised_class_imgs, class_outputs = net(noisy_class_imgs, task='all')
        loss_task1_b = criterion_task1(denoised_class_imgs, clean_class_imgs)
        # (loss_task1_b*task1_weight).backward()
        loss_task2 = criterion_task2(class_outputs, class_labels)
        # (loss_task2*task2_weight).backward()

        total_loss = (loss_task1_a*task1_weight + loss_task1_b*task1_weight + loss_task2*task2_weight)/(task1_weight*2 + task2_weight)
        total_loss.backward()

        optimizer.step()
        running_loss += total_loss.item()
        avg_loss = running_loss / (step + 1)
        pbar.set_description(f"Epoch {epoch+1}/{epochs} Training - Loss: {avg_loss:.4f}")
        pbar.update(1)

    pbar.close()

    net.eval()
    initial_psnr_1 = 0.0
    initial_ssim_1 = 0.0
    initial_psnr_2 = 0.0
    initial_ssim_2 = 0.0
    total_psnr_1 = 0.0
    total_ssim_1 = 0.0
    total_psnr_2 = 0.0
    total_ssim_2 = 0.0
    correct_count = 0
    count_1 = 0
    count_2 = 0
    with torch.no_grad():
        for noisy_imgs, clean_imgs in SIDD_val_dataloader:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            denoised_imgs, _ = net(noisy_imgs, task='task1')

            for i in range(noisy_imgs.size(0)):
                initial_psnr_1 += psnr(noisy_imgs[i], clean_imgs[i]).item()
                initial_ssim_1 += ssim(noisy_imgs[i].unsqueeze(0), clean_imgs[i].unsqueeze(0), data_range=1.0, size_average=True).item()
                total_psnr_1 += psnr(denoised_imgs[i], clean_imgs[i]).item()
                total_ssim_1 += ssim(denoised_imgs[i].unsqueeze(0), clean_imgs[i].unsqueeze(0), data_range=1.0, size_average=True).item()
                count_1 += 1

        for noisy_class_imgs, clean_class_imgs, class_labels in CIFAR10_val_dataloader:
            noisy_class_imgs, clean_class_imgs, class_labels = noisy_class_imgs.to(device), clean_class_imgs.to(device), class_labels.to(device)
            denoised_class_imgs, class_outputs = net(noisy_class_imgs, task='all')

            for i in range(noisy_class_imgs.size(0)):
                initial_psnr_2 += psnr(noisy_class_imgs[i], clean_class_imgs[i]).item()
                initial_ssim_2 += ssim(noisy_class_imgs[i].unsqueeze(0), clean_class_imgs[i].unsqueeze(0), data_range=1.0, size_average=True).item()
                total_psnr_2 += psnr(denoised_class_imgs[i], clean_class_imgs[i]).item()
                total_ssim_2 += ssim(denoised_class_imgs[i].unsqueeze(0), clean_class_imgs[i].unsqueeze(0), data_range=1.0, size_average=True).item()
                _, predicted = torch.max(class_outputs[i], 0)
                if predicted.item() == class_labels[i].item():
                    correct_count += 1
                count_2 += 1
    
    avg_initial_psnr_1 = initial_psnr_1 / count_1
    avg_initial_ssim_1 = initial_ssim_1 / count_1
    avg_total_psnr_1 = total_psnr_1 / count_1
    avg_total_ssim_1 = total_ssim_1 / count_1

    avg_initial_psnr_2 = initial_psnr_2 / count_2
    avg_initial_ssim_2 = initial_ssim_2 / count_2
    avg_total_psnr_2 = total_psnr_2 / count_2
    avg_total_ssim_2 = total_ssim_2 / count_2

    accuracy = correct_count / count_2 * 100.0

    avg_loss = running_loss / loader_len
    loss_list.append(avg_loss)
    psnr_1_list.append(avg_total_psnr_1)
    psnr_2_list.append(avg_total_psnr_2)
    accuracy_list.append(accuracy)
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}")
    print(f"Task 1 - Initial PSNR: {avg_initial_psnr_1:.4f}, SSIM: {avg_initial_ssim_1:.4f} => Denoised PSNR: {avg_total_psnr_1:.4f}, SSIM: {avg_total_ssim_1:.4f}")
    print(f"Task 2 - Initial PSNR: {avg_initial_psnr_2:.4f}, SSIM: {avg_initial_ssim_2:.4f} => Denoised PSNR: {avg_total_psnr_2:.4f}, SSIM: {avg_total_ssim_2:.4f}")
    print(f"Task 2 - Classification Accuracy: {accuracy:.2f}%")


# 保存模型
save_folder = './saved_models'
model_name = 'mtl'
epoch = epochs
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

os.makedirs(save_folder, exist_ok=True)
filename = f"{model_name}_{timestamp}_ep{epoch}_loss{loss_list[-1]:.4f}_psnr1{psnr_1_list[-1]:.2f}dB_acc{accuracy_list[-1]:.2f}.pth"
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
plt.plot(range(1, epochs + 1), accuracy_list, marker='o')
plt.title('Validation Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid()
plt.savefig(f'accuracy_curve_{timestamp}.png')
plt.close()