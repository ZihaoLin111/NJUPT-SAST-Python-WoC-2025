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
    
CIFAR10_train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4), # 随机裁剪
    torchvision.transforms.RandomHorizontalFlip(), # 随机旋转
    torchvision.transforms.ToTensor()
])

CIFAR10_val_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

clean_path = 'Task2/data'
corrupted_path = "MTL/data/CIFAR-10-C"

full_cifar10_dataset = PairedCIFAR10(
    clean_path=clean_path,
    corrupted_path=corrupted_path,
    transform=CIFAR10_train_transform,
    corruption_type="gaussian_noise",
    severity=3
)

CIFAR10_train_size = int(0.8 * len(full_cifar10_dataset))
CIFAR10_val_size = len(full_cifar10_dataset) - CIFAR10_train_size
CIFAR10_train_dataset, CIFAR10_val_dataset = torch.utils.data.random_split(full_cifar10_dataset, [CIFAR10_train_size, CIFAR10_val_size],generator=torch.Generator().manual_seed(42))
CIFAR10_val_dataset.dataset.transform = CIFAR10_val_transform
CIFAR10_train_dataloader = DataLoader(CIFAR10_train_dataset, batch_size=128, shuffle=True, num_workers=4)
CIFAR10_val_dataloader = DataLoader(CIFAR10_val_dataset, batch_size=128, shuffle=False, num_workers=4)

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    ResNet_model = ResNet18(num_classes=10).to(device)
    DIDN_model = DIDN().to(device)
    DnCNN_model = DnCNN().to(device)

    ResNet_path = './saved_models/ResNet18_20260205_0422_ep150_train_best_acc_100.000val_best_acc88.400.pth'
    ResNet_model.load_state_dict(torch.load(ResNet_path, map_location=device))

    epochs = 150

    criterion_task1 = nn.L1Loss()
    criterion_task2 = nn.CrossEntropyLoss()
    mtl_loss = UncertaintyWeightingLoss(task_num=2).to(device)

    optimizer = torch.optim.Adam([
        {'params':DnCNN_model.parameters()}, 
        {'params':mtl_loss.parameters(), 'lr':1e-3}
        ],lr=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs, 
        eta_min=1e-6        
    )

    loss_list = []
    psnr_list = []
    ssim_list = []
    acc_list = []

    for epoch in range(epochs):
        DnCNN_model.train()
        ResNet_model.eval()

        train_loss = 0.0

        pbar = tqdm.tqdm(CIFAR10_train_dataloader)
        for batch_idx, (corrupted_imgs, clean_imgs, labels) in enumerate(pbar):
            corrupted_imgs = corrupted_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            denoised_imgs = DnCNN_model(corrupted_imgs)
            resnet_input = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)).to(device)(denoised_imgs)
            outputs = ResNet_model(resnet_input)

            loss_task1 = criterion_task1(denoised_imgs, clean_imgs)
            loss_task2 = criterion_task2(outputs, labels)

            losses = [loss_task1, loss_task2]

            if batch_idx < 50:
                w = 0.5
            elif batch_idx < 100:
                w = 0.75
            else:
                w = 1.0

            total_loss = loss_task1 + loss_task2 * 0.5
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += total_loss.item()
            pbar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/(batch_idx+1):.4f}, Task1 Loss: {loss_task1.item():.4f}, Task2 Loss: {loss_task2.item():.4f}")
        
        loss_list.append(train_loss / len(CIFAR10_train_dataloader))

 
        DnCNN_model.eval()
        initial_psnr = 0.0
        final_psnr = 0.0
        initial_ssim = 0.0
        final_ssim = 0.0
        initial_correct = 0
        final_correct = 0
        count = 0
        with torch.no_grad():
            for batch_idx, (corrupted_imgs, clean_imgs, labels) in enumerate(CIFAR10_val_dataloader):
                corrupted_imgs = corrupted_imgs.to(device)
                clean_imgs = clean_imgs.to(device)

                denoised_imgs = DnCNN_model(corrupted_imgs)
                
                resnet_input = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)).to(device)(corrupted_imgs)
                outputs_initial = ResNet_model(resnet_input)
                _, predicted_initial = torch.max(outputs_initial.data, 1)
                
                resnet_input = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)).to(device)(denoised_imgs)
                outputs_denoised = ResNet_model(resnet_input)
                _, predicted_denoised = torch.max(outputs_denoised.data, 1)

                initial_correct += (predicted_initial == labels.to(device)).sum().item()
                final_correct += (predicted_denoised == labels.to(device)).sum().item() 

                for i in range(corrupted_imgs.size(0)):
                    initial_psnr += psnr(corrupted_imgs[i], clean_imgs[i]).item()
                    final_psnr += psnr(denoised_imgs[i], clean_imgs[i]).item()
                    initial_ssim += ssim(corrupted_imgs[i].unsqueeze(0), clean_imgs[i].unsqueeze(0), data_range=1.0, size_average=True).item()
                    final_ssim += ssim(denoised_imgs[i].unsqueeze(0), clean_imgs[i].unsqueeze(0), data_range=1.0, size_average=True).item()
                    count += 1
                
                


        initial_psnr = initial_psnr / count
        final_psnr = final_psnr / count
        initial_ssim = initial_ssim / count
        final_ssim = final_ssim / count

        initial_acc = initial_correct / count * 100.0
        final_acc = final_correct / count * 100.0

        acc_list.append(final_acc)
        psnr_list.append(final_psnr)
        ssim_list.append(final_ssim)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(CIFAR10_train_dataloader):.4f}")
        print(f"Validation - Initial PSNR {initial_psnr:.4f} => Denoised PSNR {final_psnr:.4f}")
        print(f"Validation - Initial SSIM {initial_ssim:.4f} => Denoised SSIM {final_ssim:.4f}")
        print(f"Validation - Initial Acc {initial_acc:.2f}% => Denoised Acc {final_acc:.2f}%")


    best_psnr = max(psnr_list)
    best_epoch = psnr_list.index(best_psnr) + 1
    best_ssim = ssim_list[best_epoch - 1]
    best_acc = max(acc_list)

    # 保存模型
    save_folder = './saved_models'
    model_name = 'DnCNN_train_freeze_ResNet18_ablation1_'
    epoch = epochs
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    os.makedirs(save_folder, exist_ok=True)
    filename = f"{model_name}_{timestamp}_ep{epoch}_loss{loss_list[-1]:.4f}_best_psnr{best_psnr:.2f}dB_best_ssim{best_ssim:.2f}.pth"
    save_path = os.path.join(save_folder, filename)
    torch.save(DnCNN_model.state_dict(), save_path)

    print(f"模型参数以保存至: {save_path}")

    run_log = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "epochs": epochs,
        "optimizer": "Adam",
        "lr": 1e-4,
        "train_loss_last": loss_list[-1],
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        "best_acc": best_acc
        }

    # 保存JSONL
    jsonl_path = 'runs.jsonl'
    with open(jsonl_path, 'a', encoding="utf-8") as f:
        f.write(json.dumps(run_log) + '\n')

    # 绘制并保存图表
    # 损失曲线
    plt.figure()
    plt.plot(range(1, epochs + 1), loss_list, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid()
    plt.savefig(f'{model_name}_loss_curve_{timestamp}.png')
    plt.close()

    # PSNR曲线
    plt.figure()
    plt.plot(range(1, epochs + 1), psnr_list, label='PSNR')
    plt.title('Validation PSNR Curve')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid()
    plt.savefig(f'{model_name}_psnr_curve_{timestamp}.png')
    plt.close()

    # Acc曲线
    plt.figure()
    plt.plot(range(1, epochs + 1), acc_list, label='Accuracy')
    plt.title('Validation Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()
    plt.savefig(f'{model_name}_acc_curve_{timestamp}.png')
    plt.close()