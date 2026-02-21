import torch
import torch.nn as nn
import torchvision
from model import DUB, DIDN
import tqdm
from dataset import SIDD_Dataset
from pytorch_msssim import ssim
import datetime
import os
import json
import csv
import matplotlib.pyplot as plt


transform = torchvision.transforms.ToTensor()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = DIDN().to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

data_path = 'data/SIDD_Small_sRGB_Only'

full_dataset_train = SIDD_Dataset(data_path, transform=transform, crop_size=256, train=True)
full_dataset_val = SIDD_Dataset(data_path, transform=transform, crop_size=256, train=False)

train_size = int(0.8 * len(full_dataset_train))
val_size = int(0.1 * len(full_dataset_train))

train_dataset, _ = torch.utils.data.random_split(full_dataset_train, [train_size, len(full_dataset_train) - train_size])
val_dataset, _ = torch.utils.data.random_split(full_dataset_val, [val_size, len(full_dataset_val) - val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)

epochs = 10
loss_list = []
psnr_list = []
for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    pbar = tqdm.tqdm(enumerate(train_loader, 1), total=len(train_loader))
    for batch_idx, (noisy_image, gt_image) in pbar:
        noisy_image = noisy_image.to(device)
        gt_image = gt_image.to(device)

        optimizer.zero_grad()
        denoised_image = net(noisy_image)
        loss = criterion(denoised_image, gt_image)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        avg_loss_so_far = running_loss / batch_idx
        pbar.set_description(f"Epoch {epoch+1}/{epochs}")
        pbar.set_postfix(loss=loss.item(), avg_loss=avg_loss_so_far)

    net.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    initial_psnr = 0.0
    initial_ssim = 0.0
    with torch.no_grad():
        for noisy_image, gt_image in val_loader:
            noisy_image = noisy_image.to(device)
            gt_image = gt_image.to(device)

            for i in range(noisy_image.size(0)):
                initial_psnr += psnr(noisy_image[i], gt_image[i]).item()
                initial_ssim += ssim(noisy_image[i].unsqueeze(0), gt_image[i].unsqueeze(0), data_range=1.0).item()
            outputs = net(noisy_image)

            outputs = torch.clamp(outputs, 0.0, 1.0)
            gt_image = torch.clamp(gt_image, 0.0, 1.0)

            for i in range(noisy_image.size(0)):
                total_psnr += psnr(outputs[i], gt_image[i]).item()
                total_ssim += ssim(outputs[i].unsqueeze(0), gt_image[i].unsqueeze(0), data_range=1.0).item()

            
            count += noisy_image.size(0)

    initial_psnr_avg = initial_psnr / count
    initial_ssim_avg = initial_ssim / count
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_loss = running_loss / len(train_loader)
    loss_list.append(avg_loss)
    psnr_list.append(avg_psnr)
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}')
    print(f'Iniatial: PSNR: {initial_psnr_avg:.4f}dB, SSIM: {initial_ssim_avg:.4f}  => Denoised: PSNR: {avg_psnr:.4f}dB, SSIM: {avg_ssim:.4f}')


# 保存模型
save_folder = './saved_models'
model_name = 'DIDN'
epoch = epochs
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

os.makedirs(save_folder, exist_ok=True)
filename = f"{model_name}_{timestamp}_ep{epoch}_loss{loss_list[-1]:.4f}_psnr{psnr_list[-1]:.2f}dB.pth"
save_path = os.path.join(save_folder, filename)
torch.save(net.state_dict(), save_path)

print(f"模型参数已保存至: {save_path}")

run_log = {
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "DIDN",
    "epochs": epochs,
    "optimizer": "Adam",
    "lr": 1e-4,
    "train_loss_last": loss_list[-1],
    "val_psnr_last": psnr_list[-1],
}

# 保存到 JSONL
jsonl_path = "runs.jsonl"
with open(jsonl_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(run_log) + "\n")



print("Logged to", jsonl_path)

epochs_range = range(1, epochs + 1)

fig, ax1 = plt.subplots()
# 左 y 轴：PSNR
ax1.plot(epochs_range, psnr_list, label='PSNR (dB)', color='tab:blue')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('PSNR (dB)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# 右 y 轴：Loss
ax2 = ax1.twinx()
ax2.plot(epochs_range, loss_list, label='loss', color='tab:orange')
ax2.set_ylabel('Loss', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')
ax2.set_ylim(0, 0.15)   

# 标题
plt.title('PSNR and Loss over Epochs')

# 合并图例
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

# 保存
os.makedirs('test_result_charts', exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
fig_path = f"charts/train_val_PSNR_Loss_{timestamp}.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"图表已保存至 {fig_path}")

plt.show()