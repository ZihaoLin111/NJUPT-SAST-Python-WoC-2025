import os
from sympy import im
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import datetime
from PIL import Image
from tqdm import tqdm
import csv, json
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
from model import DnCNN

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = DnCNN().to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

class SIDD_Dataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform
        self.data_path = os.path.join(data_path, 'Data')

        # 读取Scene_Instances.txt文件
        scene_file = os.path.join(data_path, 'Scene_Instances.txt')
        with open(scene_file, 'r') as f:
            self.scene_instances = [line.strip() for line in f]

        # 构建样本列表
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
    
data_path = 'data/SIDD_Small_sRGB_Only'

full_dataset = SIDD_Dataset(data_path, transform=transform)

train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size, test_size]
) 

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

epochs = 10
loss_list = []
psnr_list = []
for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader))
    for batch_idx, (noisy_image, gt_image) in pbar:
        noisy_image = noisy_image.to(device)
        gt_image = gt_image.to(device)
        
        optimizer.zero_grad()
        outputs = net(noisy_image)
        loss = criterion(outputs, gt_image)
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
    if epoch == 0:
        initial_psnr = 0.0
        initial_ssim = 0.0
        initial_count = 0
    with torch.no_grad():
        for noisy_image, gt_image in val_loader:
            noisy_image = noisy_image.to(device)
            gt_image = gt_image.to(device)
            if epoch == 0:
                for i in range(noisy_image.size(0)):
                    initial_psnr += psnr(noisy_image[i], gt_image[i]).item()
                    initial_ssim += ssim(noisy_image[i].unsqueeze(0), gt_image[i].unsqueeze(0), data_range=1.0).item() # SSIM要求4D输入
                    initial_count += 1

            outputs = net(noisy_image)
            
            # 限制在0-1之间
            outputs = torch.clamp(outputs, 0.0, 1.0)
            gt_image = torch.clamp(gt_image, 0.0, 1.0)

            for i in range(outputs.size(0)):
                total_psnr += psnr(outputs[i], gt_image[i]).item()
                total_ssim += ssim(outputs[i].unsqueeze(0), gt_image[i].unsqueeze(0), data_range=1.0).item() # SSIM要求4D输入
            
            count += outputs.size(0)
    if epoch == 0:
        initial_psnr_avg = initial_psnr / initial_count
        initial_ssim_avg = initial_ssim / initial_count
    else:
        # 使用第一个epoch保存的值
        pass
    avg_loss = running_loss / len(train_loader)
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    loss_list.append(avg_loss)
    psnr_list.append(avg_psnr)
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}')
    print(f'Iniatial: PSNR: {initial_psnr_avg:.4f}dB, SSIM: {initial_ssim_avg:.4f}  => Denoised: PSNR: {avg_psnr:.4f}dB, SSIM: {avg_ssim:.4f}')

# 保存模型
save_folder = './saved_models'
model_name = 'DnCNN'
epoch = epochs
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

os.makedirs(save_folder, exist_ok=True)
filename = f"{model_name}_{timestamp}_ep{epoch}_loss{loss_list[-1]:.4f}_psnr{psnr_list[-1]:.2f}dB.pth"
save_path = os.path.join(save_folder, filename)
torch.save(net.state_dict(), save_path)

print(f"模型参数已保存至: {save_path}")

run_log = {
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "DnCNN",
    "epochs": epochs,
    "optimizer": "Adam",
    "lr": 0.001,
    "train_loss_last": loss_list[-1],
    "val_psnr_last": psnr_list[-1],
}
# 保存到 JSONL
jsonl_path = "runs.jsonl"
with open(jsonl_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(run_log) + "\n")

# 记录到 CSV
csv_path = "runs.csv"
csv_fields = ["timestamp", "model", "epochs", "optimizer", "lr", "train_loss_last", "val_psnr_last"]
file_exists = os.path.exists(csv_path)
with open(csv_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    if not file_exists:
        writer.writeheader()
    writer.writerow({k: run_log[k] for k in csv_fields})

print("Logged to", jsonl_path, "and", csv_path)
    
# 绘制准确率曲线
plt.plot(range(1, epochs+1), psnr_list, label='Validation PSNR')
plt.xlabel('Epochs')
plt.ylabel('Validation PSNR (dB)')
plt.title('Training and Validation Accuracy')
plt.legend()

os.makedirs('charts', exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
fig_path = f"charts/train_val_PSNR_{timestamp}.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"图表已保存至 {fig_path}")

plt.show()