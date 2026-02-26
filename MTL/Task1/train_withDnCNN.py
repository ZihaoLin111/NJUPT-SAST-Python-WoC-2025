import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import datetime
from PIL import Image
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
from model import DnCNN
import random

class SIDD_Dataset_Crop(Dataset):
    def __init__(self, data_path,scene_list, transform=None, crop_size=0, train=True, repeat = 40):
        self.transform = transform
        self.data_path = os.path.join(data_path, 'Data')
        self.crop_size = crop_size
        self.train = train
        self.repeat = repeat
        self.scene_list = scene_list

        # 读取Scene_Instances.txt文件
        scene_file = os.path.join(data_path, 'Scene_Instances.txt')
        with open(scene_file, 'r') as f:
            self.scene_instances = [line.strip() for line in f]

        # 构建样本列表
        self.samples = []
        for folder in self.scene_list:
            nosiy_path = os.path.join(self.data_path, folder, 'NOISY_SRGB_010.PNG')
            gt_path = os.path.join(self.data_path, folder, 'GT_SRGB_010.PNG')
    
            self.samples.append((nosiy_path, gt_path))

        print(f"正在将 {len(self.samples)} 对高分辨率图片加载到内存")
        self.loaded_images = []
        
        for n_path, g_path in tqdm(self.samples):
            # 1. 打开图片
            n_img = Image.open(n_path).convert('RGB')
            g_img = Image.open(g_path).convert('RGB')
            
            # 2. 关键步骤：强制 PIL 加载数据
            # PIL 默认是懒加载(Lazy Load)，不调用 .load() 的话它不会真的把像素读进内存
            n_img.load() 
            g_img.load()
            
            # 3. 存入列表
            self.loaded_images.append((n_img, g_img))
            
        print("内存加载完成！")
            
    def __len__(self):
        if self.train:
            return len(self.samples) * self.repeat
        else:
            return len(self.samples)
        
    def __getitem__(self, idx):
        real_idx = idx % len(self.samples)

        noisy_image_src, gt_image_src = self.loaded_images[real_idx]

        noisy_image = noisy_image_src.copy()
        gt_image = gt_image_src.copy()
        
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

        if self.train:
            if random.random() > 0.5:
                noisy_image = torchvision.transforms.functional.hflip(noisy_image)
                gt_image = torchvision.transforms.functional.hflip(gt_image)
            if random.random() > 0.5:
                noisy_image = torchvision.transforms.functional.vflip(noisy_image)
                gt_image = torchvision.transforms.functional.vflip(gt_image)
            k = random.randint(0, 3)
            if k > 0:
                noisy_image = torchvision.transforms.functional.rotate(noisy_image, angle=90 * k)
                gt_image = torchvision.transforms.functional.rotate(gt_image, angle=90 * k)

        return noisy_image, gt_image

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

epochs = 150
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = DnCNN().to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(
    net.parameters(),
    lr=1e-4,      
    weight_decay=0     
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs, 
    eta_min=1e-6        
)

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

    
data_path = 'data/SIDD_Small_sRGB_Only'

scene_file = os.path.join(data_path, 'Scene_Instances.txt')
with open(scene_file, 'r') as f:
    all_scenes = [line.strip() for line in f]

split_idx = int(0.8 * len(all_scenes))
train_scenes = all_scenes[:split_idx]
val_scenes = all_scenes[split_idx:]

train_dataset = SIDD_Dataset_Crop(
    data_path, 
    scene_list=train_scenes, 
    transform=transform, 
    crop_size=32, 
    train=True, 
    repeat=10
)

val_dataset = SIDD_Dataset_Crop(
    data_path, 
    scene_list=val_scenes, 
    transform=transform, 
    crop_size=32, 
    train=False, 
    repeat=1
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)


loss_list = []
psnr_list = []
ssim_list = []
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
        scheduler.step()
        
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
                initial_ssim += ssim(noisy_image[i].unsqueeze(0), gt_image[i].unsqueeze(0), data_range=1.0).item() # SSIM要求4D输入


            outputs = net(noisy_image)
            
            # 限制在0-1之间
            outputs = torch.clamp(outputs, 0.0, 1.0)
            gt_image = torch.clamp(gt_image, 0.0, 1.0)

            for i in range(outputs.size(0)):
                total_psnr += psnr(outputs[i], gt_image[i]).item()
                total_ssim += ssim(outputs[i].unsqueeze(0), gt_image[i].unsqueeze(0), data_range=1.0).item() # SSIM要求4D输入
            
            count += outputs.size(0)
    initial_psnr_avg = initial_psnr / count
    initial_ssim_avg = initial_ssim / count
    avg_loss = running_loss / len(train_loader)
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    loss_list.append(avg_loss)
    psnr_list.append(avg_psnr)
    ssim_list.append(avg_ssim)
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}')
    print(f'Iniatial: PSNR: {initial_psnr_avg:.4f}dB, SSIM: {initial_ssim_avg:.4f}  => Denoised: PSNR: {avg_psnr:.4f}dB, SSIM: {avg_ssim:.4f}')

best_psnr = max(psnr_list)
best_ssim = max(ssim_list)

# 保存模型
save_folder = './saved_models'
model_name = 'DnCNN'
epoch = epochs
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

os.makedirs(save_folder, exist_ok=True)
filename = f"{model_name}(for32*32)_{timestamp}_ep{epoch}_loss{loss_list[-1]:.4f}_best_psnr{best_psnr:.2f}dB_best_ssim{best_ssim:.2f}.pth"
save_path = os.path.join(save_folder, filename)
torch.save(net.state_dict(), save_path)

print(f"模型参数已保存至: {save_path}")

run_log = {
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "DnCNN",
    "epochs": epochs,
    "optimizer": "Adam",
    "lr": 1e-4,
    "train_loss_last": loss_list[-1],
    "val_psnr_best": best_psnr,
    "val_ssim_best": best_ssim,
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

plt.close()