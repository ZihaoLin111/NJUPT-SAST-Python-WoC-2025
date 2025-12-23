import os
import torch
import torch.nn as nn
import torchvision
import datetime
import json
import csv
from PIL import Image

data_path = 'data/SIDD_Small_sRGB_Only'
scene_file = os.path.join(data_path, 'Scene_Instances.txt')
with open(scene_file, 'r') as f:
    scene_instances = [line.strip() for line in f]
Data_folder = os.path.join(data_path, 'Data')
samples = []
for folder in scene_instances:
    nosiy_path = os.path.join(Data_folder, folder, 'NOISY_SRGB_010.PNG')
    gt_path = os.path.join(Data_folder, folder, 'GT_SRGB_010.PNG')
    
    samples.append((nosiy_path, gt_path))

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        self.layer1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.layer2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.layer3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    

length = len(samples)
train_size = int(0.8 * length)
train_samples = samples[:train_size]
val_size = int(0.1 * length)
val_samples = samples[train_size:train_size+val_size]
test_samples = samples[train_size+val_size:]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = SRCNN().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

epochs = 10
loss_list = []
psnr_list = []
for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    for noisy_path, gt_path in train_samples:
        noisy_image = Image.open(noisy_path).convert('RGB')
        gt_image = Image.open(gt_path).convert('RGB')
        
        w, h = noisy_image.size

        # resize to 512 * 512
        noisy_image = noisy_image.resize((512, 512))
        gt_image = gt_image.resize((512, 512))

        noisy_image = transform(noisy_image).to(device)
        gt_image = transform(gt_image).to(device)
        
        optimizer.zero_grad()
        outputs = net(noisy_image)
        loss = criterion(outputs, gt_image)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    net.eval()

    total_psnr = 0.0
    count = 0
    with torch.no_grad():
        for noisy_path, gt_path in val_samples:
            noisy_image = Image.open(noisy_path).convert('RGB')
            gt_image = Image.open(gt_path).convert('RGB')

            w, h = noisy_image.size
            
            noisy_image = noisy_image.resize((512, 512))
            gt_image = gt_image.resize((512, 512))

            noisy_image = transform(noisy_image).to(device)
            gt_image = torchvision.transforms.ToTensor()(gt_image).to(device)
            
            outputs = net(noisy_image)
            
            outputs = outputs * 0.5 + 0.5 # 逆归一化
            outputs = torch.clamp(outputs, 0.0, 1.0) # 限制在0-1之间

            total_psnr += psnr(outputs, gt_image).item()
            count += 1
            
    avg_loss = running_loss / len(train_samples)
    avg_psnr = total_psnr / count
    loss_list.append(avg_loss)
    psnr_list.append(avg_psnr)
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Val PSNR: {avg_psnr:.2f} dB')


# 保存模型
save_folder = './saved_models'
model_name = 'SRCNN'
epoch = epochs
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

os.makedirs(save_folder, exist_ok=True)
filename = f"{model_name}_{timestamp}_ep{epoch}_loss{loss_list[-1]:.4f}_psnr{psnr_list[-1]:.2f}dB.pth"
save_path = os.path.join(save_folder, filename)
torch.save(net.state_dict(), save_path)

print(f"模型参数已保存至: {save_path}")

run_log = {
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "SRCNN",
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