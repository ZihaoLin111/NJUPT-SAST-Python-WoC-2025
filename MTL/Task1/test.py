import random
from numpy import resize
from sympy import im
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
from PIL import Image
import datetime
import os
from model import DnCNN


model_path = './saved_models/DnCNN_20251224_1021_ep10_loss0.0135_psnr35.82dB.pth'
data_path = 'data/SIDD_Small_sRGB_Only'

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = DnCNN().to(device)

criterion = nn.MSELoss()
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
    
full_dataset = SIDD_Dataset(data_path, transform=transform)

train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size, test_size]
) 

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)


psnr_list = []
ssim_list = []


net.load_state_dict(torch.load(model_path))
net.eval()
initial_psnr = 0
initial_ssim = 0
total_psnr = 0
total_ssim = 0
count = 0
with torch.no_grad():
    for noisy_image, gt_image in test_loader:
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
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_initial_psnr = initial_psnr / count
    avg_initial_ssim = initial_ssim / count
    
print(f'Iniatial: PSNR: {avg_initial_psnr:.4f}dB, SSIM: {avg_initial_ssim:.4f}  => Denoised: PSNR: {avg_psnr:.4f}dB, SSIM: {avg_ssim:.4f}')

# 展示部分结果
scene_file = os.path.join(data_path, 'Scene_Instances.txt')
with open(scene_file, 'r') as f:
    scene_instances = [line.strip() for line in f]
Data_folder = os.path.join(data_path, 'Data')
random_idx = random.randint(0, len(scene_instances) - 1)
random_indice = scene_instances[random_idx]

noisy_path = os.path.join(Data_folder, random_indice, 'NOISY_SRGB_010.PNG')
gt_path = os.path.join(Data_folder, random_indice, 'GT_SRGB_010.PNG')

noisy_image = Image.open(noisy_path).convert('RGB')
gt_image = Image.open(gt_path).convert('RGB')
        
w, h = noisy_image.size

noisy_image_copy = noisy_image

noisy_image = noisy_image.resize((512, 512))

# 对照组1
noisy_image_re_copy = noisy_image.resize((w, h))

noisy_image = transform(noisy_image).unsqueeze(0).to(device)

outputs = net(noisy_image)
            
outputs = outputs.squeeze(0)
outputs = torch.clamp(outputs, 0.0, 1.0) # 限制在0-1之间
# Tensor to PIL Image
to_pil = torchvision.transforms.ToPILImage()
outputs = to_pil(outputs)
outputs = outputs.resize((w, h))

# 对照组2 
resize_gt = gt_image.resize((512, 512))
resize_gt = resize_gt.resize((w, h))

# 保存修复图和原图
outputs.save('denoised.png')
gt_image.save('gt.png')
resize_gt.save('resized_gt.png')
noisy_image_copy.save('noisy.png')
noisy_image_re_copy.save('noisy_re.png')
print('图片已保存')


# # 画图再议，在想要不要多训练几个模型然后画图比较测试结果
# # 画图的代码直接让AI帮忙改好的()

# epochs_range = range(1, epochs + 1)

# fig, ax1 = plt.subplots()
# # 左 y 轴：PSNR
# ax1.plot(epochs_range, psnr_list, label='PSNR (dB)', color='tab:blue')
# ax1.set_xlabel('Epoch')
# ax1.set_ylabel('PSNR (dB)', color='tab:blue')
# ax1.tick_params(axis='y', labelcolor='tab:blue')

# # 右 y 轴：SSIM
# ax2 = ax1.twinx()
# ax2.plot(epochs_range, ssim_list, label='SSIM', color='tab:orange')
# ax2.set_ylabel('SSIM', color='tab:orange')
# ax2.tick_params(axis='y', labelcolor='tab:orange')
# ax2.set_ylim(0, 1)   

# # 标题
# plt.title('PSNR and SSIM over Epochs')

# # 合并图例
# lines_1, labels_1 = ax1.get_legend_handles_labels()
# lines_2, labels_2 = ax2.get_legend_handles_labels()
# ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

# # 保存
# os.makedirs('test_result_charts', exist_ok=True)
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# fig_path = f"test_result_charts/train_val_PSNR_SSIM_{timestamp}.png"
# plt.savefig(fig_path, dpi=300, bbox_inches='tight')

# plt.show()