import torch
import torch.nn as nn
import torchvision
from model import DUB, DIDN
from dataset import SIDD_Dataset
from pytorch_msssim import ssim
import os
import random
from PIL import Image

model_path = './saved_models/DIDN_20251226_0343_ep20_loss0.0135_psnr33.67dB.pth'
data_path = 'data/SIDD_Small_sRGB_Only'

transform = torchvision.transforms.ToTensor()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = DIDN().to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

test_dataset = SIDD_Dataset(data_path, transform=transform, crop_size = 128, train=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

net.load_state_dict(torch.load(model_path, map_location=device))
net.eval()
initial_psnr = 0.0
initial_ssim = 0.0
total_psnr = 0.0
total_ssim = 0.0
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

i = (h - 256) // 2
j = (w - 256) // 2
noisy_image_copy = noisy_image
noisy_image_copy = noisy_image_copy.crop((j, i, j + 256, i + 256))

noisy_image = noisy_image.crop((j, i, j + 256, i + 256))

noisy_image = transform(noisy_image).unsqueeze(0).to(device)

outputs = net(noisy_image)
            
outputs = outputs.squeeze(0)
outputs = torch.clamp(outputs, 0.0, 1.0) # 限制在0-1之间
# Tensor to PIL Image
to_pil = torchvision.transforms.ToPILImage()
outputs = to_pil(outputs)

# 对照组2 
crop_gt = gt_image.crop((j, i, j + 256, i + 256))

# 保存修复图和原图
outputs.save('denoised_image.png')
crop_gt.save('ground_truth_image.png')
noisy_image_copy.save('noisy_image.png')

print('图片已保存')
