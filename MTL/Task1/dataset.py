import os
from torch.utils.data import Dataset
from PIL import Image
import random

class SIDD_Dataset(Dataset):
    def __init__(self, data_path, transform=None, crop_size=0, train=True):
        self.transform = transform
        self.data_path = os.path.join(data_path, 'Data')
        self.crop_size = crop_size
        self.train = train

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

        # resize to 512 * 512
        noisy_image = noisy_image.resize((512, 512))
        gt_image = gt_image.resize((512, 512))
        
        # if self.crop_size > 0:
        #     if self.train:
        #         # 随机裁剪
        #         w, h = noisy_image.size
        #         i = random.randint(0, h - self.crop_size)
        #         j = random.randint(0, w - self.crop_size)
        #         noisy_image = noisy_image.crop((j, i, j + self.crop_size, i + self.crop_size))
        #         gt_image = gt_image.crop((j, i, j + self.crop_size, i + self.crop_size))
        #     else:
        #         # 中心裁剪
        #         w, h = noisy_image.size
        #         i = (h - self.crop_size) // 2
        #         j = (w - self.crop_size) // 2
        #         noisy_image = noisy_image.crop((j, i, j + self.crop_size, i + self.crop_size))
        #         gt_image = gt_image.crop((j, i, j + self.crop_size, i + self.crop_size))    
            

        if self.transform:
            noisy_image = self.transform(noisy_image)
            gt_image = self.transform(gt_image)

        return noisy_image, gt_image