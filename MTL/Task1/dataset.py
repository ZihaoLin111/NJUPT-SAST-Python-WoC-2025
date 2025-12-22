import os
from torch.utils.data import Dataset
from PIL import Image


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

        if self.transform:
            noisy_image = self.transform(noisy_image)
            gt_image = self.transform(gt_image)

        return noisy_image, gt_image