import torch
import torch.nn as nn
import torchvision

class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        # 输入(3, 512, 512)，输出(64, 512, 512) 512 = (512 - 9 + 4*2) / 1 + 1
        self.layer1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        # 输入(64, 512, 512)，输出(32, 512, 512) 512 = (512 - 1 + 0*2) / 1 + 1
        self.layer2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        # 输入(32, 512, 512)，输出(3, 512, 512) 512 = (512 - 5 + 2*2) / 1 + 1
        self.layer3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x