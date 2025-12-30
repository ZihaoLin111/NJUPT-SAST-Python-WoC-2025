import re
from typing import Any
from pyparsing import C
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
    

class DnCNN(nn.Module):
    def __init__(self, num_channels=3, num_features=64, num_layers=17):
        super(DnCNN, self).__init__()
        layers = []
        # 第一层
        layers.append(nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        # 中间层
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        # 最后一层
        layers.append(nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return x - out  # 残差学习
    

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



class DUB(nn.Module): # Full DUB
    def __init__(self, in_channels, out_channels):
        super(DUB, self).__init__()
        
        # 下采样

        # Level 0
        self.con_00 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.prelu_00 = nn.PReLU(num_parameters=out_channels, init=0.25)
        self.con_01 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.prelu_01 = nn.PReLU(num_parameters=out_channels, init=0.25)
        self.downcon_0 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1) # 下采样
            
        # Level 1   
        self.con_10 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.prelu_10 = nn.PReLU(num_parameters=out_channels, init=0.25)
        self.downcon_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)# 下采样
        
        # Level 2
        self.con_20 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.prelu_20 = nn.PReLU(num_parameters=out_channels, init=0.25)
        self.con_21 = nn.Conv2d(out_channels, out_channels*4, kernel_size=1, stride=1, padding=0)
        self.up_2 = nn.PixelShuffle(upscale_factor=2) # 上采样

        # 上采样
        # Level 1
        self.con_up10 = nn.Conv2d(out_channels*2, out_channels, kernel_size=1, stride=1, padding=0)
        self.con_up11 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.prelu_up10 = nn.PReLU(num_parameters=out_channels, init=0.25)
        self.con_up12 = nn.Conv2d(out_channels, out_channels*4, kernel_size=1, stride=1, padding=0)
        self.up_1 = nn.PixelShuffle(upscale_factor=2) # 上采样

        # Level 0
        self.con_up00 = nn.Conv2d(out_channels*2, out_channels, kernel_size=1, stride=1, padding=0)
        self.con_up01 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.prelu_up00 = nn.PReLU(num_parameters=out_channels, init=0.25)
        self.con_up02 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.prelu_up01 = nn.PReLU(num_parameters=out_channels, init=0.25)
        self.con_up03 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res0 = x

        # 下采样
        # Level 0
        x0 = self.prelu_00(self.con_00(x))
        x0 = self.prelu_01(self.con_01(x0)) + res0 # 后续拼接
        
        # Level 1
        x1 = self.downcon_0(x0)
        res1 = x1
        x1 = self.prelu_10(self.con_10(x1)) + res1 # 后续拼接

        # Level 2
        x2 = self.downcon_1(x1)
        res2 = x2
        x2 = self.prelu_20(self.con_20(x2)) + res2
        x2 = self.con_21(x2)

        # 上采样
        # Level 1
        xu1 = self.up_2(x2)
        xu1 = torch.cat((xu1, x1), dim=1)
        xu1 = self.con_up10(xu1)
        resu1 = xu1
        xu1 = self.prelu_up10(self.con_up11(xu1)) + resu1
        xu1 = self.con_up12(xu1)

        # Level 0
        xu0 = self.up_1(xu1)
        xu0 = torch.cat((xu0, x0), dim=1)
        xu0 = self.con_up00(xu0)
        resu0 = xu0
        xu0 = self.prelu_up00(self.con_up01(xu0))
        xu0 = self.prelu_up01(self.con_up02(xu0)) + resu0
        xu0 = self.con_up03(xu0)

        return xu0 + res0



class DIDN(nn.Module):
    def __init__(self, in_channels=3, out_channels=128, num_dub=4):
        super(DIDN, self).__init__()
        self.entry = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.prelu = nn.PReLU(num_parameters=out_channels, init=0.25)
        self.downcon = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.dub_blocks = nn.ModuleList([DUB(out_channels, out_channels) for _ in range(num_dub)])
        self.con_10 = nn.Conv2d(out_channels * num_dub, out_channels, kernel_size=1, stride=1, padding=0)
        self.con_11 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.prelu_10 = nn.PReLU(num_parameters=out_channels, init=0.25)
        self.con_12 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1, padding=0)
        self.up = nn.PixelShuffle(upscale_factor=2)
        self.exit = nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res1 = x
        x = self.prelu(self.entry(x))
        x = self.downcon(x)
        recon = []
        for dub in self.dub_blocks:
            x = dub(x)
            recon.append(x)
        x = torch.cat(recon[::-1], dim=1)
        x = self.con_10(x)
        res2 = x
        x = self.prelu_10(self.con_11(x)) + res2
        x = self.con_12(x)
        x = self.up(x)
        x = self.exit(x)
        return x + res1


