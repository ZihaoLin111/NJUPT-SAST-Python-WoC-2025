import torch
import torch.nn as nn

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

class MTL(nn.Module): # 结合DUB和普通的CNN
    def __init__(self, in_channels=3, out_channels=128, dub_num=4):
        super(MTL, self).__init__()

        # Shared Layers
        self.entry = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU(num_parameters=out_channels, init=0.25)
        self.dub_blocks = nn.ModuleList([DUB(out_channels, out_channels) for _ in range(dub_num)])

        # For Task 1
        self.exit = nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1)

        # For Task 2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.prelu_21 = nn.PReLU(num_parameters=out_channels, init=0.25)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear((out_channels) * 8 * 8, 512) 
        self.prelu_22 = nn.PReLU(num_parameters=512, init=0.25)
        self.fc2 = nn.Linear(512, 128)
        self.prelu_23 = nn.PReLU(num_parameters=128, init=0.25)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x, task='all'):

        shared_feat = self.entry(x)
        shared_feat = self.prelu(shared_feat)
        for dub in self.dub_blocks:
            shared_feat = dub(shared_feat)

        out1, out2 = None, None

        if task == 'task1' or task == 'all':
            out1 = self.exit(shared_feat)

        if task == 'task2' or task == 'all':
            out2 = self.pool1(shared_feat)
            out2 = self.prelu_21(self.conv(out2))
            out2 = self.pool2(out2)
            out2 = out2.view(out2.size(0), -1)
            out2 = self.prelu_22(self.fc1(out2))
            out2 = self.prelu_23(self.fc2(out2))
            out2 = self.classifier(out2)
        
        return out1, out2


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
        

class Res_DUB_MTL(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, num_classes=10):
        super(Res_DUB_MTL, self).__init__()

        # Shared Layers
        self.entry = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.reslayer1 = nn.Sequential(
            ResBlock(out_channels, out_channels, stride=1),
            ResBlock(out_channels, out_channels, stride=1)
        )
        self.dublayer1 = nn.Sequential(
            DUB(out_channels, out_channels),
            DUB(out_channels, out_channels)
        )

        # For Task 1
        self.dublayer2 = nn.Sequential(
            DUB(out_channels, out_channels),
            DUB(out_channels, out_channels)
        )
        self.exit = nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1)

        # For Task 2
        self.reslayer2 = nn.Sequential(
            ResBlock(out_channels, out_channels*2, stride=2),
            ResBlock(out_channels*2, out_channels*2, stride=1)
        )
        self.reslayer3 = nn.Sequential(
            ResBlock(out_channels*2, out_channels*4, stride=2),
            ResBlock(out_channels*4, out_channels*4, stride=1)
        )
        self.reslayer4 = nn.Sequential(
            ResBlock(out_channels*4, out_channels*8, stride=2),
            ResBlock(out_channels*8, out_channels*8, stride=1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flaten = nn.Flatten()
        self.fc = nn.Linear(out_channels*8, num_classes)

    def forward(self, x, task='all'):

        shared_feat = self.entry(x)
        shared_feat = self.bn1(shared_feat)
        shared_feat = self.relu(shared_feat)
        shared_feat = self.reslayer1(shared_feat)
        shared_feat = self.dublayer1(shared_feat)

        out1, out2 = None, None

        if task == 'task1' or task == 'all':
            task1_feat = self.dublayer2(shared_feat)
            out1 = self.exit(task1_feat)

        if task == 'task2' or task == 'all':
            task2_feat = self.reslayer2(shared_feat)
            task2_feat = self.reslayer3(task2_feat)
            task2_feat = self.reslayer4(task2_feat)
            task2_feat = self.avgpool(task2_feat)
            task2_feat = self.flaten(task2_feat)
            out2 = self.fc(task2_feat)
        
        return out1, out2

       