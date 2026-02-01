import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1) # 输入(3, 32, 32)，输出(6, 32, 32)， 32 = (32 - 3 + 2*1) / 1 + 1
        self.pool = nn.MaxPool2d(2, 2) # 池化操作，压缩降维
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1) # 输入(6, 16, 16)，输出(16, 16, 16)，16 = (16 - 3 + 2*1) / 1 + 1
        self.fc1 = nn.Linear(16 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

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

class ResNet18(nn.Module): # Specialized for CIFAR-10:去掉了maxpool, kernel_size=3
    def __init__(self, num_classes=10, in_channels=3, out_channels=64):
        super(ResNet18, self).__init__()
        # Head
        self.conv_head = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_head = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # Layers
        self.layer1 = nn.Sequential(
            ResBlock(out_channels, out_channels, stride=1),
            ResBlock(out_channels, out_channels, stride=1)
        )
        self.layer2 = nn.Sequential(
            ResBlock(out_channels, out_channels*2, stride=2),
            ResBlock(out_channels*2, out_channels*2, stride=1)
        )
        self.layer3 = nn.Sequential(
            ResBlock(out_channels*2, out_channels*4, stride=2),
            ResBlock(out_channels*4, out_channels*4, stride=1)
        )
        self.layer4 = nn.Sequential(
            ResBlock(out_channels*4, out_channels*8, stride=2),
            ResBlock(out_channels*8, out_channels*8, stride=1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flaten = nn.Flatten()
        self.fc = nn.Linear(out_channels*8, num_classes)

    def forward(self, x):
        out = self.relu(self.bn_head(self.conv_head(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.flaten(out)
        out = self.fc(out)
        return out
        