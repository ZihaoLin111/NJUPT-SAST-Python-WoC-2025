import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from model import Net

model_path = './saved_models/Net_20251221_1533_ep10_acc65.660.pth'

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    # 归一化参数来自FindNormParam.ipynb计算结果
])

testset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False,
    download=True,
    transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset, 
    batch_size=128, 
    shuffle=False, 
    num_workers=4
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
state_dict = torch.load(model_path, map_location=device)
net.load_state_dict(state_dict)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

acc = 100 * correct / total
print(f'Test Accuracy: {acc:.2f}%')