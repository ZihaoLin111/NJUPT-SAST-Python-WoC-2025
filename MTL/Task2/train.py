import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from model import Net
import os, time, datetime

transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), # 随机旋转
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    # 归一化参数来自FindNormParam.ipynb计算结果
])

full_trainset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True,
    transform=transform
)

trainset, valset = torch.utils.data.random_split(full_trainset, [45000, 5000])

trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=128, 
    shuffle=True, 
    num_workers=4
)
valloader = torch.utils.data.DataLoader(
    valset,
    batch_size=128,
    shuffle=False,
    num_workers=4
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

epochs = 10
train_acc_lst = []
val_acc_lst = []

for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total
    train_acc_lst.append(train_acc)

    net.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100. * val_correct / val_total
    val_acc_lst.append(val_acc)

    print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

# 绘制准确率曲线
plt.plot(range(1, epochs+1), train_acc_lst, label='Train Accuracy')
plt.plot(range(1, epochs+1), val_acc_lst, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()

os.makedirs('charts', exist_ok=True)
timestamp = time.strftime('%Y%m%d-%H%M%S')
fig_path = f"charts/train_val_acc_{timestamp}.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"图表已保存至 {fig_path}")

plt.show()

# 保存模型
save_folder = './saved_models'
model_name = 'Net'
epoch = epochs
val_acc = val_acc_lst[-1]
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

os.makedirs(save_folder, exist_ok=True)
filename = f"{model_name}_{timestamp}_ep{epoch}_acc{val_acc:.3f}.pth"
save_path = os.path.join(save_folder, filename)
torch.save(net.state_dict(), save_path)

print(f"模型参数已保存至: {save_path}")