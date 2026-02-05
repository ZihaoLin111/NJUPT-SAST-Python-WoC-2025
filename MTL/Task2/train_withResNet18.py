import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from model import ResBlock, ResNet18
import os, time, datetime
import tqdm
import json

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4), # 随机裁剪
    torchvision.transforms.RandomHorizontalFlip(), # 随机旋转
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

transform_val = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

full_trainset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True,
    transform=transform_train
)

trainset, valset = torch.utils.data.random_split(full_trainset, [45000, 5000])
valset.dataset.transform = transform_val

trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=256, 
    shuffle=True, 
    num_workers=4
)
valloader = torch.utils.data.DataLoader(
    valset,
    batch_size=256,
    shuffle=False,
    num_workers=4
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
res_net = ResNet18().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    res_net.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4
)

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[100, 125],
    gamma=0.1
)
epochs = 150
train_acc_lst = []
val_acc_lst = []

for epoch in range(epochs):
    res_net.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm.tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = res_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    scheduler.step()

    train_acc = 100. * correct / total
    train_acc_lst.append(train_acc)

    res_net.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = res_net(inputs)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100. * val_correct / val_total
    val_acc_lst.append(val_acc)

    print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

best_train_acc = max(train_acc_lst)
print(f'Best Training Accuracy: {best_train_acc:.2f}%')
best_val_acc = max(val_acc_lst)
print(f'Best Validation Accuracy: {best_val_acc:.2f}%')

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

plt.close()

# 保存json
log_data = {
    'network': 'ResNet18',
    'optimizer': 'SGD',
    'epochs': epochs,
    'train_acc': best_train_acc,
    'val_acc': best_val_acc
}
jsonl_path = "runs.jsonl"
with open(jsonl_path, 'a') as f:
    f.write(json.dumps(log_data) + '\n')

# 保存模型
save_folder = './saved_models'
model_name = 'ResNet18'
epoch = epochs
best_train_acc = best_train_acc
best_val_acc = best_val_acc
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

os.makedirs(save_folder, exist_ok=True)
filename = f"{model_name}_{timestamp}_ep{epoch}_train_best_acc_{best_train_acc:.3f}val_best_acc{best_val_acc:.3f}.pth"
save_path = os.path.join(save_folder, filename)
torch.save(res_net.state_dict(), save_path)

print(f"模型参数已保存至: {save_path}")