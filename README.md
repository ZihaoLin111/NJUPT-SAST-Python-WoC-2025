# NJUPT-SAST-Python-WoC-2025

这是 NJUPT SAST 2025 年 Python 方向 WoC 仓库。

点击 [这里](https://njupt-sast.feishu.cn/wiki/GS5Vw6IDNiieVIkcf9MczzHInVf) 查看题目详情和提交要求。

## 提交指引

首先 fork 本仓库，在分支中进行开发，完成后提交 PR 到本仓库的对应分支即可。

> 由于题目较难，可以开发中途提交 PR，并且可以进行询问，标明“进行中”即可，评审时会酌情考虑。

---

## 方向选择

-   [x] 学术向
-   [ ] 工具向
-   [ ] 机器学习向
-   [ ] 开发向
-   [ ] 爬虫向

目前进度:

-   [ ] 计划中
-   [x] 进行中
-   [ ] 已完成

## 运行指南
1. 下载本项目代码
    ```bash
    git clone https://github.com/ZihaoLin111/NJUPT-SAST-Python-WoC-2025.git
    ```
2. 安装Python依赖
    ```bash
    # 使用uv（推荐）
    uv sync

    # 使用pip
    pip install .
    ```



## 任务说明

### 1.引言

在实际应用中，真实世界噪声会严重降低预训练分类模型的准确率。本项目探索了一种多任务学习方法，通过冻结预训练的分类任务网络，微调去噪网络，使得去噪网络能够针对性输出最易于分类网络识别的图像

### 2.相关工作

#### 2.1.去噪神经网络
对于去噪任务，我参考了DnCNN[^1]

DnCNN模型分为三部分，默认总层数nums_layers=17。第一部分，输入层，由Conv+ReLU组成，卷积核大小为3\*3，padding=1，映射至64通道；第二部分，中间层，其中包含nums_layer-2个重复的小层，每小层由Conv+BN+ReLU组成；第三部分，Conv做为输出层，重新映射回3通道

DnCNN预测的是图片的噪声，最后输出结果为“含噪声的原图” - “模型预测的噪声”

#### 2.2.分类神经网络
对于分类任务，我参考了ResNet18[^2]

该模型的基本组成单位为残差块，由Conv+BN+ReLU+Conv+BN，卷积核大小为3\*3，当输入维度与输出维度不一致时额外执行Conv+BN，卷积核大小为1\*1，以匹配维度，最后ReLU输出

为了适应CIFAR10数据集的32\*32低分辨率输入，我将起始层的7\*7卷积改为了3\*3卷积，并删除了MaxPooling

在残差块的基础上，网络分为四个阶段，每阶段由两个残差块组成，从第二阶段开始，每阶段残差块步长设为2，使得空间尺寸减半，特征通道数加倍

最后通过平均池化层和全连接层输出10个类别的分布

### 3.方法

#### 3.1.预训练

我分别使用了SIDD_Small_sRGB数据集[^3]和CIFAR-10数据集[^4]对DnCNN和ResNet18进行了训练。在150轮训练后DnCNN在验证集上的最好去噪成绩为PSNR:32.56dB，SSIM:0.83；ResNet18在验证集上的准确率为84.4%

对于SIDD_Small_sRGB数据集，我将每次输入的带有噪声图像和真实图像随机切割出32\*32的小块，以使得模型适应该尺寸的感受野，此外我采用了反复输入整个数据集40次和随机旋转的数据增强方法。对于CIFAR-10数据集，我采用了随机旋转和随机偏移的数据增强方法

#### 3.2.多任务学习

此项目中我采用了CIFAR10-C数据集[^5]用于多任务学习，该数据集包含了CIFAR-10测试集的10种不同类型的常见图像损坏，每种类型又包含了5个不同程度的损坏，分别为1-5级，级别越高损坏程度越严重。该项目中使用的损坏类型为Gaussian Noise，级别为3

训练时我将损坏图像输入到DnCNN中，得到去噪后的图像，再将去噪后的图像输入到ResNet18中，得到分类结果。损失函数由两部分组成：去噪损失 $\mathcal{L}_{D}$ 和分类损失 $\mathcal{L}_{C}$ ，去噪损失使用均方误差（MSE），分类损失使用交叉熵损失（Cross-Entropy Loss）。在训练过程中，我冻结了ResNet18的参数，只更新DnCNN的参数，以使得DnCNN能够针对性地输出最易于ResNet18识别的图像。

#### 3.3.Loss设置
在训练中我使用了三种权重组合:
1）固定权重
$$\mathcal{L}_{Total} = \mathcal{L}_D + \mathcal{L}_C * 0.5$$
2）动态权重 $\omega$ =[0.5, 0.75, 1.0]，根据epoch数而改变
$$\mathcal{L}_{Total} = \mathcal{L}_D + \mathcal{L}_C * \omega$$
3）不确定权重[^6]
$$\mathcal{L}_{Total} = \frac{1}{2\sigma^2_1}\mathcal{L}_D+\frac{1}{2\sigma^2_2}\mathcal{L}_C+\log(\sigma_1\sigma_2)$$

#### 3.4.训练细节
我使用了Adam优化器，初始学习率为1e-4，使用余弦退火，学习率下限为1e-6

### 4.实验
直接使用预训练的ResNet18在CIFAR10-C的Gaussian Noise上进行测试
|   Severity   |  Accuracy   |
| ---- | ---- |
|   1   |   75.89%   |
|   2   |   62.19%   |
|   3   |   47.77%   |
|   4   |   41.36%   |
|   5   |   35.85%   |

使用多任务学习方法和三种Loss设置微调DnCNN后在CIFAR10-C的Gaussian Noise上进行测试
|   Severity   |  Loss 1 Accuracy   |  Loss 2 Accuracy   |  Loss 3 Accuracy   |
| ---- | ---- | ---- | ---- |
|   1   |   75.40%   |   77.05%   |   75.75%   | 
|   2   |   74.20%   |   73.65%   |   73.45%   |
|   3   |   71.90%   |   72.45%   |   72.60%   |
|   4   |   69.55%   |   68.85%   |   68.35%   |
|   5   |   66.10%   |   64.80%   |   65.30%   |

### 5.结论
通过多任务学习方法微调去噪网络，我们成功提升了预训练分类模型在受损图像上的准确率，尤其是在损坏程度较高的情况下，准确率提升显著。这表明针对性地优化去噪网络以适应分类任务的需求，可以有效增强模型在特定任务场景下的性能。

### 6.局限与未来工作
该项目在低损坏程度的图像上提升有限，未来可以探索更复杂的去噪网络，以进一步提升模型在低损坏程度的图像上的表现。此外，未来还可以尝试在其他类型的损坏上进行类似的多任务学习，以验证方法的泛化能力。

训练时得到的图表表明，验证集的准确率在最初就已经到达较高的水平，但后续的训练并没有带来显著提升，甚至在某些阶段出现了过拟合的迹象。亟需更多实验探索背后原因并改善方法。


[^1]: K. Zhang, W. Zuo, Y. Chen, D. Meng and L. Zhang, "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising," in IEEE Transactions on Image Processing, vol. 26, no. 7, pp. 3142-3155, July 2017.

[^2]: He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.

[^3]: Abdelrahman Abdelhamed, Lin S., Brown M. S. "A High-Quality Denoising Dataset for Smartphone Cameras", IEEE Computer Vision and Pattern Recognition (CVPR), June 2018.

[^4]: Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.

[^5]: Hendrycks D, Dietterich T. Benchmarking neural network robustness to common corruptions and perturbations[J]. arXiv preprint arXiv:1903.12261, 2019.

[^6]: Kendall A, Gal Y, Cipolla R. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 7482-7491.