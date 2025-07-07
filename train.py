# 导入必要的库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 神经网络函数
import torch.optim as optim  # 优化器
from torchvision import datasets, transforms  # 数据集和图像变换
from torch.utils.data import DataLoader  # 数据加载器
from ddpm_model import DDPM  # 导入DDPM模型
from unet import UNet  # 导入UNet模型
import os  # 操作系统接口

# 训练参数配置
batch_size = 128  # 每批数据量
epochs = 100  # 训练轮数
lr = 1e-3  # 学习率
# 自动选择设备(GPU优先)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = UNet().to(device)  # 创建UNet模型并移动到设备
ddpm = DDPM(model).to(device)  # 创建DDPM模型(包含UNet)并移动到设备

# 设置优化器(Adam优化器)
optimizer = optim.Adam(ddpm.parameters(), lr=lr)  # 优化DDPM的所有参数

# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 将像素值归一化到[-1,1]范围
])

# 加载FashionMNIST数据集
train_dataset = datasets.FashionMNIST(
    root='./data',  # 数据存储路径
    train=True,  # 使用训练集
    download=True,  # 如果不存在则下载
    transform=transform  # 应用上述变换
)

# 创建数据加载器
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size,  # 按批次加载
    shuffle=True  # 打乱数据顺序
)

# 训练循环
for epoch in range(epochs):
    # 遍历所有批次数据
    for batch_idx, (data, _) in enumerate(train_loader):
        # 1. 准备数据
        data = data.to(device)  # 将数据移动到指定设备
        
        # 2. 随机采样时间步(为每个样本随机选择不同的时间步)
        t = torch.randint(0, ddpm.timesteps, (data.size(0),), device=device)
        
        # 3. 前向扩散过程: 向数据添加噪声
        noisy_data, noise = ddpm(data, t)
        
        # 4. 预测噪声: 使用UNet预测添加的噪声
        pred_noise = ddpm.reverse(noisy_data, t)
        
        # 5. 计算损失: 比较预测噪声和真实噪声
        loss = F.mse_loss(pred_noise, noise)  # 均方误差损失
        
        # 6. 反向传播和优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数
        
        # 每100个批次打印一次训练信息
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch+1}/{epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')
    
    # 7. 模型保存: 每10个epoch保存一次
    if (epoch + 1) % 10 == 0:
        os.makedirs('models', exist_ok=True)  # 创建模型保存目录
        torch.save(ddpm.state_dict(), f'models/ddpm_epoch_{epoch+1}.pth')  # 保存模型参数

print("训练完成!")  # 训练结束提示
