# 导入必要的库
import torch  # PyTorch深度学习框架
from ddpm_model import DDPM  # 导入DDPM模型
from unet import UNet  # 导入UNet模型
import matplotlib.pyplot as plt  # 绘图库
import os  # 操作系统接口

# 测试参数配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择设备(GPU优先)
model_path = "models/ddpm_epoch_100.pth"  # 训练好的模型路径(假设已完成100个epoch的训练)

# 加载训练好的模型
model = UNet().to(device)  # 初始化UNet并移动到设备
ddpm = DDPM(model).to(device)  # 初始化DDPM(包含UNet)并移动到设备
ddpm.load_state_dict(torch.load(model_path))  # 加载预训练权重
ddpm.eval()  # 设置为评估模式(关闭dropout等训练专用层)

# 生成样本(不计算梯度以节省内存)
with torch.no_grad():  # 禁用梯度计算
    # 调用DDPM的sample方法生成样本
    # 参数说明: (16,1,28,28)表示生成16个1通道28x28的图像(适应MNIST尺寸)
    samples = ddpm.sample((16, 1, 28, 28), device)  # 生成16个样本

# 可视化生成的样本
fig, axes = plt.subplots(4, 4, figsize=(8, 8))  # 创建4x4的子图
for i, ax in enumerate(axes.flat):
    # 显示第i个样本
    ax.imshow(samples[i].cpu().squeeze(), cmap='gray')  # 转换为CPU数据并去除维度1
    ax.axis('off')  # 关闭坐标轴
plt.tight_layout()  # 自动调整子图间距
plt.savefig('generated_samples.png')  # 保存图像到文件
plt.show()  # 显示图像

print("测试完成! 生成的样本已保存为 generated_samples.png")  # 输出完成信息
