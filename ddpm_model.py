# 导入必要的PyTorch库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 神经网络函数
from math import sqrt  # 数学平方根函数

# 定义DDPM(去噪扩散概率模型)类
class DDPM(nn.Module):
    def __init__(self, model, beta_start=1e-4, beta_end=0.02, timesteps=1000):
        """
        初始化DDPM模型
        :param model: 用于预测噪声的神经网络模型
        :param beta_start: beta调度的起始值(控制噪声添加速度)
        :param beta_end: beta调度的结束值
        :param timesteps: 扩散过程的总时间步数
        """
        super().__init__()  # 调用父类初始化
        self.model = model  # 保存噪声预测模型
        self.timesteps = timesteps  # 保存时间步数
        
        # 注册缓冲区(这些张量会被保存到模型状态中但不需要梯度)
        # beta调度: 控制每个时间步添加的噪声量
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, timesteps))
        # alpha = 1 - beta: 表示保留原始数据的比例
        self.register_buffer('alphas', 1. - self.betas)
        # alpha的累积乘积: 表示经过多个时间步后保留原始数据的比例
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        # alpha累积乘积的平方根: 用于前向扩散过程
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        # 1-alpha累积乘积的平方根: 用于前向扩散过程
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))
    
    def forward(self, x, t):
        """
        前向扩散过程: 逐步向数据添加噪声
        :param x: 输入数据(通常是图像)
        :param t: 当前时间步(决定添加多少噪声)
        :return: 噪声数据, 添加的噪声
        """
        # 生成与输入数据形状相同的随机噪声
        noise = torch.randn_like(x)
        
        # 获取当前时间步的alpha累积乘积平方根
        # view(-1,1,1,1)是为了广播到与x相同的维度
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        # 获取当前时间步的1-alpha累积乘积平方根
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # 根据扩散公式添加噪声:
        # 噪声数据 = sqrt(alpha_cumprod) * 原始数据 + sqrt(1-alpha_cumprod) * 噪声
        noisy_x = sqrt_alpha_cumprod_t * x + sqrt_one_minus_alpha_cumprod_t * noise
        
        return noisy_x, noise  # 返回噪声数据和添加的噪声
    
    def reverse(self, x, t):
        """
        反向去噪过程: 预测并去除噪声
        :param x: 噪声数据
        :param t: 当前时间步
        :return: 预测的噪声
        """
        # 使用模型预测噪声
        # 模型通常是U-Net等结构，输入噪声数据和当前时间步，输出预测的噪声
        return self.model(x, t)
    
    def sample(self, shape, device):
        """
        从模型生成样本(图像)
        :param shape: 生成样本的形状(如[批大小, 通道数, 高度, 宽度])
        :param device: 计算设备(cpu或cuda)
        :return: 生成的样本
        """
        # 从标准正态分布初始化随机噪声
        x = torch.randn(shape, device=device)
        
        # 从最后一步开始逐步去噪
        for t in reversed(range(self.timesteps)):
            with torch.no_grad():  # 禁用梯度计算(节省内存)
                # 1. 预测当前时间步的噪声
                pred_noise = self.reverse(x, torch.full((shape[0],), t, device=device, dtype=torch.long))
                
                # 获取当前时间步的参数
                alpha_t = self.alphas[t]  # 当前时间步的alpha值
                alpha_cumprod_t = self.alphas_cumprod[t]  # 累积alpha值
                beta_t = self.betas[t]  # 当前时间步的beta值
                
                # 2. 决定是否添加额外噪声(最后一步不加噪声)
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                # 3. 根据去噪公式更新x:
                # x = (x - (1-alpha_t)/sqrt(1-alpha_cumprod_t)*pred_noise)/sqrt(alpha_t) + sqrt(beta_t)*noise
                x = 1 / torch.sqrt(alpha_t) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * pred_noise) + torch.sqrt(beta_t) * noise
        
        # 返回最终生成的样本
        return x
