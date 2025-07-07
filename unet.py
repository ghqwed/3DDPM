# 导入PyTorch相关模块
import torch  # 深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 神经网络函数

# 时间嵌入层: 将时间步t转换为向量表示
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        """
        初始化时间嵌入层
        :param dim: 嵌入向量的维度
        """
        super().__init__()
        self.dim = dim  # 保存嵌入维度
        half_dim = dim // 2  # 实际使用一半维度
        
        # 计算位置编码: 类似Transformer的位置编码
        # 1. 计算对数间隔的频率
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        # 2. 创建指数衰减的频率向量
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        # 3. 注册为缓冲区(不参与训练但会保存到模型状态)
        self.register_buffer('emb', emb)
    
    def forward(self, t):
        """
        前向传播: 将时间步t转换为嵌入向量
        :param t: 时间步张量(形状[batch_size])
        :return: 时间嵌入向量(形状[batch_size, dim])
        """
        # 1. 将时间步与频率向量相乘
        emb = t.float()[:, None] * self.emb[None, :]
        # 2. 拼接正弦和余弦编码
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

# UNet的基本构建块
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, downsample=False):
        """
        初始化UNet块
        :param in_ch: 输入通道数
        :param out_ch: 输出通道数
        :param time_emb_dim: 时间嵌入维度
        :param downsample: 是否进行下采样
        """
        super().__init__()
        # 时间嵌入的MLP(将时间信息映射到特征空间)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.downsample = downsample  # 是否下采样
        
        # 第一个卷积层(保持空间分辨率)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        # 第二个卷积层(根据是否下采样决定步长)
        if downsample:
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
        else:
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
            
        # 批归一化和激活函数
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()  # Swish激活函数
    
    def forward(self, x, t):
        """
        前向传播
        :param x: 输入特征图
        :param t: 时间嵌入向量
        :return: 输出特征图
        """
        # 1. 第一个卷积
        h = self.conv1(x)
        # 2. 处理时间嵌入并加到特征图上
        time_emb = self.act(self.time_mlp(t))
        h = h + time_emb[:, :, None, None]  # 广播到特征图尺寸
        # 3. 第二个卷积
        h = self.conv2(h)
        # 4. 批归一化和激活
        h = self.norm(h)
        h = self.act(h)
        return h

# UNet模型: 用于预测噪声的神经网络
class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, chs=(64, 128, 256, 512), time_emb_dim=128):
        """
        初始化UNet
        :param in_ch: 输入通道数(对于灰度图像为1)
        :param out_ch: 输出通道数(与输入相同)
        :param chs: 各层的通道数列表
        :param time_emb_dim: 时间嵌入维度
        """
        super().__init__()
        # 时间嵌入处理网络
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_emb_dim),  # 时间嵌入层
            nn.Linear(time_emb_dim, time_emb_dim),  # 线性变换
            nn.SiLU(),  # 激活函数
            nn.Linear(time_emb_dim, time_emb_dim)  # 再次线性变换
        )
        
        # 下采样路径(编码器)
        self.down = nn.ModuleList()  # 存储下采样块
        ch_prev = in_ch  # 初始通道数
        for i, ch in enumerate(chs):
            # 除了最后一层都进行下采样
            downsample = i < len(chs)-1  
            # 添加下采样块
            self.down.append(Block(ch_prev, ch, time_emb_dim, downsample))
            ch_prev = ch  # 更新通道数
        
        # 中间层(瓶颈层)
        self.mid = Block(ch_prev, ch_prev, time_emb_dim)
        
        # 上采样路径(解码器)
        self.up = nn.ModuleList()
        for ch in reversed(chs):
            # 上采样块的输入是当前特征图与skip connection的拼接
            self.up.append(Block(ch_prev + ch, ch, time_emb_dim))
            ch_prev = ch  # 更新通道数
        
        # 最终输出层
        self.final = nn.Sequential(
            nn.Conv2d(ch_prev, out_ch, 1),  # 1x1卷积调整通道数
            nn.AdaptiveAvgPool2d((28, 28))  # 确保输出尺寸为28x28(适应MNIST尺寸)
        )
    
    def forward(self, x, t):
        """
        前向传播
        :param x: 输入图像(带噪声)
        :param t: 时间步
        :return: 预测的噪声
        """
        # 1. 处理时间嵌入
        t = self.time_mlp(t)
        
        # 2. 下采样(编码)
        hs = []  # 保存各层的特征图用于skip connection
        for block in self.down:
            x = block(x, t)  # 通过下采样块
            hs.append(x)  # 保存特征图
        
        # 3. 中间层(瓶颈)
        x = self.mid(x, t)
        
        # 4. 上采样(解码)
        for block in self.up:
            # 上采样(扩大特征图尺寸)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            # 获取对应的skip connection
            skip = hs.pop()
            # 确保skip connection和x的尺寸匹配
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='nearest')
            # 拼接skip connection
            x = torch.cat([x, skip], dim=1)
            # 通过上采样块
            x = block(x, t)
        
        # 5. 最终输出
        return self.final(x)
