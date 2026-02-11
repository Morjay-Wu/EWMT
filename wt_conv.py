import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import pywt
import pywt.data


'''
小波变换卷积 (WTConv) 实现原理：

1. 整体架构设计：
   WTConv 利用小波变换的级联分解，对输入的不同频率带进行一系列小核卷积。先使用小波变换(WT)对输入的
   低频和高频内容进行滤波和下采样，然后在不同频率图上进行小核深度卷积，最后通过逆小波变换(IWT)构建输出。
   通过这种方式，使小核卷积能在更大的输入区域上操作，扩大感受野。

2. 主要组件和工作流程：
   a. 小波分解：将输入信号分解为低频(LL)和高频(LH,HL,HH)分量
   b. 多层级处理：对低频分量继续进行小波分解，形成多尺度表示
   c. 特征提取：在各个频率分量上分别应用卷积操作
   d. 特征融合：通过逆小波变换将处理后的多尺度特征重建为原始尺寸

3. 优势：
   a. 扩大感受野：小波变换天然具有多尺度特性，使网络能够捕获更大范围的上下文信息
   b. 频率分离：明确分离高频和低频内容，更有针对性地处理不同频率特征
   c. 高效计算：相比大卷积核，多尺度小波分解+小卷积核组合计算效率更高
'''


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    """
    创建小波变换和逆变换所需的滤波器
    
    参数:
        wave: 小波类型，如'db1'(Haar小波),'db2'等
        in_size: 输入通道数
        out_size: 输出通道数
        type: 数据类型
        
    返回:
        dec_filters: 分解滤波器(用于小波变换)
        rec_filters: 重建滤波器(用于逆小波变换)
    """
    # 获取指定类型的小波
    w = pywt.Wavelet(wave)
    
    # 创建分解高通和低通滤波器
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)  # 高通滤波器
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)  # 低通滤波器
    
    # 构建四个滤波器 (LL, LH, HL, HH) 用于2D小波变换
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),   # LL: 低频-低频
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),   # LH: 低频-高频
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),   # HL: 高频-低频
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)],  # HH: 高频-高频
                              dim=0)
    
    # 复制滤波器到所有输入通道
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    # 创建重建高通和低通滤波器
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])  # 高通滤波器
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])  # 低通滤波器
    
    # 构建四个滤波器用于2D逆小波变换
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),   # LL
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),   # LH
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),   # HL
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)],  # HH
                              dim=0)
    
    # 复制滤波器到所有输出通道
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    """
    执行2D小波变换
    
    参数:
        x: 输入张量 [批次大小, 通道数, 高度, 宽度]
        filters: 小波分解滤波器
        
    返回:
        变换后的张量 [批次大小, 通道数, 4(LL,LH,HL,HH), 高度/2, 宽度/2]
    """
    b, c, h, w = x.shape
    # 保证滤波器与输入在同一设备
    if filters.device != x.device:
        filters = filters.to(x.device)
    # 计算滤波器的padding
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    # 应用卷积进行小波变换，使用stride=2实现下采样
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    # 重塑张量以便于后续处理，分离四个子带
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x

def inverse_wavelet_transform(x, filters):
    """
    执行2D逆小波变换
    
    参数:
        x: 输入张量 [批次大小, 通道数, 4(LL,LH,HL,HH), 高度/2, 宽度/2]
        filters: 小波重建滤波器
        
    返回:
        重建后的张量 [批次大小, 通道数, 高度, 宽度]
    """
    b, c, _, h_half, w_half = x.shape
    # 保证滤波器与输入在同一设备
    if filters.device != x.device:
        filters = filters.to(x.device)
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    # 重塑张量，合并通道和子带维度
    x = x.reshape(b, c * 4, h_half, w_half)
    # 使用转置卷积进行上采样和重建
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class WTConv2d(nn.Module):
    """
    小波变换卷积层
    
    将输入先通过小波变换分解，在不同频率子带上应用卷积，然后通过逆小波变换重建
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        """
        初始化WTConv2d层
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数（当前实现要求等于输入通道数）
            kernel_size: 卷积核大小
            stride: 步长
            bias: 是否使用偏置
            wt_levels: 小波变换的层级数
            wt_type: 小波类型，默认为'db1'(Haar小波)
        """
        super(WTConv2d, self).__init__()

        # 当前实现要求输入输出通道数相同
        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels  # 小波变换的层级数
        self.stride = stride
        self.dilation = 1
        self.kernel_size = kernel_size

        # 创建小波变换和逆变换滤波器
        dec_filters, rec_filters = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        # 作为 buffer 注册，随模块一起搬迁设备，且不参与优化器
        self.register_buffer('wt_filter', dec_filters)
        self.register_buffer('iwt_filter', rec_filters)

        # 定义小波变换和逆变换函数（不捕获静态Tensor，避免设备不一致）
        self.wt_function = None
        self.iwt_function = None

        # 基本卷积层（应用于原始输入）
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size // 2, stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        # 缩放因子
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        # 在各个小波层级上应用的卷积层
        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding=kernel_size // 2, stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        # 各小波层级的缩放因子，初始值较小以稳定训练
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        # 处理步长>1的情况
        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量 [批次大小, 通道数, 高度, 宽度]
            
        返回:
            处理后的张量 [批次大小, 通道数, 高度/stride, 宽度/stride]
        """
        # 存储各层级的低频和高频分量
        x_ll_in_levels = []  # 各层级的低频分量
        x_h_in_levels = []   # 各层级的高频分量
        shapes_in_levels = [] # 各层级的形状

        # 当前层级的低频分量，初始为输入x
        curr_x_ll = x

        # 自顶向下，逐级进行小波分解
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            
            # 确保尺寸是偶数，否则进行填充
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            # 小波变换
            # 使用缓冲区中的滤波器，并在函数内部对齐设备
            curr_x = wavelet_transform(curr_x_ll, self.wt_filter)
            # 提取低频分量(LL)
            curr_x_ll = curr_x[:, :, 0, :, :]

            # 对所有分量应用卷积变换
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            # 存储处理后的低频和高频分量
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])  # LL
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :]) # LH, HL, HH

        # 自底向上，逐级进行逆小波重建
        next_x_ll = 0  # 初始化下一级的低频分量

        for i in range(self.wt_levels - 1, -1, -1):
            # 获取当前层级的分量和形状
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            # 将下一级的低频分量加到当前低频分量上
            curr_x_ll = curr_x_ll + next_x_ll

            # 重建当前层级
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            # 使用缓冲区中的滤波器，并在函数内部对齐设备
            next_x_ll = inverse_wavelet_transform(curr_x, self.iwt_filter)

            # 裁剪到原始尺寸
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        # 小波处理后的结果
        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        # 原始卷积路径处理
        x = self.base_scale(self.base_conv(x))
        
        # 合并两个路径的结果
        x = x + x_tag

        # 应用步长
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

class _ScaleModule(nn.Module):
    """
    缩放模块，用于对特征进行加权
    """
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        """
        初始化缩放模块
        
        参数:
            dims: 缩放参数的维度
            init_scale: 缩放参数的初始值
            init_bias: 偏置初始值(未使用)
        """
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        """
        对输入进行缩放
        """
        return torch.mul(self.weight, x)

# 测试代码：输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    # 创建WTConv2d实例
    block = WTConv2d(32, 32)
    # 创建随机输入张量
    input = torch.rand(1, 32, 64, 64)
    # 前向传播
    output = block(input)
    # 打印形状
    print("input.shape:", input.shape)
    print("output.shape:", output.shape)