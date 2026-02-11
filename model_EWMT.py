#改小波、多头注意力、损失固定比值


import torch
import torch.nn as nn
import torch.nn.functional as F

from ssim import msssim,ssim

import numpy as np

import pywt

from dwt_def import dwt2astensor, idwt2astensor
from wavelet import DWT_Haar, IWT_Haar

from wt_conv import WTConv2d

#################### SwinIR-lite helpers (for RFEncoder ablation) ####################
def _pad_to_window_size(x: torch.Tensor, window_size: int):
    B, C, H, W = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, pad_h, pad_w

def _unpad_from_window_size(x: torch.Tensor, pad_h: int, pad_w: int):
    if pad_h > 0:
        x = x[:, :, :-pad_h, :]
    if pad_w > 0:
        x = x[:, :, :, :-pad_w]
    return x

def _window_partition(x: torch.Tensor, window_size: int):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    windows = x.view(-1, C, window_size, window_size)
    return windows

def _window_unpartition(windows: torch.Tensor, window_size: int, B: int, C: int, H: int, W: int):
    num_h = H // window_size
    num_w = W // window_size
    x = windows.view(B, num_h, num_w, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(B, C, H, W)
    return x

class WindowAttentionLite(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 8):
        super(WindowAttentionLite, self).__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x_pad, pad_h, pad_w = _pad_to_window_size(x, self.window_size)
        Bp, Cp, Hp, Wp = x_pad.shape
        qkv = self.qkv(x_pad)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        qw = _window_partition(q, self.window_size)
        kw = _window_partition(k, self.window_size)
        vw = _window_partition(v, self.window_size)

        Bn, Cw, ws, _ = qw.shape
        N = ws * ws
        qw = qw.view(Bn, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        kw = kw.view(Bn, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        vw = vw.view(Bn, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        attn = torch.matmul(qw, kw.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, vw)
        out = out.permute(0, 1, 3, 2).contiguous().view(Bn, Cw, ws, ws)

        xw = _window_unpartition(out, self.window_size, Bp, Cp, Hp, Wp)
        xw = self.proj(xw)
        xw = _unpad_from_window_size(xw, pad_h, pad_w)
        return xw

class SwinLiteBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 8, mlp_ratio: float = 2.0):
        super(SwinLiteBlock, self).__init__()
        self.attn = WindowAttentionLite(dim, num_heads, window_size)
        self.norm1 = nn.GroupNorm(num_groups=min(8, dim), num_channels=dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=False)
        )
        self.norm2 = nn.GroupNorm(num_groups=min(8, dim), num_channels=dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class SwinLiteStage(nn.Module):
    def __init__(self, dim: int, depth: int = 4, num_heads: int = 4, window_size: int = 8):
        super(SwinLiteStage, self).__init__()
        self.blocks = nn.Sequential(*[
            SwinLiteBlock(dim, num_heads, window_size) for _ in range(depth)
        ])

    def forward(self, x):
        return self.blocks(x)

class SwinResidualRFBlock(nn.Module):
    """
    RF阶段的SwinIR样式残差块：Conv3x3投影到目标通道 + GN+GELU + SwinLiteStage + 残差
    """
    def __init__(self, in_channels: int, out_channels: int, depth: int = 4, num_heads: int = 4, window_size: int = 8):
        super(SwinResidualRFBlock, self).__init__()
        self.proj = Conv3X3WithPadding(in_channels, out_channels)
        self.norm = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.act = nn.GELU()
        self.stage = SwinLiteStage(dim=out_channels, depth=depth, num_heads=num_heads, window_size=window_size)
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        y = self.proj(x)
        y = self.act(self.norm(y))
        y = y + self.stage(y)
        return y + self.shortcut(x)

NUM_BANDS = 4


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.ReplicationPad2d(1),
        nn.Conv2d(in_channels, out_channels, 3, stride=stride)
    )


# loss函数**************************
class CompoundLoss(nn.Module):
    def __init__(self, pretrained, alpha=0.85, normalize=True):
        super(CompoundLoss, self).__init__()
        self.pretrained = pretrained
        self.alpha = alpha
        self.normalize = normalize
        # 创建2级小波变换
        self.DWT = DWT_Haar()
        
    def forward(self, prediction, target):
        # 第一级小波分解
        prediction_LL, prediction_HL, prediction_LH, prediction_HH = self.DWT(prediction)
        target_LL, target_HL, target_LH, target_HH = self.DWT(target)
        
        # 第二级小波分解(对LL子带继续分解)
        prediction_LL_LL, prediction_LL_HL, prediction_LL_LH, prediction_LL_HH = self.DWT(prediction_LL)
        target_LL_LL, target_LL_HL, target_LL_LH, target_LL_HH = self.DWT(target_LL)
        
        # 第一级高频损失
        level1_H_loss = (
            F.l1_loss(prediction_HL, target_HL) +
            F.l1_loss(prediction_LH, target_LH) +
            F.l1_loss(prediction_HH, target_HH)
        )
        
        # 第二级损失
        level2_LL_loss = F.l1_loss(prediction_LL_LL, target_LL_LL)
        level2_H_loss = (
            F.l1_loss(prediction_LL_HL, target_LL_HL) +
            F.l1_loss(prediction_LL_LH, target_LL_LH) +
            F.l1_loss(prediction_LL_HH, target_LL_HH)
        )
        
        # 权重调整，第二级细节给予更高权重
        wavelet_loss = level1_H_loss + 1.2 * level2_LL_loss + 1.2 * level2_H_loss
        
        # 其他损失保持不变
        feature_loss = F.l1_loss(self.pretrained(prediction), self.pretrained(target))
        vision_loss = self.alpha * (1.0 - msssim(prediction, target, normalize=self.normalize))
        
        return wavelet_loss + feature_loss + vision_loss

class FEncoder(nn.Sequential):
    def __init__(self):
        channels = [NUM_BANDS, 32, 64, 128]
        super(FEncoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            conv3x3(channels[2], channels[3]),
            nn.ReLU(True)
        )

class REncoder(nn.Sequential):
    def __init__(self):
        channels = [NUM_BANDS, 32, 64, 128]
        super(REncoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            conv3x3(channels[2], channels[3]),
            nn.ReLU(True)
        )

class Decoder(nn.Sequential):
    def __init__(self):
        channels = [128, 64, 32, NUM_BANDS]
        super(Decoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            nn.Conv2d(channels[2], channels[3], 1)
        )


class Conv3X3WithPadding(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv3X3WithPadding, self).__init__(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, stride=stride)
        )
        
# 修改RFEncoder使用新的混合注意力残差块
class RFEncoder(nn.Module):
    def __init__(self, swin_depth: int = 4, swin_heads: int = 4, swin_window: int = 8):
        super(RFEncoder, self).__init__()
        channels = [NUM_BANDS * 2, 32, 64, 128]
        self.blocks = nn.Sequential(
            SwinResidualRFBlock(channels[0], channels[1], depth=swin_depth, num_heads=swin_heads, window_size=swin_window),
            SwinResidualRFBlock(channels[1], channels[2], depth=swin_depth, num_heads=swin_heads, window_size=swin_window),
            SwinResidualRFBlock(channels[2], channels[3], depth=swin_depth, num_heads=swin_heads, window_size=swin_window),
        )

    def forward(self, x):
        return self.blocks(x)

class RFDecoder(nn.Sequential):
    def __init__(self):
        channels = [128, 64, 32, NUM_BANDS]
        super(RFDecoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            nn.Conv2d(channels[2], channels[3], 1)
        )

class Pretrained(nn.Sequential):
    def __init__(self):
        channels = [NUM_BANDS, 32, 64, 128]
        super(Pretrained, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2], 2),
            nn.ReLU(True),
            conv3x3(channels[2], channels[3], 2),
            nn.ReLU(True)
        )

class FusionNet_2(nn.Module):
    def __init__(self):
        super(FusionNet_2, self).__init__()
        self.Landsat_encoder = FEncoder()
        self.MLRes_encoder = REncoder()
        self.decoder = Decoder()
        self.device = torch.device('cuda')
        # 增强模块固定为 SwinIR 风格
        self.RFEncoder = RFEncoder(swin_depth=4, swin_heads=4, swin_window=8)
        self.RFDecoder = RFDecoder()
        
        # 添加小波变换层
        self.wt_conv = WTConv2d(NUM_BANDS, NUM_BANDS, wt_levels=2).to(self.device)

    def forward(self, inputs):
        # inputs[0]:低分参考 Modis_ref
        # inputs[1]:高分参考 Land_ref
        # inputs[-1]:低分预测 Modis_pre

        Land_ref = inputs[1]
        Modis_pre = inputs[-1]
        res_data = Modis_pre - Land_ref

        # 使用WTConv2d进行小波变换
        Land_wt = self.wt_conv(Land_ref)
        res_wt = self.wt_conv(res_data)

        # 高分特征提取
        Landsat_encoder = self.Landsat_encoder(Land_wt)
        MLRes_encoder = self.MLRes_encoder(res_wt)

        # 特征融合
        fused_features = Landsat_encoder + MLRes_encoder

        # 预测特征重构得到预测影像
        result_pre = self.decoder(fused_features)

        # 自适应增强 (使用混合注意力机制)
        result_rfe = self.RFEncoder(torch.cat((result_pre, Land_ref), 1))
        result = self.RFDecoder(result_rfe)

        return result
    
if __name__ == "__main__":
    print("开始模型测试...")
    
    # 1. 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[1] 使用设备: {device}")
    
    # 2. 模型初始化
    print("\n[2] 初始化模型...")
    model = FusionNet_2().to(device)
    
    # 3. 打印模型信息
    print("\n[3] 模型信息:")
    print("-" * 50)
    print(model)
    print("-" * 50)
    
    # 4. 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[4] 模型参数统计:")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"参数比例: {trainable_params/total_params*100:.2f}%")
    
    # 5. 测试数据准备
    print("\n[5] 准备测试数据...")
    test_configs = [
        {"batch_size": 1, "height": 64, "width": 64},
        {"batch_size": 2, "height": 128, "width": 128},
        {"batch_size": 1, "height": 256, "width": 256}
    ]
    
    # 6. 模型测试
    print("\n[6] 开始模型测试...")
    model.eval()  # 设置为评估模式
    
    last_output = None  # 用于存储最后一次的输出，供损失函数测试使用
    last_land_ref = None  # 用于存储最后一次的参考数据，供损失函数测试使用
    
    for config in test_configs:
        print(f"\n测试配置: batch_size={config['batch_size']}, size={config['height']}x{config['width']}")
        try:
            # 创建测试数据
            modis_ref = torch.randn(config['batch_size'], NUM_BANDS, 
                                  config['height'], config['width']).to(device)
            land_ref = torch.randn(config['batch_size'], NUM_BANDS, 
                                 config['height'], config['width']).to(device)
            modis_pre = torch.randn(config['batch_size'], NUM_BANDS, 
                                  config['height'], config['width']).to(device)
            
            inputs = [modis_ref, land_ref, modis_pre]
            
            # 测试前向传播
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                # 记录开始时间
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                output = model(inputs)
                end_time.record()
                
                # 保存最后一次的输出和参考数据，用于损失函数测试
                last_output = output
                last_land_ref = land_ref
                
                # 等待GPU完成
                torch.cuda.synchronize()
                
                # 计算运行时间
                elapsed_time = start_time.elapsed_time(end_time)
                
                # 获取内存使用情况
                if torch.cuda.is_available():
                    max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # 转换为MB
                
                print(f"✓ 测试通过")
                print(f"  - 输入形状: {[x.shape for x in inputs]}")
                print(f"  - 输出形状: {output.shape}")
                print(f"  - 运行时间: {elapsed_time:.2f} ms")
                print(f"  - GPU内存使用: {max_memory:.2f} MB")
                
                # 测试小波变换
                print("\n  测试小波变换:")
                Land_wt = model.wt_conv(land_ref)
                print(f"  - 小波变换后的形状: {Land_wt.shape}")
                
                # 注意力机制示例与混合实现已移除，增强模块固定为 SwinIR 风格
                
        except Exception as e:
            print(f"✗ 测试失败")
            print(f"  错误信息: {str(e)}")
            
    # 7. 测试损失函数
    print("\n[7] 测试损失函数...")
    try:
        if last_output is not None and last_land_ref is not None:
            # 创建预训练模型
            pretrained = Pretrained().to(device)
            criterion = CompoundLoss(pretrained=pretrained).to(device)
            
            # 使用最后一次的输出数据计算损失
            with torch.no_grad():
                loss = criterion(last_output, last_land_ref)
                print(f"✓ 损失函数测试通过")
                print(f"  - 损失值: {loss.item():.4f}")
        else:
            print("✗ 损失函数测试失败")
            print("  错误信息: 没有可用的输出数据进行测试")
    except Exception as e:
        print(f"✗ 损失函数测试失败")
        print(f"  错误信息: {str(e)}")
    
    print("\n测试完成！")