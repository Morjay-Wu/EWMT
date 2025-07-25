# Enhanced Wavelet Multi-Head Transformer (EWMT) Model

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssim import msssim,ssim

import numpy as np

import pywt

from wavelet import DWT_Haar

from wt_conv import WTConv2d

NUM_BANDS = 4


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.ReplicationPad2d(1),
        nn.Conv2d(in_channels, out_channels, 3, stride=stride)
    )


# --- Compound Loss Function ---
class CompoundLoss(nn.Module):
    def __init__(self, pretrained, alpha=0.85, normalize=True):
        super(CompoundLoss, self).__init__()
        self.pretrained = pretrained
        self.alpha = alpha
        self.normalize = normalize
        # Two-level discrete wavelet transform
        self.DWT = DWT_Haar()
        
    def forward(self, prediction, target):
        # First level of wavelet decomposition
        prediction_LL, prediction_HL, prediction_LH, prediction_HH = self.DWT(prediction)
        target_LL, target_HL, target_LH, target_HH = self.DWT(target)
        
        # Second level of wavelet decomposition (on the LL sub-band)
        prediction_LL_LL, prediction_LL_HL, prediction_LL_LH, prediction_LL_HH = self.DWT(prediction_LL)
        target_LL_LL, target_LL_HL, target_LL_LH, target_LL_HH = self.DWT(target_LL)
        
        # Level 1 high-frequency loss
        level1_H_loss = (
            F.l1_loss(prediction_HL, target_HL) +
            F.l1_loss(prediction_LH, target_LH) +
            F.l1_loss(prediction_HH, target_HH)
        )
        
        # Level 2 loss
        level2_LL_loss = F.l1_loss(prediction_LL_LL, target_LL_LL)
        level2_H_loss = (
            F.l1_loss(prediction_LL_HL, target_LL_HL) +
            F.l1_loss(prediction_LL_LH, target_LL_LH) +
            F.l1_loss(prediction_LL_HH, target_LL_HH)
        )
        
        # Weighted combination, giving higher weight to second-level details
        wavelet_loss = level1_H_loss + 1.2 * level2_LL_loss + 1.2 * level2_H_loss
        
        # Other loss components
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

class ChannelAttention(nn.Module):
    """
    Standard Channel Attention mechanism.
    It computes channel weights using global pooling and an MLP.
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Ensure the intermediate channel number is at least 1
        reduced_channels = max(1, in_planes // ratio)
        
        self.fc1 = nn.Conv2d(in_planes, reduced_channels, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(reduced_channels, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class MultiHeadSpatialAttention(nn.Module):
    """
    Multi-Head Spatial Attention mechanism.
    Multiple convolutional heads with different kernel sizes process spatial statistics
    independently, and their results are then merged.
    """
    def __init__(self, num_heads=4):
        super(MultiHeadSpatialAttention, self).__init__()
        self.num_heads = num_heads
        
        # Define different kernel sizes and their corresponding padding
        # A combination of 3x3, 5x5, 7x7, and 9x9 kernels is used.
        kernel_sizes = [3, 5, 7, 9]
        paddings = [1, 2, 3, 4]
        
        # Each head has an independent spatial attention convolution with a different kernel size.
        self.spatial_convs = nn.ModuleList()
        for i in range(num_heads):
            # Cycle through the predefined kernel sizes
            kernel_idx = i % len(kernel_sizes)
            kernel_size = kernel_sizes[kernel_idx]
            padding = paddings[kernel_idx]
            
            # Create the convolutional layer
            self.spatial_convs.append(
                nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
            )
        
        # Projection layer to merge the multi-head results
        self.proj = nn.Conv2d(num_heads, 1, kernel_size=1, bias=False)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Calculate global spatial statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        spatial_info = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        
        # Apply each head's spatial attention
        head_outputs = []
        for conv in self.spatial_convs:
            # Each head processes the same spatial information independently
            head_out = conv(spatial_info)  # [B, 1, H, W]
            head_outputs.append(head_out)
        
        # Concatenate all head outputs [B, num_heads, H, W]
        multi_head_attention = torch.cat(head_outputs, dim=1)
        
        # Project and merge multi-head results
        fused_attention = self.proj(multi_head_attention)  # [B, 1, H, W]
        
        # Apply sigmoid activation
        attention_weights = self.sigmoid(fused_attention)
        
        # Apply attention weights
        output = x * attention_weights
        
        return output

class HybridAttentionResidualBlock(nn.Module):
    """
    Hybrid Attention Residual Block.
    Combines standard channel attention with multi-head spatial attention.
    """
    def __init__(self, in_channels, out_channels, num_spatial_heads=4):
        super(HybridAttentionResidualBlock, self).__init__()
        
        # Main path convolution
        self.conv = Conv3X3WithPadding(in_channels, out_channels)
        self.relu = nn.ReLU(True)
        
        # Channel attention (standard single-head)
        self.channel_attention = ChannelAttention(out_channels)
        
        # Multi-head spatial attention
        self.spatial_attention = MultiHeadSpatialAttention(
            num_heads=num_spatial_heads
        )
        
        # Shortcut connection (if input/output channels differ)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, inputs):
        # Main path
        x = self.conv(inputs)
        x = self.relu(x)
        
        # Apply channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Apply multi-head spatial attention
        x = self.spatial_attention(x)
        
        # Residual connection
        residual = self.shortcut(inputs)
        output = x + residual
        
        return output

class RFEncoder(nn.Sequential):
    """
    Encoder using the Hybrid Attention Residual Blocks.
    """
    def __init__(self):
        channels = [NUM_BANDS*2, 32, 64, 128]
        super(RFEncoder, self).__init__(
            HybridAttentionResidualBlock(channels[0], channels[1], num_spatial_heads=4),
            HybridAttentionResidualBlock(channels[1], channels[2], num_spatial_heads=4),
            HybridAttentionResidualBlock(channels[2], channels[3], num_spatial_heads=4),
        )

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
    """
    A simple pretrained network used for feature extraction in the loss function.
    """
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
        self.RFEncoder = RFEncoder()
        self.RFDecoder = RFDecoder()
        
        # Add the wavelet convolution layer
        self.wt_conv = WTConv2d(NUM_BANDS, NUM_BANDS, wt_levels=2).to(self.device)

    def forward(self, inputs):
        # inputs[0]: Coarse reference (e.g., Modis_ref)
        # inputs[1]: Fine reference (e.g., Land_ref)
        # inputs[-1]: Coarse prediction date (e.g., Modis_pre)

        Land_ref = inputs[1]
        Modis_pre = inputs[-1]
        res_data = Modis_pre - Land_ref

        # Apply wavelet transform using WTConv2d
        Land_wt = self.wt_conv(Land_ref)
        res_wt = self.wt_conv(res_data)

        # High-resolution feature extraction
        Landsat_encoder = self.Landsat_encoder(Land_wt)
        MLRes_encoder = self.MLRes_encoder(res_wt)

        # Feature fusion
        fused_features = Landsat_encoder + MLRes_encoder

        # Reconstruct the predicted image from fused features
        result_pre = self.decoder(fused_features)

        # Adaptive enhancement using the hybrid attention mechanism
        result_rfe = self.RFEncoder(torch.cat((result_pre, Land_ref), 1))
        result = self.RFDecoder(result_rfe)

        return result
    
if __name__ == "__main__":
    print("Starting model test...")
    
    # 1. Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[1] Using device: {device}")
    
    # 2. Model initialization
    print("\n[2] Initializing model...")
    model = FusionNet_2().to(device)
    
    # 3. Print model summary
    print("\n[3] Model Information:")
    print("-" * 50)
    print(model)
    print("-" * 50)
    
    # 4. Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[4] Model Parameter Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    # 5. Prepare test data
    print("\n[5] Preparing test data...")
    test_configs = [
        {"batch_size": 1, "height": 64, "width": 64},
        {"batch_size": 2, "height": 128, "width": 128},
        {"batch_size": 1, "height": 256, "width": 256}
    ]
    
    # 6. Model testing
    print("\n[6] Starting model forward pass tests...")
    model.eval()
    
    last_output = None
    last_land_ref = None
    
    for config in test_configs:
        print(f"\nTesting configuration: batch_size={config['batch_size']}, size={config['height']}x{config['width']}")
        try:
            # Create dummy test data
            modis_ref = torch.randn(config['batch_size'], NUM_BANDS, 
                                  config['height'], config['width']).to(device)
            land_ref = torch.randn(config['batch_size'], NUM_BANDS, 
                                 config['height'], config['width']).to(device)
            modis_pre = torch.randn(config['batch_size'], NUM_BANDS, 
                                  config['height'], config['width']).to(device)
            
            inputs = [modis_ref, land_ref, modis_pre]
            
            # Test forward pass
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                output = model(inputs)
                end_time.record()
                
                last_output = output
                last_land_ref = land_ref
                
                torch.cuda.synchronize()
                
                elapsed_time = start_time.elapsed_time(end_time)
                
                if torch.cuda.is_available():
                    max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
                
                print(f"✓ Test passed")
                print(f"  - Input shapes: {[x.shape for x in inputs]}")
                print(f"  - Output shape: {output.shape}")
                print(f"  - Inference time: {elapsed_time:.2f} ms")
                print(f"  - Peak GPU memory: {max_memory:.2f} MB")
                
                # Test wavelet transform
                print("\n  Testing Wavelet Transform:")
                Land_wt = model.wt_conv(land_ref)
                print(f"  - Shape after WT: {Land_wt.shape}")
                
                # Test attention mechanisms
                print("\n  Testing Attention Mechanisms:")
                ca = ChannelAttention(NUM_BANDS).to(device)
                ca_output = ca(land_ref)
                print(f"  - Channel Attention output shape: {ca_output.shape}")
                
                msa = MultiHeadSpatialAttention(num_heads=4).to(device)
                msa_output = msa(land_ref)
                print(f"  - Multi-Head Spatial Attention output shape: {msa_output.shape}")
                
                # Test hybrid attention residual block
                hybrid_block = HybridAttentionResidualBlock(NUM_BANDS, 32, num_spatial_heads=4).to(device)
                hybrid_output = hybrid_block(land_ref)
                print(f"  - Hybrid Attention Block output shape: {hybrid_output.shape}")
                
        except Exception as e:
            print(f"✗ Test failed")
            print(f"  - Error: {str(e)}")
            
    # 7. Test loss function
    print("\n[7] Testing loss function...")
    try:
        if last_output is not None and last_land_ref is not None:
            pretrained = Pretrained().to(device)
            criterion = CompoundLoss(pretrained=pretrained).to(device)
            
            with torch.no_grad():
                loss = criterion(last_output, last_land_ref)
                print(f"✓ Loss function test passed")
                print(f"  - Calculated loss: {loss.item():.4f}")
        else:
            print("✗ Loss function test failed")
            print("  - Error: No output data available for testing.")
    except Exception as e:
        print(f"✗ Loss function test failed")
        print(f"  - Error: {str(e)}")
    
    print("\nTest completed!")