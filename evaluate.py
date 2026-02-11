import numpy as np
import rasterio
from pathlib import Path
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def calculate_sam(img_tgt, img_fus):
    """计算光谱角映射器 (SAM)

    Args:
        img_tgt (numpy.ndarray): 目标图像，形状通常为 (C, H, W) 或 (B, C, H, W)。
        img_fus (numpy.ndarray): 融合/预测图像，形状与 img_tgt 相同。

    Returns:
        float: 平均光谱角 (度数)。
    """
    # 移除可能存在的批次维度 (如果输入是单个样本)
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)

    # 检查维度是否为 (C, H, W)
    if img_tgt.ndim != 3 or img_fus.ndim != 3:
        raise ValueError(f"输入图像维度应为 3 (C, H, W)，但得到 Tgt: {img_tgt.ndim}, Fus: {img_fus.ndim}")

    C, H, W = img_tgt.shape

    # --- 移除错误的全局最大值归一化 ---
    # img_tgt = img_tgt / np.max(img_tgt)
    # img_fus = img_fus / np.max(img_fus)
    # -----------------------------------

    # 将图像重塑为 (H*W, C)，每行是一个像素的光谱向量
    target_pixels = img_tgt.reshape(C, -1).T  # (H*W, C)
    pred_pixels = img_fus.reshape(C, -1).T    # (H*W, C)

    # 对每个像素计算点积
    dot_products = np.sum(target_pixels * pred_pixels, axis=1)  # (H*W,)

    # 计算每个像素的 L2 范数
    norm_target = np.linalg.norm(target_pixels, axis=1)  # (H*W,)
    norm_pred = np.linalg.norm(pred_pixels, axis=1)      # (H*W,)

    # 计算余弦角，添加 epsilon 防止除零
    epsilon = 1e-8 # 一个很小的正数，用于数值稳定性
    denominator = norm_target * norm_pred + epsilon
    cos_angles = dot_products / denominator

    # 使用 np.clip 将结果限制在 arccos 的有效输入范围 [-1, 1]
    cos_angles = np.clip(cos_angles, -1.0, 1.0)

    # 计算角度（弧度）
    angles_rad = np.arccos(cos_angles)

    # 转换为度数
    angles_deg = angles_rad * 180.0 / np.pi

    # 计算所有像素角度的平均值
    # 注意：如果存在零向量像素导致 norm 为0，epsilon 会防止除零错误，
    # 但对应的 cos_angle 可能接近 0 (如果点积也接近0)，角度接近 90 度。
    # 如果需要严格排除零向量像素的计算，可以在 reshape 后进行过滤。
    # 这里使用标准平均值。
    mean_sam_deg = np.mean(angles_deg)
    mean_angles_rad = np.mean(angles_rad)
    return mean_sam_deg

def calculate_ergas(target, pred, ratio=0.3):
    """计算ERGAS (Relative Dimensionless Global Error)"""
    # 每个波段的RMSE
    rmse_per_band = np.sqrt(np.mean((target - pred)**2, axis=(1, 2)))
    # 每个波段的均值
    mean_per_band = np.mean(target, axis=(1, 2))
  
    # 防止除以零
    mean_per_band[mean_per_band == 0] = 1e-8
  
    # 计算每个波段的误差比
    error_ratio = (rmse_per_band / mean_per_band)**2
  
    return 100 * ratio * np.sqrt(np.mean(error_ratio))


def calculate_metrics(pred_path, target_path, norm_value=10000):
    """计算所有评估指标"""
    # 读取图像
    with rasterio.open(pred_path) as src:
        pred = src.read().astype(np.float32) / norm_value  # (C, H, W)
  
    with rasterio.open(target_path) as src:
        target = src.read().astype(np.float32) / norm_value

    # 验证形状
    assert pred.shape == target.shape, f"形状不匹配: Pred {pred.shape} vs Target {target.shape}"
  
    # 转换为[0,1]范围
    pred = np.clip(pred, 0, 1)
    target = np.clip(target, 0, 1)
  
    # 初始化结果字典
    metrics = {}
  
    # 计算RMSE
    metrics['RMSE'] = np.sqrt(np.mean((pred - target)**2))
  
    # 计算PSNR (分波段计算后平均)
    psnr_values = [psnr(target[b], pred[b], data_range=1.0) 
                  for b in range(target.shape[0])]
    metrics['PSNR'] = np.mean(psnr_values)
  
    # 计算SSIM (分波段计算后平均)
    ssim_values = []
    for b in range(target.shape[0]):
        ssim_val = ssim(target[b], pred[b], data_range=1.0,
                       win_size=11, channel_axis=None)
        ssim_values.append(ssim_val)
    metrics['SSIM'] = np.mean(ssim_values)
  
    # 计算相关系数
    metrics['Correlation'] = np.corrcoef(pred.flatten(), target.flatten())[0,1]
    # 添加CC指标（与Correlation相同）
    metrics['CC'] = metrics['Correlation']
  
    # 计算SAM
    metrics['SAM'] = calculate_sam(target, pred)
  
    # 计算ERGAS (假设分辨率比为6，可根据实际情况修改)
    metrics['ERGAS'] = calculate_ergas(target, pred, ratio=2/35)
  
    return metrics

def batch_evaluate(test_dir, pred_suffix="01_L.tif", gt_suffix="01_L_gt.tif"):
    """批量评估文件夹中的所有样本"""
    test_dir = Path(test_dir)
    results = []
  
    # 遍历所有子目录
    for sample_dir in test_dir.iterdir():
        if not sample_dir.is_dir():
            continue
          
        # 构建文件路径
        pred_path = sample_dir / pred_suffix
        gt_path = sample_dir / gt_suffix
      
        # 跳过不完整样本
        if not pred_path.exists() or not gt_path.exists():
            print(f"跳过不完整样本: {sample_dir.name}")
            continue
          
        try:
            # 计算指标
            metrics = calculate_metrics(str(pred_path), str(gt_path))
            metrics['Sample'] = sample_dir.name
            results.append(metrics)
            print(f"已处理: {sample_dir.name}")
        except Exception as e:
            print(f"处理失败 {sample_dir.name}: {str(e)}")
  
    # 转换为DataFrame
    df = pd.DataFrame(results)
  
    # 计算平均值
    avg = df.mean(numeric_only=True) 
    avg['Sample'] = 'Average'
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
  
    return df

if __name__ == "__main__":
    # 配置参数
    test_folder = "/106552404021/MLFF-GAN-main/MLFF-GAN-main/out/test"
    output_csv = "/106552404021/MLFF-GAN-main/MLFF-GAN-main/out/test/evaluation_results.csv"

    # 执行批量评估
    results_df = batch_evaluate(test_folder)
  
    # 打印控制台结果
    print("\n评估结果汇总:")
    print("测试文件夹：{}".format(test_folder))
    print(results_df.round(4).to_string(index=False))
  
    # 保存CS文件
    results_df.to_csv(output_csv, index=False)
    print(f"\n结果已保存至: {output_csv}")
    # 示例输出:
    """
    评估结果:
    RMSE: 0.0194
    PSNR: 35.4112
    SSIM: 0.8865
    Correlation: 0.9716
    SAM: 3.75°
    ERGAS: 126.3276
    """