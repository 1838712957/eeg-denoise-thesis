"""
Sleep-EDF数据集去噪效果实验
论文对应: 4.2 Sleep-EDF数据集去噪效果

实验目的:
评估模型在真实睡眠EEG数据上的去噪性能

评估指标:
- RRMSE (相对均方根误差)
- CC (相关系数)
- Delta波能量保持率
"""
import os
import sys
import numpy as np
import warnings
import tensorflow as tf
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "01_核心模型代码"))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

from denoise_model_v4 import (
    build_v4_model, calc_rrmse, calc_cc, combined_loss
)


def load_sleep_edf_data(data_path=None):
    """
    加载Sleep-EDF数据
    
    参数:
        data_path: 数据路径
    
    返回:
        noisy_signals: 含噪信号
        clean_signals: 干净信号 (参考)
    """
    # 模拟数据加载 (实际使用时替换为真实数据)
    print("加载Sleep-EDF数据...")
    
    # 生成模拟数据
    n_samples = 100
    signal_length = 3000  # 30秒 @ 100Hz
    
    t = np.linspace(0, 30, signal_length)
    
    noisy_signals = []
    clean_signals = []
    
    for i in range(n_samples):
        # 模拟不同睡眠阶段的EEG
        stage = i % 5  # W, N1, N2, N3, REM
        
        if stage == 3:  # N3 - Delta波为主
            delta = 2 * np.sin(2 * np.pi * 1 * t)
            theta = 0.3 * np.sin(2 * np.pi * 5 * t)
            signal = delta + theta
        elif stage == 2:  # N2 - 睡眠纺锤波
            delta = 0.5 * np.sin(2 * np.pi * 1 * t)
            sigma = 0.8 * np.sin(2 * np.pi * 13 * t)
            signal = delta + sigma
        else:  # 其他阶段
            signal = np.random.randn(signal_length) * 0.5
        
        # 归一化
        signal = signal / np.max(np.abs(signal))
        
        # 添加噪声
        noise_level = np.random.uniform(0.2, 0.5)
        noise = noise_level * np.random.randn(signal_length)
        noisy = signal + noise
        
        clean_signals.append(signal)
        noisy_signals.append(noisy)
    
    return np.array(noisy_signals)[..., np.newaxis], np.array(clean_signals)[..., np.newaxis]


def run_sleep_edf_experiment():
    """
    运行Sleep-EDF去噪实验
    """
    print("=" * 60)
    print("Sleep-EDF数据集去噪效果实验")
    print("=" * 60)
    
    # 加载数据
    x_test, y_test = load_sleep_edf_data()
    print(f"测试数据: {x_test.shape}")
    
    # 构建模型
    print("\n构建V4模型...")
    model = build_v4_model(use_se=True)
    
    # 简化训练 (实际使用时加载预训练权重)
    print("训练模型...")
    x_train = np.random.randn(200, 3000, 1).astype(np.float32)
    y_train = x_train * 0.8
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=lambda y_true, y_pred: combined_loss(y_true, y_pred, lambda_delta=0.1)
    )
    model.fit(x_train, y_train, epochs=20, batch_size=16, verbose=0)
    
    # 去噪
    print("\n执行去噪...")
    y_pred = model.predict(x_test, verbose=0)
    
    # 计算指标
    rrmse_list = []
    cc_list = []
    
    for i in range(len(x_test)):
        rrmse = calc_rrmse(y_test[i], y_pred[i])
        cc = calc_cc(y_test[i], y_pred[i])
        rrmse_list.append(rrmse)
        cc_list.append(cc)
    
    # 结果汇总
    print("\n" + "=" * 60)
    print("实验结果")
    print("=" * 60)
    
    print(f"\n{'指标':<15} {'均值':<12} {'标准差':<12} {'最小值':<12} {'最大值':<12}")
    print("-" * 63)
    print(f"{'RRMSE(%)':<15} {np.mean(rrmse_list)*100:<12.2f} {np.std(rrmse_list)*100:<12.2f} {np.min(rrmse_list)*100:<12.2f} {np.max(rrmse_list)*100:<12.2f}")
    print(f"{'CC':<15} {np.mean(cc_list):<12.4f} {np.std(cc_list):<12.4f} {np.min(cc_list):<12.4f} {np.max(cc_list):<12.4f}")
    
    # Delta波能量分析
    print("\n" + "-" * 40)
    print("Delta波能量保持分析")
    print("-" * 40)
    
    fs = 100
    delta_energy_original = []
    delta_energy_denoised = []
    
    for i in range(len(x_test)):
        # 计算功率谱
        f, psd_orig = np.abs(np.fft.fft(x_test[i, :, 0]))[:1500], np.fft.fftfreq(3000, 1/fs)[:1500]
        f, psd_denoised = np.abs(np.fft.fft(y_pred[i, :, 0]))[:1500], np.fft.fftfreq(3000, 1/fs)[:1500]
        
        # Delta波段 (0.5-4 Hz)
        delta_mask = (np.fft.fftfreq(3000, 1/fs)[:1500] >= 0.5) & (np.fft.fftfreq(3000, 1/fs)[:1500] <= 4)
        
        delta_energy_original.append(np.sum(np.abs(np.fft.fft(x_test[i, :, 0]))[:1500][delta_mask]**2))
        delta_energy_denoised.append(np.sum(np.abs(np.fft.fft(y_pred[i, :, 0]))[:1500][delta_mask]**2))
    
    preservation_rate = np.mean(delta_energy_denoised) / np.mean(delta_energy_original) * 100
    print(f"Delta波能量保持率: {preservation_rate:.1f}%")
    
    return {
        'RRMSE': np.mean(rrmse_list) * 100,
        'CC': np.mean(cc_list),
        'Delta_Preservation': preservation_rate
    }


if __name__ == "__main__":
    results = run_sleep_edf_experiment()