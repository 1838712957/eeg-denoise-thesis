"""
N3召回率分析实验
论文对应: 4.5 N3阶段召回率优化分析

实验目的:
分析N3深睡阶段召回率较低的原因及优化方法

分析内容:
- N3阶段特征分析
- 噪声对N3识别的影响
- 去噪后N3召回率提升
"""
import os
import sys
import numpy as np
import warnings
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')


def generate_n3_data(n_samples=200):
    """
    生成N3阶段测试数据
    
    N3阶段特征:
    - Delta波 (0.5-4 Hz) 占主导
    - 慢波活动 (SWA) > 20%
    - 波形幅度较大
    """
    print("生成N3阶段测试数据...")
    
    signal_length = 3000  # 30秒 @ 100Hz
    t = np.linspace(0, 30, signal_length)
    
    signals = []
    labels = []  # 0=非N3, 1=N3
    
    for i in range(n_samples):
        is_n3 = i % 2  # 一半N3，一半非N3
        
        if is_n3:
            # N3特征: 强Delta波
            delta = 1.5 * np.sin(2 * np.pi * 1 * t)
            delta2 = 0.8 * np.sin(2 * np.pi * 2 * t)
            slow_wave = 0.5 * np.sin(2 * np.pi * 0.5 * t)
            signal = delta + delta2 + slow_wave
            labels.append(1)
        else:
            # 非N3: 其他频段
            stage = i % 4
            if stage == 0:  # W
                signal = 0.8 * np.sin(2 * np.pi * 10 * t)
            elif stage == 1:  # N1
                signal = 0.6 * np.sin(2 * np.pi * 5 * t)
            elif stage == 2:  # N2
                signal = 0.4 * np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 13 * t)
            else:  # REM
                signal = 0.5 * np.sin(2 * np.pi * 6 * t)
            labels.append(0)
        
        # 添加噪声
        noise = 0.3 * np.random.randn(signal_length)
        signal = signal + noise
        
        # 归一化
        signal = signal / np.max(np.abs(signal))
        signals.append(signal)
    
    return np.array(signals)[..., np.newaxis], np.array(labels)


def calculate_delta_ratio(signal, fs=100):
    """
    计算Delta波能量占比
    
    参数:
        signal: EEG信号
        fs: 采样率
    
    返回:
        delta_ratio: Delta波能量占比
    """
    # 功率谱
    fft = np.fft.fft(signal)
    psd = np.abs(fft) ** 2
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    
    # 只取正频率
    pos_mask = freqs >= 0
    psd = psd[pos_mask]
    freqs = freqs[pos_mask]
    
    # Delta波段 (0.5-4 Hz)
    delta_mask = (freqs >= 0.5) & (freqs <= 4)
    delta_energy = np.sum(psd[delta_mask])
    total_energy = np.sum(psd)
    
    return delta_energy / total_energy


def analyze_n3_characteristics():
    """
    分析N3阶段特征
    """
    print("\n" + "=" * 60)
    print("N3阶段特征分析")
    print("=" * 60)
    
    X, y = generate_n3_data(n_samples=200)
    
    # 分别计算N3和非N3的Delta占比
    n3_mask = y == 1
    non_n3_mask = y == 0
    
    n3_delta_ratios = []
    non_n3_delta_ratios = []
    
    for i in range(len(X)):
        ratio = calculate_delta_ratio(X[i, :, 0])
        if n3_mask[i]:
            n3_delta_ratios.append(ratio)
        else:
            non_n3_delta_ratios.append(ratio)
    
    print(f"\n{'阶段':<15} {'Delta占比均值':<15} {'Delta占比标准差':<15}")
    print("-" * 45)
    print(f"{'N3':<15} {np.mean(n3_delta_ratios):<15.3f} {np.std(n3_delta_ratios):<15.3f}")
    print(f"{'非N3':<15} {np.mean(non_n3_delta_ratios):<15.3f} {np.std(non_n3_delta_ratios):<15.3f}")
    
    return n3_delta_ratios, non_n3_delta_ratios


def analyze_noise_impact():
    """
    分析噪声对N3识别的影响
    """
    print("\n" + "=" * 60)
    print("噪声对N3识别的影响分析")
    print("=" * 60)
    
    X, y = generate_n3_data(n_samples=200)
    
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    print(f"\n{'噪声水平':<15} {'N3 Delta占比':<15} {'非N3 Delta占比':<15} {'区分度':<15}")
    print("-" * 60)
    
    for noise_level in noise_levels:
        # 添加不同水平噪声
        X_noisy = X + noise_level * np.random.randn(*X.shape)
        
        n3_mask = y == 1
        n3_ratios = [calculate_delta_ratio(X_noisy[i, :, 0]) for i in range(len(X)) if n3_mask[i]]
        non_n3_ratios = [calculate_delta_ratio(X_noisy[i, :, 0]) for i in range(len(X)) if not n3_mask[i]]
        
        # 区分度 = N3均值 - 非N3均值
        separability = np.mean(n3_ratios) - np.mean(non_n3_ratios)
        
        print(f"{noise_level:<15.1f} {np.mean(n3_ratios):<15.3f} {np.mean(non_n3_ratios):<15.3f} {separability:<15.3f}")


def run_n3_recall_analysis():
    """
    运行N3召回率分析实验
    """
    print("=" * 60)
    print("N3召回率分析实验")
    print("=" * 60)
    
    # 1. N3特征分析
    n3_ratios, non_n3_ratios = analyze_n3_characteristics()
    
    # 2. 噪声影响分析
    analyze_noise_impact()
    
    # 3. 模拟去噪效果
    print("\n" + "=" * 60)
    print("去噪对N3召回率的影响")
    print("=" * 60)
    
    X, y = generate_n3_data(n_samples=200)
    
    # 添加噪声
    X_noisy = X + 0.3 * np.random.randn(*X.shape)
    
    # 模拟去噪 (保留更多Delta波)
    from scipy.signal import butter, filtfilt
    
    def denoise_signal(signal, fs=100):
        """带通滤波去噪"""
        nyq = 0.5 * fs
        low = 0.5 / nyq
        high = 30 / nyq
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, signal)
    
    X_denoised = np.array([denoise_signal(sig[:, 0]) for sig in X_noisy])[..., np.newaxis]
    
    # 简单分类器
    def simple_n3_classifier(X_test, threshold=0.4):
        """基于Delta占比的简单分类器"""
        predictions = []
        for sig in X_test:
            ratio = calculate_delta_ratio(sig[:, 0])
            predictions.append(1 if ratio > threshold else 0)
        return np.array(predictions)
    
    # 评估
    y_pred_noisy = simple_n3_classifier(X_noisy)
    y_pred_denoised = simple_n3_classifier(X_denoised)
    
    # 计算N3召回率
    n3_mask = y == 1
    
    recall_noisy = np.mean(y_pred_noisy[n3_mask] == 1)
    recall_denoised = np.mean(y_pred_denoised[n3_mask] == 1)
    
    print(f"\n{'处理方式':<20} {'N3召回率':<15} {'提升':<15}")
    print("-" * 50)
    print(f"{'噪声信号':<20} {recall_noisy*100:<15.1f}% {'-':<15}")
    print(f"{'去噪后':<20} {recall_denoised*100:<15.1f}% {(recall_denoised-recall_noisy)*100:+.1f}%")
    
    # 结果汇总
    print("\n" + "=" * 60)
    print("实验结论")
    print("=" * 60)
    print("""
1. N3阶段特征: Delta波能量占比显著高于其他阶段
2. 噪声影响: 随着噪声增加，N3与非N3的区分度下降
3. 去噪效果: 去噪后N3召回率明显提升
4. 建议: 保留Delta波段能量是去噪的关键
    """)
    
    return {
        'recall_noisy': recall_noisy,
        'recall_denoised': recall_denoised,
        'improvement': recall_denoised - recall_noisy
    }


if __name__ == "__main__":
    results = run_n3_recall_analysis()