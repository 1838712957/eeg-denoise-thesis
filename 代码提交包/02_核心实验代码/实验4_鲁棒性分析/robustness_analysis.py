"""
鲁棒性分析 - 不同信号质量下的去噪性能
论文对应: 4.5 不同信号质量与伪迹严重程度下的鲁棒性分析

实验目的:
评估模型在不同噪声水平下的稳定性

实验设计:
1. 干净信号组 (噪声 ≤ 0.2)
2. 严重伪迹组 (噪声 > 0.5)

评估指标:
- RRMSE (相对均方根误差)
- CC (相关系数)
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


def generate_data_with_noise_level(n_samples=100, signal_length=3000, noise_level=0.3):
    """生成指定噪声水平的测试数据"""
    t = np.linspace(0, 30, signal_length)
    
    clean_signals = []
    noisy_signals = []
    
    for _ in range(n_samples):
        # 模拟EEG信号
        delta = np.random.uniform(0.5, 4) * np.sin(2 * np.pi * np.random.uniform(0.5, 2) * t)
        theta = np.random.uniform(0.3, 0.8) * np.sin(2 * np.pi * np.random.uniform(4, 8) * t)
        alpha = np.random.uniform(0.2, 0.5) * np.sin(2 * np.pi * np.random.uniform(8, 13) * t)
        
        clean = delta + theta + alpha
        clean = clean / np.max(np.abs(clean))
        
        noise = noise_level * np.random.randn(signal_length)
        noisy = clean + noise
        
        clean_signals.append(clean)
        noisy_signals.append(noisy)
    
    return np.array(noisy_signals)[..., np.newaxis], np.array(clean_signals)[..., np.newaxis]


def run_robustness_analysis():
    """
    运行鲁棒性分析实验
    """
    print("=" * 60)
    print("鲁棒性分析 - 不同信号质量下的去噪性能")
    print("=" * 60)
    
    # 构建模型
    print("\n构建V4模型...")
    model = build_v4_model(use_se=True)
    
    # 使用中等噪声数据训练
    print("训练模型...")
    x_train, y_train = generate_data_with_noise_level(n_samples=200, noise_level=0.3)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=lambda y_true, y_pred: combined_loss(y_true, y_pred, lambda_delta=0.1)
    )
    
    model.fit(x_train, y_train, epochs=30, batch_size=16, verbose=0)
    
    # ========== 测试不同噪声水平 ==========
    print("\n" + "-" * 40)
    print("测试不同噪声水平...")
    print("-" * 40)
    
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.65]
    results = {'noise_level': [], 'RRMSE': [], 'CC': []}
    
    for noise in noise_levels:
        x_test, y_test = generate_data_with_noise_level(n_samples=50, noise_level=noise)
        y_pred = model.predict(x_test, verbose=0)
        
        rrmse = np.mean([calc_rrmse(y_test[i], y_pred[i]) for i in range(len(y_test))])
        cc = np.mean([calc_cc(y_test[i], y_pred[i]) for i in range(len(y_test))])
        
        results['noise_level'].append(noise)
        results['RRMSE'].append(rrmse * 100)
        results['CC'].append(cc)
        
        print(f"噪声水平 {noise:.2f}: RRMSE={rrmse*100:.2f}%, CC={cc:.4f}")
    
    # ========== 分组分析 ==========
    print("\n" + "=" * 60)
    print("分组分析结果")
    print("=" * 60)
    
    # 干净信号组 (噪声 ≤ 0.2)
    clean_idx = [i for i, n in enumerate(results['noise_level']) if n <= 0.2]
    clean_rrmse = np.mean([results['RRMSE'][i] for i in clean_idx])
    clean_cc = np.mean([results['CC'][i] for i in clean_idx])
    
    # 严重伪迹组 (噪声 > 0.5)
    severe_idx = [i for i, n in enumerate(results['noise_level']) if n > 0.5]
    severe_rrmse = np.mean([results['RRMSE'][i] for i in severe_idx])
    severe_cc = np.mean([results['CC'][i] for i in severe_idx])
    
    print(f"\n{'测试分组':<15} {'噪声分布':<15} {'RRMSE(%)':<12} {'CC':<10}")
    print("-" * 52)
    print(f"{'干净信号组':<15} {'Noise ≤ 0.2':<15} {clean_rrmse:<12.2f} {clean_cc:<10.4f}")
    print(f"{'严重伪迹组':<15} {'Noise > 0.5':<15} {severe_rrmse:<12.2f} {severe_cc:<10.4f}")
    
    # 计算CC下降率
    cc_drop_rate = (clean_cc - severe_cc) / clean_cc * 100
    print(f"\nCC下降率: {cc_drop_rate:.1f}%")
    
    if cc_drop_rate < 20:
        print("结论: 模型具有出色的抗干扰鲁棒性 (CC下降率 < 20%)")
    
    return results


if __name__ == "__main__":
    results = run_robustness_analysis()