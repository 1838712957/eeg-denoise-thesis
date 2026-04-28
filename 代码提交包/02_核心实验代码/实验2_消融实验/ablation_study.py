"""
消融实验 - 核心模块有效性验证
论文对应: 4.3 核心模块消融实验分析

实验目的:
验证多尺度残差结构、SE注意力机制、Delta波保护损失函数的有效性

实验设计:
1. Baseline: 基础1D-CNN网络
2. V4_Complete: 完整网络模型(多尺度残差+SE注意力+Delta保护)

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
    build_v4_model, build_baseline_model, 
    calc_rrmse, calc_cc, calc_delta_energy_preservation,
    combined_loss
)


def generate_test_data(n_samples=100, signal_length=3000, noise_level=0.3):
    """
    生成测试数据
    
    参数:
        n_samples: 样本数量
        signal_length: 信号长度
        noise_level: 噪声水平
    """
    # 生成干净的合成EEG信号
    t = np.linspace(0, 30, signal_length)
    
    clean_signals = []
    noisy_signals = []
    
    for _ in range(n_samples):
        # 模拟EEG信号: 多频率成分混合
        delta = np.random.uniform(0.5, 4) * np.sin(2 * np.pi * np.random.uniform(0.5, 2) * t)
        theta = np.random.uniform(0.3, 0.8) * np.sin(2 * np.pi * np.random.uniform(4, 8) * t)
        alpha = np.random.uniform(0.2, 0.5) * np.sin(2 * np.pi * np.random.uniform(8, 13) * t)
        beta = np.random.uniform(0.1, 0.3) * np.sin(2 * np.pi * np.random.uniform(13, 30) * t)
        
        clean = delta + theta + alpha + beta
        clean = clean / np.max(np.abs(clean))  # 归一化
        
        # 添加噪声
        noise = noise_level * np.random.randn(signal_length)
        noisy = clean + noise
        
        clean_signals.append(clean)
        noisy_signals.append(noisy)
    
    return np.array(noisy_signals)[..., np.newaxis], np.array(clean_signals)[..., np.newaxis]


def run_ablation_experiment():
    """
    运行消融实验
    """
    print("=" * 60)
    print("消融实验 - 核心模块有效性验证")
    print("=" * 60)
    
    # 生成测试数据
    print("\n生成测试数据...")
    x_test, y_test = generate_test_data(n_samples=200, noise_level=0.3)
    
    # 分割训练集和测试集
    split = 150
    x_train, x_val = x_test[:split], x_test[split:]
    y_train, y_val = y_test[:split], y_test[split:]
    
    results = {}
    
    # ========== Baseline模型 ==========
    print("\n" + "-" * 40)
    print("训练 Baseline 模型...")
    print("-" * 40)
    
    baseline_model = build_baseline_model()
    baseline_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    
    baseline_model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=50,
        batch_size=16,
        verbose=0
    )
    
    # 评估
    y_pred_baseline = baseline_model.predict(x_val, verbose=0)
    
    rrmse_baseline = np.mean([calc_rrmse(y_val[i], y_pred_baseline[i]) for i in range(len(y_val))])
    cc_baseline = np.mean([calc_cc(y_val[i], y_pred_baseline[i]) for i in range(len(y_val))])
    delta_baseline = np.mean([calc_delta_energy_preservation(y_val[i], y_pred_baseline[i]) for i in range(len(y_val))])
    
    results['Baseline'] = {
        'RRMSE': rrmse_baseline * 100,
        'CC': cc_baseline,
        'Delta_Preservation': delta_baseline
    }
    
    print(f"Baseline - RRMSE: {rrmse_baseline*100:.2f}%, CC: {cc_baseline:.4f}, Delta保持率: {delta_baseline:.2f}%")
    
    # ========== V4_Complete模型 ==========
    print("\n" + "-" * 40)
    print("训练 V4_Complete 模型...")
    print("-" * 40)
    
    v4_model = build_v4_model(use_se=True)
    v4_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=lambda y_true, y_pred: combined_loss(y_true, y_pred, lambda_delta=0.1)
    )
    
    v4_model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=50,
        batch_size=16,
        verbose=0
    )
    
    # 评估
    y_pred_v4 = v4_model.predict(x_val, verbose=0)
    
    rrmse_v4 = np.mean([calc_rrmse(y_val[i], y_pred_v4[i]) for i in range(len(y_val))])
    cc_v4 = np.mean([calc_cc(y_val[i], y_pred_v4[i]) for i in range(len(y_val))])
    delta_v4 = np.mean([calc_delta_energy_preservation(y_val[i], y_pred_v4[i]) for i in range(len(y_val))])
    
    results['V4_Complete'] = {
        'RRMSE': rrmse_v4 * 100,
        'CC': cc_v4,
        'Delta_Preservation': delta_v4
    }
    
    print(f"V4_Complete - RRMSE: {rrmse_v4*100:.2f}%, CC: {cc_v4:.4f}, Delta保持率: {delta_v4:.2f}%")
    
    # ========== 结果汇总 ==========
    print("\n" + "=" * 60)
    print("消融实验结果汇总")
    print("=" * 60)
    print(f"\n{'模型':<15} {'RRMSE(%)':<12} {'CC':<10} {'Delta保持率(%)':<15}")
    print("-" * 52)
    print(f"{'Baseline':<15} {results['Baseline']['RRMSE']:<12.2f} {results['Baseline']['CC']:<10.4f} {results['Baseline']['Delta_Preservation']:<15.2f}")
    print(f"{'V4_Complete':<15} {results['V4_Complete']['RRMSE']:<12.2f} {results['V4_Complete']['CC']:<10.4f} {results['V4_Complete']['Delta_Preservation']:<15.2f}")
    
    # 计算提升
    print("\n" + "-" * 40)
    print("V4_Complete 相比 Baseline 的提升:")
    print("-" * 40)
    rrmse_improve = (results['Baseline']['RRMSE'] - results['V4_Complete']['RRMSE']) / results['Baseline']['RRMSE'] * 100
    cc_improve = (results['V4_Complete']['CC'] - results['Baseline']['CC']) / results['Baseline']['CC'] * 100
    delta_improve = results['V4_Complete']['Delta_Preservation'] - results['Baseline']['Delta_Preservation']
    
    print(f"RRMSE降低: {rrmse_improve:.2f}%")
    print(f"CC提升: {cc_improve:.2f}%")
    print(f"Delta保持率提升: {delta_improve:.2f}个百分点")
    
    return results


if __name__ == "__main__":
    results = run_ablation_experiment()