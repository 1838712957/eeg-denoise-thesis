"""
数据量对模型性能的影响实验
论文对应: 4.7 数据量对模型性能的影响

实验目的:
验证训练数据量对去噪性能的影响

实验设计:
训练数据比例: 10%, 30%, 50%, 100%

评估指标:
- 验证损失
- RRMSE
- CC
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


def generate_training_data(n_samples=2000, signal_length=3000, noise_level=0.3):
    """生成训练数据"""
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


def run_data_scaling_experiment():
    """
    运行数据量缩放实验
    """
    print("=" * 60)
    print("数据量对模型性能的影响实验")
    print("=" * 60)
    
    # 生成全部数据
    print("\n生成训练数据...")
    x_all, y_all = generate_training_data(n_samples=2000)
    
    # 固定测试集
    test_size = 200
    x_test, y_test = x_all[-test_size:], y_all[-test_size:]
    
    # 训练数据比例
    ratios = [0.1, 0.3, 0.5, 1.0]
    train_sizes = [180, 540, 900, 1800]
    
    results = []
    
    for ratio, n_train in zip(ratios, train_sizes):
        print(f"\n{'='*40}")
        print(f"训练数据比例: {ratio*100:.0f}% ({n_train} 样本)")
        print("=" * 40)
        
        # 准备数据
        x_train = x_all[:n_train]
        y_train = y_all[:n_train]
        
        # 构建模型
        model = build_v4_model(use_se=True)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=lambda y_true, y_pred: combined_loss(y_true, y_pred, lambda_delta=0.1)
        )
        
        # 训练
        history = model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=50,
            batch_size=32,
            verbose=0
        )
        
        # 评估
        val_loss = min(history.history['val_loss'])
        y_pred = model.predict(x_test, verbose=0)
        
        rrmse = np.mean([calc_rrmse(y_test[i], y_pred[i]) for i in range(len(y_test))])
        cc = np.mean([calc_cc(y_test[i], y_pred[i]) for i in range(len(y_test))])
        
        results.append({
            'ratio': ratio,
            'n_train': n_train,
            'val_loss': val_loss,
            'RRMSE': rrmse * 100,
            'CC': cc
        })
        
        print(f"验证损失: {val_loss:.6f}")
        print(f"RRMSE: {rrmse*100:.2f}%")
        print(f"CC: {cc:.4f}")
    
    # ========== 结果汇总 ==========
    print("\n" + "=" * 60)
    print("数据量实验结果汇总")
    print("=" * 60)
    print(f"\n{'数据比例':<10} {'训练样本':<10} {'验证损失':<12} {'RRMSE(%)':<10} {'CC':<10}")
    print("-" * 52)
    for r in results:
        print(f"{r['ratio']*100:.0f}%{'':<7} {r['n_train']:<10} {r['val_loss']:<12.6f} {r['RRMSE']:<10.2f} {r['CC']:<10.4f}")
    
    # 分析趋势
    print("\n" + "-" * 40)
    print("分析结论:")
    print("-" * 40)
    rrmse_improve = (results[0]['RRMSE'] - results[-1]['RRMSE']) / results[0]['RRMSE'] * 100
    print(f"数据量从10%增加到100%时:")
    print(f"  - RRMSE降低: {rrmse_improve:.1f}%")
    print(f"  - CC提升: {(results[-1]['CC'] - results[0]['CC']):.4f}")
    
    return results


if __name__ == "__main__":
    results = run_data_scaling_experiment()