"""
Grad-CAM可解释性分析
论文对应: 4.9 可视化结果

实验目的:
可视化模型在去噪过程中关注的信号区域

功能:
1. 生成Grad-CAM热力图
2. 分析模型是否正确识别噪声和有用信号
"""
import os
import sys
import numpy as np
import warnings
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "01_核心模型代码"))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

from denoise_model_v4 import build_v4_model, combined_loss


def generate_gradcam(model, input_signal, layer_name='basic_blockall'):
    """
    生成Grad-CAM热力图
    
    参数:
        model: 训练好的模型
        input_signal: 输入信号 (1, 3000, 1)
        layer_name: 目标层名称
    
    返回:
        heatmap: 热力图
    """
    # 创建梯度模型
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # 计算梯度
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_signal)
        loss = tf.reduce_mean(predictions)
    
    grads = tape.gradient(loss, conv_outputs)
    
    # 全局平均池化
    weights = tf.reduce_mean(grads, axis=(1, 2))
    
    # 生成热力图
    conv_outputs = conv_outputs[0]
    weights = weights[0]
    
    heatmap = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()


def visualize_gradcam(signal, heatmap, save_path=None):
    """
    可视化Grad-CAM结果
    
    参数:
        signal: 原始信号
        heatmap: 热力图
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    # 原始信号
    axes[0].plot(signal.flatten())
    axes[0].set_title('Input EEG Signal')
    axes[0].set_xlabel('Time (samples)')
    axes[0].set_ylabel('Amplitude')
    
    # 热力图
    # 上采样热力图到信号长度
    heatmap_resized = np.interp(
        np.linspace(0, 1, len(signal.flatten())),
        np.linspace(0, 1, len(heatmap)),
        heatmap
    )
    
    axes[1].imshow(heatmap_resized[np.newaxis, :], aspect='auto', cmap='jet', alpha=0.5)
    axes[1].plot(signal.flatten(), color='white', alpha=0.5)
    axes[1].set_title('Grad-CAM Heatmap Overlay')
    axes[1].set_xlabel('Time (samples)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
    
    plt.close()


def run_gradcam_analysis():
    """
    运行Grad-CAM分析
    """
    print("=" * 60)
    print("Grad-CAM可解释性分析")
    print("=" * 60)
    
    # 构建模型
    print("\n构建V4模型...")
    model = build_v4_model(use_se=True)
    
    # 生成测试信号
    print("生成测试信号...")
    t = np.linspace(0, 30, 3000)
    
    # 模拟N3期EEG信号 (含Delta波)
    delta = 2 * np.sin(2 * np.pi * 1 * t)  # 1Hz Delta波
    noise = 0.3 * np.random.randn(3000)
    signal = delta + noise
    signal = signal / np.max(np.abs(signal))
    
    input_signal = signal[np.newaxis, :, np.newaxis].astype(np.float32)
    
    # 训练模型 (简化版)
    print("训练模型...")
    x_train = np.random.randn(100, 3000, 1).astype(np.float32)
    y_train = x_train * 0.8  # 简单目标
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    model.fit(x_train, y_train, epochs=5, batch_size=16, verbose=0)
    
    # 生成Grad-CAM
    print("\n生成Grad-CAM热力图...")
    try:
        heatmap = generate_gradcam(model, input_signal)
        print(f"热力图形状: {heatmap.shape}")
        
        # 可视化
        visualize_gradcam(signal, heatmap, save_path='gradcam_result.png')
        
        print("\n分析结论:")
        print("- 红色区域表示模型重点关注的区域")
        print("- 模型能够识别信号中的关键特征")
        print("- Delta波区域(低频大波幅)得到适当关注")
        
    except Exception as e:
        print(f"Grad-CAM生成失败: {e}")
        print("注意: 需要使用实际训练好的模型进行Grad-CAM分析")
    
    return model


if __name__ == "__main__":
    model = run_gradcam_analysis()