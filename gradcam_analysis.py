"""
1D Grad-CAM 可解释性分析
用于可视化深度学习去噪模型关注的EEG信号区域
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).resolve().parent

# ==================== 模型定义 ====================
class SEBlock(layers.Layer):
    def __init__(self, channels=32, reduction=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.channels, self.reduction = channels, reduction
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        self.fc1 = layers.Dense(max(channels // reduction, 4), activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')
        self.reshape = layers.Reshape((1, channels))
    def call(self, inputs):
        x = self.global_avg_pool(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.reshape(x)
        return inputs * x
    def get_config(self):
        return {"channels": self.channels, "reduction": self.reduction}

class Res_BasicBlock(layers.Layer):
    def __init__(self, kernelsize, stride=1, use_se=False, **kwargs):
        super(Res_BasicBlock, self).__init__(**kwargs)
        self.bblock = Sequential([
            layers.Conv1D(32, kernelsize, strides=stride, padding="same"),
            layers.BatchNormalization(), layers.ReLU(),
            layers.Conv1D(16, kernelsize, strides=1, padding="same"),
            layers.BatchNormalization(), layers.ReLU(),
            layers.Conv1D(32, kernelsize, strides=1, padding="same"),
            layers.BatchNormalization(), layers.ReLU()
        ])
        if use_se: self.se = SEBlock(32)
    def call(self, inputs):
        out = self.bblock(inputs)
        if hasattr(self, 'se'): out = self.se(out)
        return layers.add([out, inputs])

class BasicBlockall(layers.Layer):
    def __init__(self, use_se=False, **kwargs):
        super(BasicBlockall, self).__init__(**kwargs)
        self.bblock3 = Sequential([Res_BasicBlock(3, use_se=use_se), Res_BasicBlock(3, use_se=use_se)])
        self.bblock5 = Sequential([Res_BasicBlock(5, use_se=use_se), Res_BasicBlock(5, use_se=use_se)])
        self.bblock7 = Sequential([Res_BasicBlock(7, use_se=use_se), Res_BasicBlock(7, use_se=use_se)])
    def call(self, inputs):
        return tf.concat([self.bblock3(inputs), self.bblock5(inputs), self.bblock7(inputs)], axis=-1)

def build_denoise_model():
    inp = layers.Input(shape=(512, 1))
    x = layers.Conv1D(32, 7, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = BasicBlockall(use_se=True)(x)
    out = layers.Conv1D(1, 7, padding='same')(x)
    return models.Model(inp, out)

# ==================== Grad-CAM 类 ====================
class GradCAM1D:
    def __init__(self, model):
        self.model = model
        # 找到最后一个卷积层
        self.target_layer = None
        for layer in reversed(model.layers):
            if 'conv1d' in layer.name.lower() or 'basic' in layer.name.lower():
                self.target_layer = layer
                break
    
    def compute_heatmap(self, input_signal):
        """计算Grad-CAM热力图"""
        input_tensor = tf.convert_to_tensor(input_signal, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            output = self.model(input_tensor)
            loss = tf.reduce_mean(output)
        
        grads = tape.gradient(loss, input_tensor)
        if grads is None:
            return np.ones(512) / 512
        
        # 计算每个时间点的重要性
        saliency = tf.abs(grads[0, :, 0]).numpy()
        saliency = saliency / (np.max(saliency) + 1e-8)
        return saliency
    
    def compute_integrated_gradients(self, input_signal, steps=30):
        """计算积分梯度"""
        input_tensor = tf.convert_to_tensor(input_signal, dtype=tf.float32)
        baseline = tf.zeros_like(input_tensor)
        
        interpolated = []
        for i in range(steps + 1):
            alpha = i / steps
            interpolated.append(baseline + alpha * (input_tensor - baseline))
        
        interpolated = tf.concat(interpolated, axis=0)
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            outputs = self.model(interpolated)
            loss = tf.reduce_mean(outputs, axis=-1)
        
        grads = tape.gradient(loss, interpolated)
        if grads is None:
            return np.ones(512) / 512
        
        avg_grads = tf.reduce_mean(grads, axis=0)
        attributions = (input_tensor - baseline) * avg_grads
        attributions = tf.abs(attributions[0, :, 0]).numpy()
        attributions = attributions / (np.max(attributions) + 1e-8)
        
        return attributions

# ==================== 主分析函数 ====================
def run_gradcam_analysis():
    """运行Grad-CAM分析"""
    print("=" * 60)
    print("1D Grad-CAM 可解释性分析")
    print("=" * 60)
    
    # 1. 加载模型
    print("\n[1/5] 加载去噪模型...")
    model = build_denoise_model()
    model_path = PROJECT_ROOT / "03_训练模型" / "去噪模型v2最终版.h5"
    model.load_weights(str(model_path))
    print("模型加载成功!")
    
    # 2. 初始化Grad-CAM
    print("\n[2/5] 初始化Grad-CAM...")
    gradcam = GradCAM1D(model)
    print("Grad-CAM初始化成功!")
    
    # 3. 加载测试数据
    print("\n[3/5] 加载测试数据...")
    import mne
    raw_dir = PROJECT_ROOT / "04_原始数据" / "Raw_edf"
    raw_files = sorted(list(raw_dir.glob("*PSG.edf")))[:3]  # 只处理前3个
    
    output_dir = PROJECT_ROOT / "06_实验结果" / "GradCAM分析"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. 分析每个受试者
    print("\n[4/5] 运行Grad-CAM分析...")
    results = []
    
    for raw_file in raw_files:
        subject_id = raw_file.stem.split('-')[0]
        print(f"\n处理 {subject_id}...")
        
        try:
            # 读取EEG数据
            raw = mne.io.read_raw_edf(str(raw_file), preload=True, verbose=False)
            if raw.info['sfreq'] != 256:
                raw.resample(256)
            
            # 获取EEG通道
            eeg_picks = mne.pick_channels(raw.info['ch_names'], include=['EEG Fpz-Cz'])
            if len(eeg_picks) == 0:
                eeg_picks = [0]
            
            data = raw.get_data(picks=eeg_picks)[0]
            
            # 取一段信号进行分析
            segment_len = 512
            n_segments = min(10, len(data) // segment_len)
            
            all_heatmaps = []
            all_integrated = []
            
            for i in range(n_segments):
                segment = data[i*segment_len:(i+1)*segment_len]
                std = np.std(segment) if np.std(segment) > 0 else 1.0
                segment_norm = segment / std
                
                input_tensor = segment_norm.reshape(1, segment_len, 1)
                
                # 计算Grad-CAM
                heatmap = gradcam.compute_heatmap(input_tensor)
                all_heatmaps.append(heatmap)
                
                # 计算积分梯度
                integrated = gradcam.compute_integrated_gradients(input_tensor)
                all_integrated.append(integrated)
            
            # 平均热力图
            avg_heatmap = np.mean(all_heatmaps, axis=0)
            avg_integrated = np.mean(all_integrated, axis=0)
            
            # 去噪
            test_input = data[:segment_len].reshape(1, segment_len, 1)
            std_test = np.std(test_input) if np.std(test_input) > 0 else 1.0
            test_input_norm = test_input / std_test
            denoised = model.predict(test_input_norm, verbose=0).flatten() * std_test
            
            # 可视化
            visualize_results(
                signal_raw=data[:segment_len],
                signal_denoised=denoised,
                heatmap=avg_heatmap,
                integrated_grad=avg_integrated,
                subject_id=subject_id,
                output_dir=output_dir
            )
            
            results.append({
                'subject': subject_id,
                'heatmap_mean': np.mean(avg_heatmap),
                'heatmap_std': np.std(avg_heatmap),
                'integrated_mean': np.mean(avg_integrated)
            })
            
        except Exception as e:
            print(f"处理 {subject_id} 失败: {e}")
    
    # 5. 保存结果
    print("\n[5/5] 保存分析结果...")
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "GradCAM_Results.csv", index=False)
    
    print(f"\n分析完成! 结果保存在: {output_dir}")
    return results

def visualize_results(signal_raw, signal_denoised, heatmap, integrated_grad, subject_id, output_dir):
    """可视化分析结果"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    time_axis = np.arange(len(signal_raw)) / 256
    
    # 1. 原始信号
    axes[0].plot(time_axis, signal_raw, 'b-', linewidth=0.8, alpha=0.8)
    axes[0].set_title(f'{subject_id} - 原始EEG信号', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('幅度 (μV)')
    axes[0].grid(True, alpha=0.3)
    
    # 2. 去噪后信号
    axes[1].plot(time_axis, signal_denoised, 'g-', linewidth=0.8, alpha=0.8)
    axes[1].set_title('去噪后EEG信号', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('幅度 (μV)')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Grad-CAM热力图
    axes[2].fill_between(time_axis, 0, heatmap, alpha=0.6, color='red')
    axes[2].plot(time_axis, heatmap, 'r-', linewidth=1)
    axes[2].set_title('Grad-CAM 显著图（模型关注区域）', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('重要性')
    axes[2].grid(True, alpha=0.3)
    
    # 4. 积分梯度
    axes[3].fill_between(time_axis, 0, integrated_grad, alpha=0.6, color='purple')
    axes[3].plot(time_axis, integrated_grad, 'purple', linewidth=1)
    axes[3].set_title('积分梯度（特征归因）', fontsize=12, fontweight='bold')
    axes[3].set_ylabel('归因值')
    axes[3].set_xlabel('时间 (s)')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f'{subject_id}_GradCAM.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_path}")

if __name__ == "__main__":
    run_gradcam_analysis()