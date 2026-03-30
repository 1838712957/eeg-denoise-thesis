"""
EEG去噪算法效果测试脚本
测试模型在不同受试者上的去噪效果
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import mne
import warnings
import sys
from tensorflow.keras import layers, models, Sequential
from pathlib import Path
from scipy.signal import welch

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ================= 路径配置 =================
PROJECT_ROOT = Path(__file__).resolve().parent
raw_dir = PROJECT_ROOT / "04_原始数据" / "Raw_edf"
model_path = PROJECT_ROOT / "03_训练模型" / "去噪模型v2最终版.h5"
output_dir = PROJECT_ROOT / "测试结果"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

TARGET_SFREQ = 256
SEGMENT_LEN = 512

# ================= 网络结构定义 =================
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
        config = super(SEBlock, self).get_config()
        config.update({"channels": self.channels, "reduction": self.reduction})
        return config

class Res_BasicBlock(layers.Layer):
    def __init__(self, kernelsize, stride=1, use_se=False, se_reduction=16, **kwargs):
        super(Res_BasicBlock, self).__init__(**kwargs)
        self.kernelsize, self.stride, self.use_se, self.se_reduction = kernelsize, stride, use_se, se_reduction
        self.bblock = Sequential([
            layers.Conv1D(32, kernelsize, strides=stride, padding="same"),
            layers.BatchNormalization(), layers.ReLU(),
            layers.Conv1D(16, kernelsize, strides=1, padding="same"),
            layers.BatchNormalization(), layers.ReLU(),
            layers.Conv1D(32, kernelsize, strides=1, padding="same"),
            layers.BatchNormalization(), layers.ReLU()
        ])
        if use_se: self.se = SEBlock(32, se_reduction)
    
    def call(self, inputs):
        out = self.bblock(inputs)
        if self.use_se: out = self.se(out)
        return layers.add([out, inputs])
    
    def get_config(self):
        config = super(Res_BasicBlock, self).get_config()
        config.update({"kernelsize": self.kernelsize, "stride": self.stride, 
                       "use_se": self.use_se, "se_reduction": self.se_reduction})
        return config

class BasicBlockall(layers.Layer):
    def __init__(self, stride=1, use_se=False, se_reduction=16, **kwargs):
        super(BasicBlockall, self).__init__(**kwargs)
        self.stride, self.use_se, self.se_reduction = stride, use_se, se_reduction
        self.bblock3 = Sequential([Res_BasicBlock(3, use_se=use_se), Res_BasicBlock(3, use_se=use_se)])
        self.bblock5 = Sequential([Res_BasicBlock(5, use_se=use_se), Res_BasicBlock(5, use_se=use_se)])
        self.bblock7 = Sequential([Res_BasicBlock(7, use_se=use_se), Res_BasicBlock(7, use_se=use_se)])
    
    def call(self, inputs):
        return tf.concat([self.bblock3(inputs), self.bblock5(inputs), self.bblock7(inputs)], axis=-1)
    
    def get_config(self):
        config = super(BasicBlockall, self).get_config()
        config.update({"stride": self.stride, "use_se": self.use_se, "se_reduction": self.se_reduction})
        return config

def build_model():
    """构建去噪模型"""
    inp = layers.Input(shape=(512, 1))
    x = layers.Conv1D(32, 7, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = BasicBlockall(use_se=True)(x)
    out = layers.Conv1D(1, 7, padding='same')(x)
    return models.Model(inp, out)

def calculate_metrics(raw, cleaned):
    """计算去噪指标"""
    if np.std(raw) == 0 or np.std(cleaned) == 0:
        return 0, 0, 0
    
    # 噪声减少率 (NRR)
    var_raw, var_clean = np.var(raw), np.var(cleaned)
    nrr = (var_raw - var_clean) / var_raw * 100
    
    # 相关系数 (CC)
    cc = np.corrcoef(raw, cleaned)[0, 1]
    
    # 信噪比改善 (SNR Improvement)
    snr_raw = 10 * np.log10(np.mean(raw**2) / np.var(raw) + 1e-10)
    snr_clean = 10 * np.log10(np.mean(cleaned**2) / np.var(cleaned) + 1e-10)
    snr_imp = snr_clean - snr_raw
    
    return nrr, cc, snr_imp

def plot_comparison(raw, cleaned, subject_id, seg_idx, save_path):
    """绘制时域对比图"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    time_axis = np.arange(len(raw)) / TARGET_SFREQ
    
    # 原始信号
    axes[0].plot(time_axis, raw * 1e6, color='gray', linewidth=0.8)
    axes[0].set_title(f'原始EEG信号 - {subject_id} 片段{seg_idx}', fontsize=12)
    axes[0].set_ylabel('幅度 (μV)')
    axes[0].grid(True, alpha=0.3)
    
    # 去噪后信号
    axes[1].plot(time_axis, cleaned * 1e6, color='blue', linewidth=0.8)
    axes[1].set_title('去噪后EEG信号', fontsize=12)
    axes[1].set_ylabel('幅度 (μV)')
    axes[1].grid(True, alpha=0.3)
    
    # 叠加对比
    axes[2].plot(time_axis, raw * 1e6, color='gray', alpha=0.5, label='原始信号', linewidth=0.8)
    axes[2].plot(time_axis, cleaned * 1e6, color='blue', label='去噪后信号', linewidth=0.8)
    axes[2].set_title('信号对比', fontsize=12)
    axes[2].set_ylabel('幅度 (μV)')
    axes[2].set_xlabel('时间 (秒)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_psd_comparison(raw, cleaned, subject_id, save_path):
    """绘制频域对比图 (功率谱密度)"""
    fs = TARGET_SFREQ
    
    # 计算PSD
    freqs_raw, psd_raw = welch(raw, fs, nperseg=256)
    freqs_clean, psd_clean = welch(cleaned, fs, nperseg=256)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.semilogy(freqs_raw, psd_raw, color='gray', alpha=0.7, label='原始信号', linewidth=1.5)
    ax.semilogy(freqs_clean, psd_clean, color='blue', label='去噪后信号', linewidth=1.5)
    
    # 标注关键频带
    ax.axvspan(0.5, 4, color='purple', alpha=0.1, label='Delta (0.5-4 Hz)')
    ax.axvspan(4, 8, color='blue', alpha=0.1, label='Theta (4-8 Hz)')
    ax.axvspan(8, 13, color='green', alpha=0.1, label='Alpha (8-13 Hz)')
    ax.axvspan(13, 30, color='orange', alpha=0.1, label='Beta (13-30 Hz)')
    
    ax.set_xlim(0, 40)
    ax.set_xlabel('频率 (Hz)', fontsize=12)
    ax.set_ylabel('功率谱密度 (V²/Hz)', fontsize=12)
    ax.set_title(f'功率谱密度对比 - {subject_id}', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# ================= 主测试流程 =================
print("=" * 60)
print("EEG去噪算法效果测试")
print("=" * 60)

# 1. 加载模型
print("\n[1/4] 正在加载去噪模型...")
try:
    model = build_model()
    model.load_weights(str(model_path))
    print(f"✅ 模型加载成功: {model_path.name}")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit()

# 2. 查找数据文件
raw_files = glob.glob(str(raw_dir / "*PSG.edf"))
print(f"\n[2/4] 发现 {len(raw_files)} 个受试者数据文件")

if len(raw_files) == 0:
    print("❌ 未找到数据文件！")
    exit()

# 3. 处理每个受试者
all_results = []

for idx, raw_path in enumerate(raw_files[:3]):  # 只测试前3个受试者
    subject_id = os.path.basename(raw_path).split('-')[0]
    print(f"\n[3/4] 处理受试者 {idx+1}/{min(3, len(raw_files))}: {subject_id}")
    
    try:
        # 读取EEG数据
        raw = mne.io.read_raw_edf(raw_path, preload=True, verbose=False)
        if raw.info['sfreq'] != TARGET_SFREQ:
            raw.resample(TARGET_SFREQ)
        
        # 获取EEG通道
        eeg_picks = mne.pick_channels(raw.info['ch_names'], include=['EEG Fpz-Cz'])
        if len(eeg_picks) == 0:
            eeg_picks = [0]
        data_raw = raw.get_data(picks=eeg_picks)[0]
        
        # 切片处理
        n_segments = min(len(data_raw) // SEGMENT_LEN, 500)  # 限制片段数量
        print(f"   - 数据长度: {len(data_raw)/TARGET_SFREQ:.1f}秒, 处理 {n_segments} 个片段")
        
        # 准备输入数据
        batch_in = []
        scales = []
        for i in range(n_segments):
            seg = data_raw[i*SEGMENT_LEN : (i+1)*SEGMENT_LEN]
            std = np.std(seg) if np.std(seg) != 0 else 1.0
            batch_in.append(seg / std)
            scales.append(std)
        
        # 模型推理
        print(f"   - 正在进行去噪推理...")
        input_tensor = np.array(batch_in).reshape(-1, SEGMENT_LEN, 1)
        predictions = model.predict(input_tensor, batch_size=128, verbose=0)
        predictions = np.squeeze(predictions)
        
        # 还原信号
        data_clean = np.array([predictions[i].flatten() * scales[i] for i in range(n_segments)]).flatten()
        data_raw_segment = data_raw[:len(data_clean)]
        
        # 计算指标
        nrr, cc, snr_imp = calculate_metrics(data_raw_segment, data_clean)
        print(f"   - NRR: {nrr:.2f}%, CC: {cc:.4f}, SNR改善: {snr_imp:.2f} dB")
        
        all_results.append({
            'subject': subject_id,
            'nrr': nrr,
            'cc': cc,
            'snr_imp': snr_imp
        })
        
        # 选择噪声最大的片段进行可视化
        noise_levels = [np.var(batch_in[i]) for i in range(n_segments)]
        noisy_indices = np.argsort(noise_levels)[-3:]  # 取噪声最大的3个片段
        
        # 绘制时域对比图
        for plot_idx, seg_idx in enumerate(noisy_indices[:1]):  # 只画1个
            start = seg_idx * SEGMENT_LEN
            end = (seg_idx + 1) * SEGMENT_LEN
            
            raw_seg = data_raw[start:end]
            clean_seg = predictions[seg_idx] * scales[seg_idx]
            
            save_path = os.path.join(output_dir, f"{subject_id}_时域对比.png")
            plot_comparison(raw_seg, clean_seg, subject_id, seg_idx, save_path)
            print(f"   - 时域对比图已保存: {subject_id}_时域对比.png")
        
        # 绘制频域对比图
        save_path = os.path.join(output_dir, f"{subject_id}_频域对比.png")
        plot_psd_comparison(data_raw_segment, data_clean, subject_id, save_path)
        print(f"   - 频域对比图已保存: {subject_id}_频域对比.png")
        
    except Exception as e:
        print(f"   ❌ 处理失败: {e}")

# 4. 输出总结报告
print("\n" + "=" * 60)
print("[4/4] 测试结果汇总")
print("=" * 60)
print(f"{'受试者':<12} | {'NRR (%)':<10} | {'CC':<8} | {'SNR改善 (dB)':<12}")
print("-" * 50)

for result in all_results:
    print(f"{result['subject']:<12} | {result['nrr']:<10.2f} | {result['cc']:<8.4f} | {result['snr_imp']:<12.2f}")

if all_results:
    avg_nrr = np.mean([r['nrr'] for r in all_results])
    avg_cc = np.mean([r['cc'] for r in all_results])
    avg_snr = np.mean([r['snr_imp'] for r in all_results])
    print("-" * 50)
    print(f"{'平均':<12} | {avg_nrr:<10.2f} | {avg_cc:<8.4f} | {avg_snr:<12.2f}")

print("\n" + "=" * 60)
print(f"✅ 测试完成！结果已保存至: {output_dir}")
print("=" * 60)
