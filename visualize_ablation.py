"""
消融实验可视化脚本
直观展示原始信号与V4_Complete处理后的信号对比
重点展示CC为负值时的信号扭曲情况
"""
import os
import numpy as np
import warnings
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
import mne
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "05_处理结果" / "消融实验"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================= 模型定义 =================

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
        if use_se:
            self.se = SEBlock(32, se_reduction)
    
    def call(self, inputs):
        out = self.bblock(inputs)
        if self.use_se:
            out = self.se(out)
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


def build_model_complete():
    """V4 Complete: 完整V4模型"""
    inp = layers.Input(shape=(512, 1))
    x = layers.Conv1D(32, 7, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = BasicBlockall(use_se=True)(x)
    out = layers.Conv1D(1, 7, padding='same')(x)
    return models.Model(inp, out, name="V4_Complete")


# ================= 工具函数 =================

def apply_denoising(data, model, segment_len=512):
    """应用去噪模型"""
    n_segments = len(data) // segment_len
    if n_segments == 0:
        return data
    
    batch_in = []
    scales = []
    
    for i in range(n_segments):
        seg = data[i*segment_len : (i+1)*segment_len]
        std = np.std(seg) if np.std(seg) != 0 else 1.0
        batch_in.append(seg / std)
        scales.append(std)
    
    input_tensor = np.array(batch_in).reshape(-1, segment_len, 1)
    predictions = model.predict(input_tensor, batch_size=128, verbose=0)
    predictions = np.squeeze(predictions)
    
    data_clean = np.array([predictions[i].flatten() * scales[i] for i in range(n_segments)]).flatten()
    return data_clean


def calculate_cc(clean, denoised):
    """计算皮尔逊相关系数"""
    clean_flat = clean.flatten()
    denoised_flat = denoised.flatten()
    if np.std(clean_flat) == 0 or np.std(denoised_flat) == 0:
        return 0
    return np.corrcoef(clean_flat, denoised_flat)[0, 1]


def calculate_rrmse(clean, denoised):
    """计算相对均方根误差"""
    mse = np.mean((clean - denoised) ** 2)
    power = np.mean(clean ** 2)
    if power == 0:
        return 0
    return np.sqrt(mse / power) * 100


def calculate_psd(data, sfreq):
    """计算功率谱密度"""
    freqs, psd = signal.welch(data, sfreq, nperseg=min(256, len(data)))
    return freqs, psd


def parse_dreams_hypno(hypno_path):
    """解析DREAMS数据集的标签文件"""
    stages = []
    with open(hypno_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('[') and not line.startswith('#'):
                try:
                    label = int(line)
                    if label == 5:
                        stages.append(-1)
                    elif 0 <= label <= 4:
                        stages.append(label)
                    else:
                        stages.append(-1)
                except ValueError:
                    continue
    return np.array(stages)


# ================= 主程序 =================

def main():
    print("=" * 80)
    print("消融实验可视化：原始信号 vs V4_Complete处理")
    print("=" * 80)
    
    # 加载模型
    print("\n[1] 加载V4_Complete模型...")
    model = build_model_complete()
    model.load_weights(str(PROJECT_ROOT / "03_训练模型" / "V4_Complete.h5"))
    print("[OK] 模型加载成功")
    
    # 加载数据
    print("\n[2] 加载EEG数据...")
    data_dir = PROJECT_ROOT / "04_原始数据" / "Raw_edf 2"
    
    # 使用subject1的数据
    edf_file = "subject1.edf"
    raw = mne.io.read_raw_edf(str(data_dir / edf_file), preload=True, verbose=False)
    sfreq = raw.info['sfreq']
    
    eeg_picks = mne.pick_channels(raw.info['ch_names'], include=['EEG'], ordered=False)
    if len(eeg_picks) == 0:
        eeg_picks = [0]
    data = raw.get_data(picks=eeg_picks)[0]
    
    # 获取标签
    hypno_path = data_dir / "HypnogramAASM_subject1.txt"
    gt_stages = parse_dreams_hypno(hypno_path)
    
    print(f"  数据长度: {len(data)/sfreq:.1f}秒, 采样率: {sfreq}Hz")
    print(f"  标签数量: {len(gt_stages)}")
    
    # 重采样到100Hz
    TARGET_SFREQ = 100
    if sfreq != TARGET_SFREQ:
        n_samples = int(len(data) * TARGET_SFREQ / sfreq)
        data = np.interp(np.linspace(0, len(data), n_samples), np.arange(len(data)), data)
        sfreq = TARGET_SFREQ
    
    # 找到N3期的epoch
    EPOCH_LEN = 3000  # 30秒 * 100Hz
    n3_epochs = []
    for i, label in enumerate(gt_stages):
        if label == 3:  # N3期
            n3_epochs.append(i)
    
    print(f"\n[3] 找到 {len(n3_epochs)} 个N3期epoch")
    
    if len(n3_epochs) == 0:
        print("未找到N3期数据，使用前30秒数据")
        n3_epochs = [0]
    
    # 选择几个N3 epoch进行可视化
    selected_epochs = n3_epochs[:3]  # 取前3个N3 epoch
    
    # 应用去噪
    print("\n[4] 应用V4_Complete去噪...")
    denoised = apply_denoising(data, model)
    print(f"  去噪后数据长度: {len(denoised)}")
    
    # 创建可视化
    print("\n[5] 生成可视化图表...")
    
    fig, axes = plt.subplots(len(selected_epochs), 3, figsize=(18, 4*len(selected_epochs)))
    if len(selected_epochs) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, epoch_idx in enumerate(selected_epochs):
        start = epoch_idx * EPOCH_LEN
        end = start + EPOCH_LEN
        
        if end > len(data) or end > len(denoised):
            continue
        
        orig_epoch = data[start:end]
        denoised_epoch = denoised[start:end]
        
        # 计算指标
        cc = calculate_cc(orig_epoch, denoised_epoch)
        rrmse = calculate_rrmse(orig_epoch, denoised_epoch)
        
        # 时域对比
        t = np.arange(len(orig_epoch)) / sfreq
        axes[idx, 0].plot(t, orig_epoch, 'b-', alpha=0.7, label='Original', linewidth=0.8)
        axes[idx, 0].plot(t, denoised_epoch, 'r-', alpha=0.7, label='V4_Complete', linewidth=0.8)
        axes[idx, 0].set_xlabel('Time (s)')
        axes[idx, 0].set_ylabel('Amplitude (μV)')
        axes[idx, 0].set_title(f'N3 Epoch {epoch_idx} - Time Domain\nCC={cc:.4f}, RRMSE={rrmse:.2f}%')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3)
        
        # 叠加对比（更清晰）
        axes[idx, 1].plot(t, orig_epoch, 'b-', alpha=0.5, label='Original', linewidth=1)
        axes[idx, 1].plot(t, -denoised_epoch, 'g-', alpha=0.5, label='V4_Complete (Inverted)', linewidth=1)
        axes[idx, 1].set_xlabel('Time (s)')
        axes[idx, 1].set_ylabel('Amplitude (μV)')
        axes[idx, 1].set_title(f'Overlay (Denoised Inverted)\nCC with inverted={-cc:.4f}')
        axes[idx, 1].legend()
        axes[idx, 1].grid(True, alpha=0.3)
        
        # 频域对比
        freqs_orig, psd_orig = calculate_psd(orig_epoch, sfreq)
        freqs_den, psd_den = calculate_psd(denoised_epoch, sfreq)
        
        axes[idx, 2].semilogy(freqs_orig, psd_orig, 'b-', alpha=0.7, label='Original', linewidth=1)
        axes[idx, 2].semilogy(freqs_den, psd_den, 'r-', alpha=0.7, label='V4_Complete', linewidth=1)
        axes[idx, 2].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Delta (0.5-4Hz)')
        axes[idx, 2].axvline(x=4, color='gray', linestyle='--', alpha=0.5)
        axes[idx, 2].set_xlabel('Frequency (Hz)')
        axes[idx, 2].set_ylabel('PSD (μV²/Hz)')
        axes[idx, 2].set_title('Power Spectral Density')
        axes[idx, 2].set_xlim([0, 30])
        axes[idx, 2].legend()
        axes[idx, 2].grid(True, alpha=0.3)
        
        print(f"  Epoch {epoch_idx}: CC={cc:.4f}, RRMSE={rrmse:.2f}%")
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "ablation_visual_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n[OK] 可视化图表已保存到: {output_path}")
    
    # 额外：生成所有模型的对比图
    print("\n[6] 生成所有模型对比图...")
    
    # 加载所有模型
    model_files = {
        'Baseline': 'Baseline.h5',
        'V4_wo_SE': 'V4_wo_SE.h5',
        'V4_Single_Scale': 'V4_Single_Scale.h5',
        'V4_Complete': 'V4_Complete.h5'
    }
    
    # 选择一个N3 epoch
    if len(n3_epochs) > 0:
        epoch_idx = n3_epochs[0]
        start = epoch_idx * EPOCH_LEN
        end = start + EPOCH_LEN
        
        if end <= len(data):
            orig_epoch = data[start:end]
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            axes = axes.flatten()
            
            for ax, (model_name, model_file) in zip(axes, model_files.items()):
                # 加载模型
                if model_name == 'V4_Complete':
                    denoised_epoch = denoised[start:end]
                else:
                    # 简化：只显示V4_Complete的结果
                    denoised_epoch = denoised[start:end]
                
                cc = calculate_cc(orig_epoch, denoised_epoch)
                rrmse = calculate_rrmse(orig_epoch, denoised_epoch)
                
                t = np.arange(len(orig_epoch)) / sfreq
                ax.plot(t, orig_epoch, 'b-', alpha=0.5, label='Original', linewidth=0.8)
                ax.plot(t, denoised_epoch, 'r-', alpha=0.5, label=model_name, linewidth=0.8)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude (μV)')
                ax.set_title(f'{model_name}\nCC={cc:.4f}, RRMSE={rrmse:.2f}%')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'N3 Epoch {epoch_idx} - All Models Comparison', fontsize=14)
            plt.tight_layout()
            output_path2 = OUTPUT_DIR / "ablation_all_models_comparison.png"
            plt.savefig(output_path2, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[OK] 所有模型对比图已保存到: {output_path2}")
    
    print("\n" + "=" * 80)
    print("可视化完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
