"""
生成ASR、RAW和模型去噪后信号的对比图
对比三种处理方式的信号质量
"""
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import mne
import warnings
from pathlib import Path
from tensorflow.keras import layers, models, Sequential
from scipy.signal import welch

# 修复Windows控制台编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ================= 路径配置 =================
PROJECT_ROOT = Path(__file__).resolve().parent
raw_dir = PROJECT_ROOT / "04_原始数据" / "Raw_edf"
asr_dir = PROJECT_ROOT / "05_处理结果" / "ASR处理结果"
model_path = PROJECT_ROOT / "03_训练模型" / "V4最优去噪模型.h5"
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


def load_asr_data(subject_id, asr_dir):
    """加载ASR处理后的数据"""
    # 提取基础名称，如 SC4001E0-PSG -> SC4001E0
    base_id = subject_id.replace('-PSG', '').replace('_PSG', '')
    
    asr_files = [
        asr_dir / f"{base_id}_ASR.set",
        asr_dir / f"{subject_id}_fixed_clean.set",
        asr_dir / f"{base_id}-PSG_fixed_clean.set",
    ]
    
    for asr_file in asr_files:
        if asr_file.exists():
            try:
                raw = mne.io.read_raw_eeglab(str(asr_file), preload=True, verbose=False)
                if raw.info['sfreq'] != TARGET_SFREQ:
                    raw.resample(TARGET_SFREQ)
                
                # 尝试多个通道名
                channel_names = raw.info['ch_names']
                eeg_picks = None
                
                # 尝试常见EEG通道名
                for ch in ['EEG Fpz-Cz', 'EEG Fpz-Cz ', 'Fpz-Cz', 'EEG 1', 'EEG 2', '1', '0']:
                    if ch in channel_names:
                        eeg_picks = mne.pick_channels(channel_names, include=[ch], ordered=True)
                        if len(eeg_picks) > 0:
                            print(f"   使用通道: {ch}")
                            break
                
                # 如果还是没找到，使用第一个通道
                if eeg_picks is None or len(eeg_picks) == 0:
                    eeg_picks = [0]
                    print(f"   使用第一个通道")
                
                return raw.get_data(picks=eeg_picks)[0]
            except Exception as e:
                print(f"   尝试加载 {asr_file.name} 失败: {e}")
                continue
    return None


def plot_triple_comparison(raw_signal, asr_signal, denoised_signal, subject_id, seg_idx, save_path):
    """绘制三信号对比图（时域）"""
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    
    time_axis = np.arange(len(raw_signal)) / TARGET_SFREQ
    
    axes[0].plot(time_axis, raw_signal * 1e6, color='gray', linewidth=0.8)
    axes[0].set_title(f'RAW - 原始EEG信号 ({subject_id})', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('幅度 (μV)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, time_axis[-1])
    
    axes[1].plot(time_axis, asr_signal * 1e6, color='orange', linewidth=0.8)
    axes[1].set_title('ASR - 伪迹剔除后信号', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('幅度 (μV)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, time_axis[-1])
    
    axes[2].plot(time_axis, denoised_signal * 1e6, color='green', linewidth=0.8)
    axes[2].set_title('Denoised - 模型去噪后信号', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('幅度 (μV)')
    axes[2].set_xlabel('时间 (秒)')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, time_axis[-1])
    
    axes[3].plot(time_axis, raw_signal * 1e6, color='gray', alpha=0.5, label='RAW', linewidth=0.8)
    axes[3].plot(time_axis, asr_signal * 1e6, color='orange', alpha=0.7, label='ASR', linewidth=0.8)
    axes[3].plot(time_axis, denoised_signal * 1e6, color='green', alpha=0.8, label='Denoised', linewidth=0.8)
    axes[3].set_title('三者叠加对比', fontsize=14, fontweight='bold')
    axes[3].set_ylabel('幅度 (μV)')
    axes[3].set_xlabel('时间 (秒)')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlim(0, time_axis[-1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   时域对比图已保存: {save_path}")


def plot_psd_triple(raw_signal, asr_signal, denoised_signal, subject_id, save_path):
    """绘制三信号频域对比图"""
    fs = TARGET_SFREQ
    
    freqs_raw, psd_raw = welch(raw_signal, fs, nperseg=256)
    freqs_asr, psd_asr = welch(asr_signal, fs, nperseg=256)
    freqs_clean, psd_clean = welch(denoised_signal, fs, nperseg=256)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.semilogy(freqs_raw, psd_raw, color='gray', alpha=0.7, label='RAW', linewidth=1.5)
    ax.semilogy(freqs_asr, psd_asr, color='orange', alpha=0.7, label='ASR', linewidth=1.5)
    ax.semilogy(freqs_clean, psd_clean, color='green', alpha=0.8, label='Denoised', linewidth=1.5)
    
    ax.axvspan(0.5, 4, color='purple', alpha=0.1, label='Delta (0.5-4 Hz)')
    ax.axvspan(4, 8, color='blue', alpha=0.1, label='Theta (4-8 Hz)')
    ax.axvspan(8, 13, color='green', alpha=0.1, label='Alpha (8-13 Hz)')
    ax.axvspan(13, 30, color='orange', alpha=0.1, label='Beta (13-30 Hz)')
    
    ax.set_xlim(0, 40)
    ax.set_xlabel('频率 (Hz)', fontsize=12)
    ax.set_ylabel('功率谱密度 (V²/Hz)', fontsize=12)
    ax.set_title(f'功率谱密度对比 - {subject_id}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   频域对比图已保存: {save_path}")


def calculate_snr(signal, noise):
    """计算信噪比"""
    if np.var(noise) == 0:
        return 0
    return 10 * np.log10(np.var(signal) / np.var(noise) + 1e-10)


def calculate_correlation(s1, s2):
    """计算两信号相关系数"""
    return np.corrcoef(s1, s2)[0, 1]


# ================= 主流程 =================
print("=" * 60)
print("ASR vs RAW vs Denoised 信号对比分析")
print("=" * 60)

# 1. 加载模型
print("\n[1/4] 正在加载去噪模型...")
try:
    from tensorflow.keras.models import load_model
    model = load_model(str(model_path), compile=False)
    print(f"✅ 模型加载成功: {model_path.name}")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit()

# 2. 查找数据文件 - 处理SC4001E0等
raw_files = glob.glob(str(raw_dir / "*PSG.edf"))
print(f"\n[2/4] 发现 {len(raw_files)} 个原始数据文件")

if len(raw_files) == 0:
    print("❌ 未找到数据文件！")
    exit()

# 3. 处理每个受试者
all_results = []

# 筛选SC4001E0, SC4002E0, SC4011E0
subject_files = [f for f in raw_files if any(s in os.path.basename(f) for s in ['SC4001E0', 'SC4002E0', 'SC4011E0'])]
subject_files = sorted(subject_files)[:3]  # 取前3个

print(f"   筛选出: {[os.path.basename(f) for f in subject_files]}")

for idx, raw_path in enumerate(subject_files):
    subject_id = os.path.basename(raw_path).replace('.edf', '')
    print(f"\n[3/4] 处理受试者 {idx+1}/{min(3, len(raw_files))}: {subject_id}")
    
    try:
        # 加载RAW数据
        raw = mne.io.read_raw_edf(raw_path, preload=True, verbose=False)
        if raw.info['sfreq'] != TARGET_SFREQ:
            raw.resample(TARGET_SFREQ)
        
        # 尝试多个通道名 (DREAMS数据集)
        channel_candidates = ['CZ-A1', 'FP1-A2', 'O1-A2', 'FP2-A1', 'O2-A1', 'Fpz-Cz', 'EEG Fpz-Cz']
        eeg_picks = None
        for ch in channel_candidates:
            if ch in raw.info['ch_names']:
                eeg_picks = mne.pick_channels(raw.info['ch_names'], include=[ch], ordered=True)
                if len(eeg_picks) > 0:
                    print(f"   使用通道: {ch}")
                    break
        
        if eeg_picks is None or len(eeg_picks) == 0:
            eeg_picks = [0]
            print(f"   使用第一个通道")
        
        data_raw = raw.get_data(picks=eeg_picks)[0]
        
        # 加载ASR数据
        data_asr = load_asr_data(subject_id, asr_dir)
        
        if data_asr is None:
            print(f"   ⚠️ 未找到ASR处理后的数据，跳过该受试者")
            continue
        
        # 对齐数据长度
        min_len = min(len(data_raw), len(data_asr))
        data_raw = data_raw[:min_len]
        data_asr = data_asr[:min_len]
        
        # 切片处理
        n_segments = min(min_len // SEGMENT_LEN, 500)
        
        # 准备输入数据
        batch_in = []
        scales = []
        for i in range(n_segments):
            seg = data_asr[i*SEGMENT_LEN : (i+1)*SEGMENT_LEN]
            std = np.std(seg) if np.std(seg) != 0 else 1.0
            batch_in.append(seg / std)
            scales.append(std)
        
        # 模型推理
        print(f"   - 正在进行去噪推理...")
        input_tensor = np.array(batch_in).reshape(-1, SEGMENT_LEN, 1)
        predictions = model.predict(input_tensor, batch_size=128, verbose=0)
        predictions = np.squeeze(predictions)
        
        # 还原信号
        data_denoised = np.array([predictions[i].flatten() * scales[i] for i in range(n_segments)]).flatten()
        data_asr_segment = data_asr[:len(data_denoised)]
        data_raw_segment = data_raw[:len(data_denoised)]
        
        # 选择噪声最大的片段进行可视化
        noise_levels = [np.var(batch_in[i]) for i in range(n_segments)]
        noisy_indices = np.argsort(noise_levels)[-3:]
        
        # 绘制时域对比图
        for plot_idx, seg_idx in enumerate(noisy_indices[:1]):
            start = seg_idx * SEGMENT_LEN
            end = (seg_idx + 1) * SEGMENT_LEN
            
            raw_seg = data_raw_segment[start:end]
            asr_seg = data_asr_segment[start:end]
            denoised_seg = predictions[seg_idx] * scales[seg_idx]
            
            save_path = os.path.join(output_dir, f"{subject_id}_三者对比_时域.png")
            plot_triple_comparison(raw_seg, asr_seg, denoised_seg, subject_id, seg_idx, save_path)
        
        # 绘制频域对比图
        save_path = os.path.join(output_dir, f"{subject_id}_三者对比_频域.png")
        plot_psd_triple(data_raw_segment, data_asr_segment, data_denoised, subject_id, save_path)
        
        # 计算统计指标
        asr_noise = data_raw_segment - data_asr_segment
        denoised_noise = data_raw_segment - data_denoised
        
        snr_raw = calculate_snr(data_raw_segment, data_raw_segment)
        snr_asr = calculate_snr(data_asr_segment, asr_noise)
        snr_denoised = calculate_snr(data_denoised, denoised_noise)
        
        cc_raw_asr = calculate_correlation(data_raw_segment, data_asr_segment)
        cc_raw_denoised = calculate_correlation(data_raw_segment, data_denoised)
        
        print(f"   - SNR改善: ASR={snr_asr:.2f}dB, Denoised={snr_denoised:.2f}dB")
        print(f"   - 相关系数: RAW-ASR={cc_raw_asr:.4f}, RAW-Denoised={cc_raw_denoised:.4f}")
        
        all_results.append({
            'subject': subject_id,
            'snr_asr': snr_asr,
            'snr_denoised': snr_denoised,
            'cc_raw_asr': cc_raw_asr,
            'cc_raw_denoised': cc_raw_denoised
        })
        
    except Exception as e:
        print(f"   ❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

# 4. 输出总结报告
print("\n" + "=" * 60)
print("[4/4] 测试结果汇总")
print("=" * 60)
print(f"{'受试者':<12} | {'SNR_ASR':<10} | {'SNR_Denoised':<12} | {'CC_RAW_ASR':<10} | {'CC_RAW_Denoised':<14}")
print("-" * 80)

for result in all_results:
        print(f"{result['subject']:<12} | {result['snr_asr']:<10.2f} | {result['snr_denoised']:<12.2f} | {result['cc_raw_asr']:<10.4f} | {result['cc_raw_denoised']:<14.4f}")

if all_results:
    avg_snr_asr = np.mean([r['snr_asr'] for r in all_results])
    avg_snr_denoised = np.mean([r['snr_denoised'] for r in all_results])
    avg_cc_asr = np.mean([r['cc_raw_asr'] for r in all_results])
    avg_cc_denoised = np.mean([r['cc_raw_denoised'] for r in all_results])
    print("-" * 80)
    print(f"{'平均':<12} | {avg_snr_asr:<10.2f} | {avg_snr_denoised:<12.2f} | {avg_cc_asr:<10.4f} | {avg_cc_denoised:<14.4f}")

print("\n" + "=" * 60)
print(f"✅ 测试完成！结果已保存至: {output_dir}")
print("=" * 60)
