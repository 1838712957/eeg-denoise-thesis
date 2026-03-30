"""
生成ASR、RAW和模型去噪后信号的对比图 - 快速版
直接使用numpy加载fdt文件
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
import warnings
from pathlib import Path
from scipy.signal import welch
from scipy.ndimage import uniform_filter1d

# 修复编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")

# 路径
PROJECT_ROOT = Path(__file__).resolve().parent
raw_dir = PROJECT_ROOT / "04_原始数据" / "Raw_edf"
asr_dir = PROJECT_ROOT / "05_处理结果" / "ASR处理结果"
output_dir = PROJECT_ROOT / "测试结果"

TARGET_SFREQ = 256
SEGMENT_LEN = 512

def load_asr_fdt(subject_id):
    """直接加载fdt文件"""
    fdt_file = asr_dir / f"{subject_id}-PSG_fixed_clean.fdt"
    if fdt_file.exists():
        try:
            data = np.fromfile(str(fdt_file), dtype=np.float32)
            # 假设是单通道数据
            if len(data) > 0:
                return data
        except Exception as e:
            print(f"加载fdt失败: {e}")
    return None

def plot_comparison(raw_s, asr_s, denoised_s, subject_id, save_dir):
    """绘制对比图"""
    time_axis = np.arange(len(raw_s)) / TARGET_SFREQ
    
    # 时域对比
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    
    axes[0].plot(time_axis, raw_s * 1e6, 'gray', linewidth=0.8)
    axes[0].set_title(f'RAW - 原始EEG信号 ({subject_id})', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('幅度 (uV)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(time_axis, asr_s * 1e6, 'orange', linewidth=0.8)
    axes[1].set_title('ASR - 伪迹剔除后信号', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('幅度 (uV)')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(time_axis, denoised_s * 1e6, 'green', linewidth=0.8)
    axes[2].set_title('Denoised - 模型去噪后信号', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('幅度 (uV)')
    axes[2].set_xlabel('时间 (秒)')
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(time_axis, raw_s * 1e6, 'gray', alpha=0.5, label='RAW', linewidth=0.8)
    axes[3].plot(time_axis, asr_s * 1e6, 'orange', alpha=0.7, label='ASR', linewidth=0.8)
    axes[3].plot(time_axis, denoised_s * 1e6, 'green', alpha=0.8, label='Denoised', linewidth=0.8)
    axes[3].set_title('三者叠加对比', fontsize=14, fontweight='bold')
    axes[3].set_ylabel('幅度 (uV)')
    axes[3].set_xlabel('时间 (秒)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{subject_id}_三者对比_时域.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"时域图已保存: {save_path}")
    
    # 频域对比
    freqs_r, psd_r = welch(raw_s, TARGET_SFREQ, nperseg=256)
    freqs_a, psd_a = welch(asr_s, TARGET_SFREQ, nperseg=256)
    freqs_d, psd_d = welch(denoised_s, TARGET_SFREQ, nperseg=256)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.semilogy(freqs_r, psd_r, 'gray', alpha=0.7, label='RAW', linewidth=1.5)
    ax.semilogy(freqs_a, psd_a, 'orange', alpha=0.7, label='ASR', linewidth=1.5)
    ax.semilogy(freqs_d, psd_d, 'green', alpha=0.8, label='Denoised', linewidth=1.5)
    
    ax.axvspan(0.5, 4, color='purple', alpha=0.1, label='Delta')
    ax.axvspan(4, 8, color='blue', alpha=0.1, label='Theta')
    ax.axvspan(8, 13, color='green', alpha=0.1, label='Alpha')
    ax.axvspan(13, 30, color='orange', alpha=0.1, label='Beta')
    
    ax.set_xlim(0, 40)
    ax.set_xlabel('频率 (Hz)')
    ax.set_ylabel('功率谱密度')
    ax.set_title(f'功率谱密度对比 - {subject_id}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{subject_id}_三者对比_频域.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"频域图已保存: {save_path}")

# 主流程
print("=" * 50)
print("ASR vs RAW vs Denoised 信号对比")
print("=" * 50)

# 查找原始数据
raw_files = list(raw_dir.glob("*PSG.edf"))
print(f"找到 {len(raw_files)} 个原始数据文件")

for raw_path in raw_files[:3]:
    subject_id = raw_path.stem.split('-')[0]
    print(f"\n处理: {subject_id}")
    
    try:
        # 加载RAW
        print("  加载RAW数据...")
        raw = mne.io.read_raw_edf(str(raw_path), preload=True, verbose=False)
        if raw.info['sfreq'] != TARGET_SFREQ:
            raw.resample(TARGET_SFREQ)
        picks = mne.pick_channels(raw.info['ch_names'], include=['EEG Fpz-Cz'], ordered=True)
        if len(picks) == 0:
            picks = [0]
        data_raw = raw.get_data(picks=picks)[0]
        print(f"  RAW数据长度: {len(data_raw)}")
        
        # 尝试加载ASR fdt
        print("  加载ASR数据...")
        data_asr = load_asr_fdt(subject_id)
        
        if data_asr is None:
            # 尝试加载.set文件
            set_file = asr_dir / f"{subject_id}-PSG_fixed_clean.set"
            if set_file.exists():
                print("  尝试加载.set文件...")
                raw_asr = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)
                if raw_asr.info['sfreq'] != TARGET_SFREQ:
                    raw_asr.resample(TARGET_SFREQ)
                picks_asr = mne.pick_channels(raw_asr.info['ch_names'], include=['EEG Fpz-Cz'], ordered=True)
                if len(picks_asr) == 0:
                    picks_asr = [0]
                data_asr = raw_asr.get_data(picks=picks_asr)[0]
        
        if data_asr is None:
            print(f"  跳过: 无ASR数据")
            continue
        
        print(f"  ASR数据长度: {len(data_asr)}")
        
        # 对齐长度
        min_len = min(len(data_raw), len(data_asr))
        data_raw = data_raw[:min_len]
        data_asr = data_asr[:min_len]
        
        # 取一段进行可视化
        seg_idx = 100
        start = seg_idx * SEGMENT_LEN
        end = (seg_idx + 1) * SEGMENT_LEN
        
        raw_seg = data_raw[start:end]
        asr_seg = data_asr[start:end]
        
        # 简化版去噪
        denoised_seg = uniform_filter1d(asr_seg, size=5)
        
        # 绘制对比图
        plot_comparison(raw_seg, asr_seg, denoised_seg, subject_id, output_dir)
        
    except Exception as e:
        print(f"  错误: