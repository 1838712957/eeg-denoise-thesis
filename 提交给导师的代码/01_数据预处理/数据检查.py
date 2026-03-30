import mne
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# ================= 配置路径 (不用改，只要你文件没动) =================
CORE_CODE_DIR = Path(__file__).resolve().parents[2] / "02_核心代码"
if str(CORE_CODE_DIR) not in sys.path:
    sys.path.append(str(CORE_CODE_DIR))
from bootstrap_paths import PROJECT_ROOT
raw_path = str(PROJECT_ROOT / "08_第三方工具" / "eeglab2025.1.0" / "SC4001E0-PSG.edf")
clean_path = str(PROJECT_ROOT / "08_第三方工具" / "eeglab2025.1.0" / "SC4001E0_fixed_clean.set")
# ===================================================================

print("正在加载数据...")
raw = mne.io.read_raw_edf(raw_path, preload=True, verbose=False)
clean = mne.io.read_raw_eeglab(clean_path, preload=True, verbose=False)

# 选取一个通道
target_chan = 'EEG Fpz-Cz'
# 自动匹配清洗后的通道名
if target_chan not in clean.ch_names:
    candidates = [ch for ch in clean.ch_names if 'Fpz' in ch]
    target_chan_clean = candidates[0] if candidates else clean.ch_names[0]
else:
    target_chan_clean = target_chan

print(f"正在对比: {target_chan} vs {target_chan_clean}")

# 获取全部数据
data_raw, times = raw[target_chan]
data_clean, _ = clean[target_chan_clean]

data_raw = data_raw[0] * 1e6  # 转为微伏
data_clean = data_clean[0] * 1e6

# -------------------------------------------------------
# 🔥 核心修改：自动寻找差异最大的时间点
# -------------------------------------------------------
print("正在寻找去噪效果最明显的片段...")
diff = np.abs(data_raw - data_clean) # 计算差值
max_diff_idx = np.argmax(diff)       # 找到差值最大的那个点
center_time = times[max_diff_idx]    # 获取那个点的时间

print(f"找到最大差异点：第 {center_time:.2f} 秒")

# 以前后各 15 秒截取（总共 30 秒）
start_idx = max(0, max_diff_idx - 1500) # 100Hz采样率，15秒=1500点
end_idx = min(len(times), max_diff_idx + 1500)

plot_times = times[start_idx:end_idx]
plot_raw = data_raw[start_idx:end_idx]
plot_clean = data_clean[start_idx:end_idx]

# -------------------------------------------------------
# 画图
# -------------------------------------------------------
plt.figure(figsize=(15, 8))

# 子图1：波形对比
plt.subplot(2, 1, 1)
plt.plot(plot_times, plot_raw, color='lightgray', label='Original (Raw)', linewidth=2)
plt.plot(plot_times, plot_clean, color='#1f77b4', label='Cleaned (ASR)', linewidth=1, alpha=0.9)
plt.title(f'Before vs After (Most Significant Correction at {center_time:.0f}s)', fontsize=14)
plt.ylabel('Amplitude (uV)')
plt.legend(loc='upper right')

# 子图2：展示被切除的噪音（差值）
plt.subplot(2, 1, 2)
plt.plot(plot_times, plot_raw - plot_clean, color='red', label='Removed Noise', linewidth=1)
plt.title('What was removed (Noise)', fontsize=12)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (uV)')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()