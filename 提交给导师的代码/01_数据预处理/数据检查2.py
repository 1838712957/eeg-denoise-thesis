import mne
import numpy as np
import os
import sys
from pathlib import Path

# ================= 配置路径 =================
# 清洗后的数据 (.set)
CORE_CODE_DIR = Path(__file__).resolve().parents[2] / "02_核心代码"
if str(CORE_CODE_DIR) not in sys.path:
    sys.path.append(str(CORE_CODE_DIR))
from bootstrap_paths import PROJECT_ROOT
clean_data_path = str(PROJECT_ROOT / "08_第三方工具" / "eeglab2025.1.0" / "SC4001E0_fixed_clean.set")
# 标签文件 (.edf) - 必须要有这个！
label_path = str(PROJECT_ROOT / "08_第三方工具" / "eeglab2025.1.0" / "SC4001EC-Hypnogram.edf")
# 输出保存的文件名
output_file = str(PROJECT_ROOT / "08_第三方工具" / "eeglab2025.1.0" / "sleep_dataset.npz")
# ===========================================

print("1. 正在加载数据...")
# 读取清洗后的脑电数据
raw = mne.io.read_raw_eeglab(clean_data_path, preload=True, verbose=False)

# 读取医生打好的标签 (Hypnogram)
annot = mne.read_annotations(label_path)
# 将标签"贴"到数据上
raw.set_annotations(annot, emit_warning=False)

print(f"   数据时长: {raw.times[-1]/3600:.2f} 小时")
print(f"   原始标签数: {len(annot)}")

# ================= 核心：建立标签映射 =================
# 将文字标签转换为数字 (0=清醒, 1=N1, 2=N2, 3=深睡, 4=REM)
# Sleep-EDF 标准：W=Wake, 1=N1, 2=N2, 3=N3, 4=N4(旧标准), R=REM
annotation_desc_2_event_id = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,  # 将 N3 和 N4 合并为 N3 (现代标准)
    'Sleep stage R': 4
}

print("2. 正在进行切片 (30秒/片)...")
# chunk_duration=30 表示把连续的睡眠阶段切成30秒一段
# events 就像是一个目录，告诉我们在哪一秒切一刀
events, _ = mne.events_from_annotations(
    raw, 
    event_id=annotation_desc_2_event_id, 
    chunk_duration=30.,
    verbose=False
)

# 创建 Epochs (切片)
# tmin=0, tmax=30 表示取事件发生后的30秒
# baseline=None 表示不再做基线校准 (我们已经洗过数据了)
epochs = mne.Epochs(
    raw, 
    events, 
    event_id=annotation_desc_2_event_id, 
    tmin=0, 
    tmax=30., 
    baseline=None, 
    preload=True,
    verbose=False
)

print(f"   成功切出片段数: {len(epochs)}")

# ================= 选择通道并保存 =================
# 通常毕设只用单通道 (Fpz-Cz) 或双通道
# 这里我们提取 Fpz-Cz 通道的数据
target_channel = 'EEG Fpz-Cz'
# 自动纠错通道名
if target_channel not in epochs.ch_names:
    target_channel = [ch for ch in epochs.ch_names if 'Fpz' in ch][0]

print(f"3. 提取通道 [{target_channel}] 并保存...")
# 拿到数据矩阵 X: (样本数, 1, 3000) -> 30秒 * 100Hz = 3000个点
X = epochs.get_data(picks=[target_channel]) 
X = X * 1e6 # 转换为微伏 (uV)，数值大一点方便训练

# 拿到标签 y: (样本数,)
y = epochs.events[:, 2]

# 保存为 .npz 格式 (Python 压缩包)
np.savez(output_file, x_data=X, y_labels=y)

print("="*30)
print(f"🎉 数据集制作完成！")
print(f"文件已保存至: {output_file}")
print(f"数据形状 (X): {X.shape} (片段数, 通道数, 时间点)")
print(f"标签形状 (y): {y.shape}")
print("="*30)
print("快去文件夹看看有没有 'sleep_dataset.npz' 这个文件！")