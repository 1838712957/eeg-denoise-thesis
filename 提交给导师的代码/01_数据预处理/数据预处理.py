import os
import glob
import mne
import numpy as np
import warnings
import sys
from pathlib import Path

# ================= 0. 屏蔽警告 =================
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ================= 1. 路径配置 =================
CORE_CODE_DIR = Path(__file__).resolve().parents[2] / "02_核心代码"
if str(CORE_CODE_DIR) not in sys.path:
    sys.path.append(str(CORE_CODE_DIR))
from bootstrap_paths import PROJECT_ROOT
dreams_dir = str(PROJECT_ROOT / "07_DREAMS数据集")
# 目标保存路径
save_dir = str(PROJECT_ROOT / "08_第三方工具" / "EEGdenoiseNet-master" / "code" / "data")
save_path = os.path.join(save_dir, "DREAMS_all_epochs.npy")

TARGET_SFREQ = 256  
EPOCH_LEN = 512     

# ================= 2. 自动修复路径 =================
# 关键一步：如果目标文件夹不存在，强制创建它！
if not os.path.exists(save_dir):
    print(f"🔧 检测到目录不存在，正在创建: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

# ================= 3. 转换逻辑 =================
print("🚀 [Step 1] 开始制作 DREAMS 伪差数据集...")
noise_epochs = []

edf_files = glob.glob(os.path.join(dreams_dir, "*.edf"))
if not edf_files:
    print(f"❌ 错误：在 {dreams_dir} 没找到 .edf 文件！")
    exit()

print(f"📂 找到 {len(edf_files)} 个伪差文件，正在处理...")

for fpath in edf_files:
    try:
        # 读取数据 (verbose=False 减少刷屏)
        raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
        
        # 重采样
        if raw.info['sfreq'] != TARGET_SFREQ:
            raw.resample(TARGET_SFREQ)
        
        # 提取所有通道数据
        data = raw.get_data() 
        
        # 遍历通道并切片
        for ch_data in data:
            n_seg = len(ch_data) // EPOCH_LEN
            for i in range(n_seg):
                seg = ch_data[i*EPOCH_LEN : (i+1)*EPOCH_LEN]
                
                # 排除纯直线的死通道 (std=0)
                if np.std(seg) > 1e-6: # 稍微设个阈值，排除极小值
                    # 标准化
                    seg = (seg - np.mean(seg)) / np.std(seg)
                    noise_epochs.append(seg)
                    
    except Exception as e:
        print(f"⚠️ 跳过文件 {os.path.basename(fpath)}: {e}")

# 保存
if len(noise_epochs) > 0:
    noise_array = np.array(noise_epochs)
    np.save(save_path, noise_array)
    print(f"\n✅ 成功！")
    print(f"   生成了 {len(noise_array)} 个伪差片段")
    print(f"   已保存至: {save_path}")
else:
    print("❌ 生成失败，有效片段为 0。")