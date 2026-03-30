import os
import glob
import mne
import sys
from pathlib import Path

# ================= 配置路径 =================
CORE_CODE_DIR = Path(__file__).resolve().parents[2] / "02_核心代码"
if str(CORE_CODE_DIR) not in sys.path:
    sys.path.append(str(CORE_CODE_DIR))
from bootstrap_paths import PROJECT_ROOT
data_dir = str(PROJECT_ROOT / "04_原始数据" / "Raw_edf")

# ================= 自动查找并读取第一个 EDF 文件 =================
edf_files = glob.glob(os.path.join(data_dir, "*.edf"))

if len(edf_files) == 0:
    print(f"❌ 错误：在 {data_dir} 里没找到 .edf 文件！")
else:
    # 只读第一个文件做代表
    sample_file = edf_files[0]
    print(f"📂 正在检查文件: {os.path.basename(sample_file)} ...\n")
    
    try:
        # 读取文件头信息（不需要加载全部数据，速度很快）
        raw = mne.io.read_raw_edf(sample_file, preload=False, verbose=False)
        
        # 获取核心信息
        ch_names = raw.ch_names  # 通道名称列表
        n_chan = len(ch_names)   # 通道总数
        sfreq = raw.info['sfreq'] # 采样率
        
        # 打印报告
        print("="*40)
        print(f"📊 【数据体检报告】")
        print(f"   - 电极(通道)总数: {n_chan} 个")
        print(f"   - 采样率 (Hz):    {sfreq}")
        print("="*40)
        
        print("\n🧐 具体电极名称列表:")
        for i, name in enumerate(ch_names):
            # 判断电极类型（简单推测）
            type_guess = "未知"
            if "EEG" in name.upper() or "FP" in name.upper() or "CZ" in name.upper():
                type_guess = "🧠 脑电 (EEG)"
            elif "EOG" in name.upper() or "EYE" in name.upper():
                type_guess = "👁️ 眼电 (EOG)"
            elif "EMG" in name.upper() or "CHIN" in name.upper():
                type_guess = "💪 肌电 (EMG)"
            elif "ECG" in name.upper() or "EKG" in name.upper():
                type_guess = "❤️ 心电 (ECG)"
                
            print(f"   [{i}] {name:<15} --> {type_guess}")
            
    except Exception as e:
        print(f"❌ 读取失败: {e}")