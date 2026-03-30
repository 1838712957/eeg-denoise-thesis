"""
诊断脚本 - 检查环境和依赖
"""
import sys
print("Python版本:", sys.version)
print("Python路径:", sys.executable)

# 检查依赖
print("\n检查依赖包:")
try:
    import tensorflow as tf
    print(f"✅ TensorFlow: {tf.__version__}")
except ImportError as e:
    print(f"❌ TensorFlow: {e}")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy: {e}")

try:
    import mne
    print(f"✅ MNE: {mne.__version__}")
except ImportError as e:
    print(f"❌ MNE: {e}")

try:
    import matplotlib
    print(f"✅ Matplotlib: {matplotlib.__version__}")
except ImportError as e:
    print(f"❌ Matplotlib: {e}")

# 检查文件路径
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
print(f"\n项目根目录: {PROJECT_ROOT}")

model_path = PROJECT_ROOT / "03_训练模型" / "去噪模型v2最终版.h5"
print(f"模型路径: {model_path}")
print(f"模型文件存在: {model_path.exists()}")

raw_dir = PROJECT_ROOT / "04_原始数据" / "Raw_edf"
print(f"数据目录: {raw_dir}")
print(f"数据目录存在: {raw_dir.exists()}")

if raw_dir.exists():
    import os
    files = os.listdir(raw_dir)
    print(f"数据文件数量: {len(files)}")
    print(f"前5个文件: {files[:5]}")