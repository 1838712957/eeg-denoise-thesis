"""
诊断脚本 - 将结果写入文件
"""
import sys
import os
from pathlib import Path

output_file = Path(__file__).resolve().parent / "diagnose_result.txt"

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("=" * 50 + "\n")
    f.write("诊断结果\n")
    f.write("=" * 50 + "\n\n")
    
    f.write(f"Python版本: {sys.version}\n")
    f.write(f"Python路径: {sys.executable}\n\n")
    
    f.write("检查依赖包:\n")
    
    try:
        import tensorflow as tf
        f.write(f"✅ TensorFlow: {tf.__version__}\n")
    except ImportError as e:
        f.write(f"❌ TensorFlow: {e}\n")
    
    try:
        import numpy as np
        f.write(f"✅ NumPy: {np.__version__}\n")
    except ImportError as e:
        f.write(f"❌ NumPy: {e}\n")
    
    try:
        import mne
        f.write(f"✅ MNE: {mne.__version__}\n")
    except ImportError as e:
        f.write(f"❌ MNE: {e}\n")
    
    try:
        import matplotlib
        f.write(f"✅ Matplotlib: {matplotlib.__version__}\n")
    except ImportError as e:
        f.write(f"❌ Matplotlib: {e}\n")
    
    # 检查文件路径
    PROJECT_ROOT = Path(__file__).resolve().parent
    f.write(f"\n项目根目录: {PROJECT_ROOT}\n")
    
    model_path = PROJECT_ROOT / "03_训练模型" / "去噪模型v2最终版.h5"
    f.write(f"模型路径: {model_path}\n")
    f.write(f"模型文件存在: {model_path.exists()}\n")
    
    raw_dir = PROJECT_ROOT / "04_原始数据" / "Raw_edf"
    f.write(f"数据目录: {raw_dir}\n")
    f.write(f"数据目录存在: {raw_dir.exists()}\n")
    
    if raw_dir.exists():
        files = os.listdir(raw_dir)
        f.write(f"数据文件数量: {len(files)}\n")
        f.write(f"前5个文件: {files[:5]}\n")
    
    f.write("\n" + "=" * 50 + "\n")
    f.write("诊断完成!\n")

print(f"诊断结果已保存到: {output_file}")