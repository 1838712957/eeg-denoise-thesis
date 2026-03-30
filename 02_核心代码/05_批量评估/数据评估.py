import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# ================= 配置 =================
# ✅ 已根据你的截图修改为正确路径
CORE_CODE_DIR = Path(__file__).resolve().parents[2] / "02_核心代码"
if str(CORE_CODE_DIR) not in sys.path:
    sys.path.append(str(CORE_CODE_DIR))
from bootstrap_paths import PROJECT_ROOT
data_path = str(PROJECT_ROOT / "05_模型输出" / "nn_output")

# ================= 1. 加载数据 =================
print(f"正在从 {data_path} 加载数据...")
try:
    # 加载 loss 历史
    history = np.load(os.path.join(data_path, 'loss_history.npy'), allow_pickle=True).item()
    
    # 加载测试集数据
    clean_eeg = np.load(os.path.join(data_path, 'EEG_test.npy'))
    noisy_eeg = np.load(os.path.join(data_path, 'noiseinput_test.npy'))
    denoised_eeg = np.load(os.path.join(data_path, 'Denoiseoutput_test.npy'))
    
    print("✅ 数据加载成功！")
    print(f"数据形状检查: EEG_test={clean_eeg.shape}")
except Exception as e:
    print(f"❌ 数据加载失败，请检查路径！错误信息: {e}")
    exit()

# ================= 2. 画 Loss 曲线 =================
plt.figure(figsize=(10, 5))

# 提取 loss (根据你的 history 结构)
train_loss = history['loss']['train_mse']
val_loss = history['loss']['val_mse']

plt.plot(train_loss, label='Train Loss', linewidth=2, color='blue')
plt.plot(val_loss, label='Validation Loss', linewidth=2, color='orange')
plt.title('Training & Validation Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# 保存图片到当前目录
plt.savefig('Loss_Curve.png', dpi=300) 
print("✅ Loss 曲线已保存为 Loss_Curve.png")
plt.show()

# ================= 3. 计算指标 (RRMSE & CC) =================
def calc_rrmse(clean, est):
    # RRMSE: 相对均方根误差 (越小越好)
    rms_clean = np.sqrt(np.mean(clean**2, axis=1))
    rms_diff = np.sqrt(np.mean((clean - est)**2, axis=1))
    # 防止除零错误
    rms_clean[rms_clean == 0] = 1e-8 
    return np.mean(rms_diff / rms_clean)

def calc_cc(clean, est):
    # CC: 皮尔逊相关系数 (越接近 1 越好)
    cc_list = []
    # 为了快速计算，如果数据量太大，只算前 2000 个
    limit = min(len(clean), 2000)
    for i in range(limit):
        # 展平为一维数组计算相关性
        c_sig = clean[i].flatten()
        e_sig = est[i].flatten()
        
        if np.std(c_sig) == 0 or np.std(e_sig) == 0:
            continue
            
        c = np.corrcoef(c_sig, e_sig)[0, 1]
        cc_list.append(c)
    return np.mean(cc_list)

# 确保数据维度一致 (去除多余的维度 1)
clean_eeg = np.squeeze(clean_eeg)
denoised_eeg = np.squeeze(denoised_eeg)
noisy_eeg = np.squeeze(noisy_eeg)

print("\n正在计算 RRMSE 和 CC 指标...")
rrmse_val = calc_rrmse(clean_eeg, denoised_eeg)
cc_val = calc_cc(clean_eeg, denoised_eeg)

print(f"\n{'='*20} 最终实验结果 {'='*20}")
print(f"RRMSE (相对误差，越低越好) :  {rrmse_val:.4f}")
print(f"CC    (相关系数，越高越好) :  {cc_val:.4f}")
print(f"{'='*56}\n")

# ================= 4. 画波形对比图 =================
# 随机挑选 3 个样本展示
indices = [10, 50, 100]  # 你可以修改这些数字来看不同的样本

plt.figure(figsize=(15, 10))
for i, idx in enumerate(indices):
    plt.subplot(3, 1, i+1)
    
    # 原始干净信号 (黑线)
    plt.plot(clean_eeg[idx], label='Original EEG (Clean)', color='black', alpha=0.8, linewidth=1.5)
    # 带噪信号 (红线，稍微透明一点)
    plt.plot(noisy_eeg[idx], label='Noisy Input', color='red', alpha=0.4, linewidth=1)
    # 去噪后信号 (绿线)
    plt.plot(denoised_eeg[idx], label='Denoised Output (Ours)', color='green', linewidth=1.5)
    
    plt.title(f'Sample {idx} Denoising Result', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Waveform_Comparison.png', dpi=300)
print("✅ 波形对比图已保存为 Waveform_Comparison.png")
plt.show()