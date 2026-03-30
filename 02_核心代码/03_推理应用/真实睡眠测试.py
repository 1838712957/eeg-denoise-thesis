import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import mne
from tensorflow.keras import layers, models, Sequential
from pathlib import Path

# ================= 1. 路径配置 (已修正：文件在不同文件夹) =================

# ✅ 1. 原始数据 (在外面这一层)
CORE_CODE_DIR = Path(__file__).resolve().parents[2] / "02_核心代码"
if str(CORE_CODE_DIR) not in sys.path:
    sys.path.append(str(CORE_CODE_DIR))
from bootstrap_paths import PROJECT_ROOT
from path_utils import pick_existing_path

edf_path = str(PROJECT_ROOT / "SC4002E0-PSG.edf")

# ✅ 2. ASR 结果 (在 eeglab 文件夹里，注意是小写文件名)
asr_path = str(PROJECT_ROOT / "08_第三方工具" / "eeglab2025.1.0" / "sc4002e_fixed_clean.set")

# ✅ 3. 你的 AI 模型
model_path = pick_existing_path(
    str(PROJECT_ROOT / "03_训练模型" / "denoise_model.h5"),
    str(PROJECT_ROOT / "denoise_model.h5"),
)

# 目标采样率
TARGET_SFREQ = 256 

# ================= 2. 网络结构定义 (保持不变) =================
class SEBlock(layers.Layer):
    def __init__(self, channels=32, reduction=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
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
        self.kernelsize = kernelsize
        self.stride = stride
        self.use_se = use_se
        self.se_reduction = se_reduction
        self.bblock = Sequential([
            layers.Conv1D(32, kernelsize, strides=stride, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv1D(16, kernelsize, strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv1D(32, kernelsize, strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU()
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
        self.stride = stride
        self.use_se = use_se
        self.se_reduction = se_reduction
        self.bblock3 = Sequential([Res_BasicBlock(3, use_se=use_se), Res_BasicBlock(3, use_se=use_se)])
        self.bblock5 = Sequential([Res_BasicBlock(5, use_se=use_se), Res_BasicBlock(5, use_se=use_se)])
        self.bblock7 = Sequential([Res_BasicBlock(7, use_se=use_se), Res_BasicBlock(7, use_se=use_se)])
    def call(self, inputs):
        return tf.concat([self.bblock3(inputs), self.bblock5(inputs), self.bblock7(inputs)], axis=-1)
    def get_config(self):
        config = super(BasicBlockall, self).get_config()
        config.update({"stride": self.stride, "use_se": self.use_se, "se_reduction": self.se_reduction})
        return config

# ================= 3. 数据读取与对齐 =================
print("="*50)
print(f"📂 1. 读取原始数据: {os.path.basename(edf_path)}")

# 1. 读原始数据
if not os.path.exists(edf_path):
    print(f"❌ 错误：找不到文件 {edf_path}")
    print(f"提示：请确认文件是否在 '{PROJECT_ROOT}' 根目录下")
    exit()
    
raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
eeg_picks = mne.pick_channels(raw.info['ch_names'], include=['EEG Fpz-Cz'])
if len(eeg_picks) == 0: 
    print("⚠️ 没找到 'EEG Fpz-Cz'，默认使用第 1 个通道")
    eeg_picks = [0]

print("   -> 重采样至 256Hz...")
raw.resample(TARGET_SFREQ)
data_raw = raw.get_data(picks=eeg_picks)[0]


# 2. 读 ASR 结果
print(f"📂 2. 读取 ASR 结果: {os.path.basename(asr_path)}")
if not os.path.exists(asr_path):
    print(f"❌ 错误：找不到文件 {asr_path}")
    print("提示：请确认文件是否在 'eeglab2025.1.0' 文件夹里")
    exit()

try:
    raw_asr = mne.io.read_raw_eeglab(asr_path, preload=True, verbose=False)
    
    # 检查采样率
    if raw_asr.info['sfreq'] != TARGET_SFREQ:
        print(f"   -> ASR 采样率 ({raw_asr.info['sfreq']}Hz) 不匹配，正在重采样...")
        raw_asr.resample(TARGET_SFREQ)
    
    # 取第 1 个通道
    print(f"   -> 提取 ASR 数据 (通道: {raw_asr.info['ch_names'][0]})...")
    data_asr = raw_asr.get_data(picks=[0])[0]

    # 对齐长度
    min_len = min(len(data_raw), len(data_asr))
    data_raw = data_raw[:min_len]
    data_asr = data_asr[:min_len]
    print(f"   -> 数据对齐完成，共 {min_len} 个采样点")

except Exception as e:
    print(f"❌ 读取 .set 文件失败: {e}")
    exit()

# ================= 4. AI 模型推理 =================
print("="*50)
print("🧠 3. AI 模型推理中...")

segment_len = 512
# 取前 3000 段
n_segments = min(len(data_raw) // segment_len, 3000)

input_data_model = [] 
scales = []           

for i in range(n_segments):
    seg = data_raw[i*segment_len : (i+1)*segment_len]
    std_val = np.std(seg)
    if std_val == 0: std_val = 1
    input_data_model.append(seg / std_val) 
    scales.append(std_val)

input_data_model = np.array(input_data_model).reshape(-1, 512, 1)

custom_objects = {'SEBlock': SEBlock, 'Res_BasicBlock': Res_BasicBlock, 'BasicBlockall': BasicBlockall}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

denoised_ai = model.predict(input_data_model, batch_size=256, verbose=0)
denoised_ai = np.squeeze(denoised_ai)

# ================= 5. 绘图对比 =================
print("="*50)
print("🎨 4. 生成 SC4002 对比图...")

# 随机挑选 3 个片段
indices = np.random.choice(len(input_data_model), 3, replace=False)

plt.figure(figsize=(15, 12))

for i, idx in enumerate(indices):
    plt.subplot(3, 1, i+1)
    
    start_idx = idx * segment_len
    end_idx = (idx + 1) * segment_len
    
    # 转微伏
    original_uv = data_raw[start_idx:end_idx] * 1e6 
    asr_uv = data_asr[start_idx:end_idx] * 1e6
    ai_uv = denoised_ai[idx] * scales[idx] * 1e6
    
    # 绘图
    plt.plot(original_uv, label='Original (Raw)', color='gray', alpha=0.4, linewidth=1)
    plt.plot(asr_uv, label='ASR (Traditional)', color='blue', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.plot(ai_uv, label='Ours (Attention-ResCNN)', color='red', linewidth=2)
    
    plt.title(f'Subject SC4002 Comparison - Segment {idx}', fontsize=12)
    plt.legend(loc="upper right")
    plt.ylabel('Amplitude ($\mu V$)')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('SC4002_Comparison.png', dpi=300)
plt.show()

print("✅ SC4002 验证完成！")