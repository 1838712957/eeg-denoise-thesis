import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import mne
from tensorflow.keras import layers, models, Sequential
from pathlib import Path

# ================= 1. 路径配置 =================
CORE_CODE_DIR = Path(__file__).resolve().parents[2] / "02_核心代码"
if str(CORE_CODE_DIR) not in sys.path:
    sys.path.append(str(CORE_CODE_DIR))
from bootstrap_paths import PROJECT_ROOT
from path_utils import pick_existing_path

base_dir = str(PROJECT_ROOT)
eeglab_dir = str(PROJECT_ROOT / "08_第三方工具" / "eeglab2025.1.0")

# 原始数据 & 标签
edf_path = os.path.join(base_dir, "SC4002E0-PSG.edf")
hypno_path = os.path.join(base_dir, "SC4002EC-Hypnogram.edf") # ✅ 加上了标签文件

# ASR 结果
asr_path = os.path.join(eeglab_dir, "sc4002e_fixed_clean.set")
model_path = pick_existing_path(
    os.path.join(base_dir, "03_训练模型", "denoise_model.h5"),
    os.path.join(base_dir, "denoise_model.h5"),
)

TARGET_SFREQ = 256 

# ================= 2. 网络结构 (保持不变) =================
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

# ================= 3. 读取数据 & 标签 =================
print(f"📂 读取原始数据: {os.path.basename(edf_path)}")
raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

# ✅ 核心步骤：读取 Hypnogram 标签
print(f"🏷️ 读取睡眠分期标签: {os.path.basename(hypno_path)}")
annot = mne.read_annotations(hypno_path)
raw.set_annotations(annot, emit_warning=False)

# 挑选通道 & 重采样
eeg_picks = mne.pick_channels(raw.info['ch_names'], include=['EEG Fpz-Cz'])
if len(eeg_picks) == 0: eeg_picks = [0]
print("🔄 重采样至 256Hz...")
raw.resample(TARGET_SFREQ) # 这一步会自动处理 annotations 的时间对应关系

# 读取 ASR 数据 (用于对比)
print(f"📂 读取 ASR 数据: {os.path.basename(asr_path)}")
raw_asr = mne.io.read_raw_eeglab(asr_path, preload=True, verbose=False)
if raw_asr.info['sfreq'] != TARGET_SFREQ: raw_asr.resample(TARGET_SFREQ)

# ================= 4. 靶向提取 N2 期数据 =================
print("🎯 正在搜索 N2 (Sleep stage 2) 片段...")
events, event_id = mne.events_from_annotations(raw, event_id=None, chunk_duration=30.) # Sleep-EDF 30s一段

# 查找 N2 的编号 (通常是 'Sleep stage 2')
n2_id = None
for key in event_id:
    if "Sleep stage 2" in key:
        n2_id = event_id[key]
        break

if n2_id is None:
    print("⚠️ 未找到 N2 标签，尝试随机抽取...")
    target_indices = np.random.choice(len(events), 5)
else:
    # 找到所有 N2 事件的索引
    n2_indices = np.where(events[:, 2] == n2_id)[0]
    print(f"✅ 找到了 {len(n2_indices)} 个 N2 片段！")
    # 随机挑 3 个 N2 片段来画图
    target_indices = np.random.choice(n2_indices, 3, replace=False)

# ================= 5. 准备模型输入 & 绘图 =================
# 获取对应时间段的数据
data_raw_full = raw.get_data(picks=eeg_picks)[0]
data_asr_full = raw_asr.get_data(picks=[0])[0] # ASR 取第1个通道

# 加载模型
print("🧠 加载模型...")
custom_objects = {'SEBlock': SEBlock, 'Res_BasicBlock': Res_BasicBlock, 'BasicBlockall': BasicBlockall}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

plt.figure(figsize=(15, 12))

for i, idx in enumerate(target_indices):
    # 这里的 events[idx, 0] 是采样点位置
    onset = events[idx, 0]
    # 我们截取 512 个点 (2秒) 进行展示，或者 30秒整个片段
    # 为了模型推理，我们切 512 点的一小段放在 N2 片段中间
    start_sample = onset + 256 * 10 # 取 30s 片段的中间第 10 秒开始
    end_sample = start_sample + 512
    
    # 防止越界
    if end_sample > len(data_raw_full): continue

    # 1. 准备数据
    seg_raw = data_raw_full[start_sample:end_sample]
    seg_asr = data_asr_full[start_sample:end_sample]
    
    # 2. 模型推理 (标准化 -> 预测 -> 还原)
    std_val = np.std(seg_raw)
    if std_val == 0: std_val = 1
    input_tensor = (seg_raw / std_val).reshape(1, 512, 1)
    
    pred = model.predict(input_tensor, verbose=0)
    seg_ai = np.squeeze(pred) * std_val # 还原幅度
    
    # 3. 绘图
    plt.subplot(3, 1, i+1)
    
    # 转微伏
    t = np.arange(512) / TARGET_SFREQ
    plt.plot(t, seg_raw * 1e6, color='gray', alpha=0.5, label='Original (Raw)')
    plt.plot(t, seg_asr * 1e6, color='blue', linestyle='--', alpha=0.8, label='ASR')
    plt.plot(t, seg_ai * 1e6, color='red', linewidth=2, label='Ours')
    
    plt.title(f'N2 Stage Analysis - Sample {i+1} (Look for Spindles!)')
    plt.legend(loc="upper right")
    plt.ylabel('Amplitude (uV)')
    plt.xlabel('Time (s)')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print("✅ N2 阶段验证完成！请观察红线是否保留了那些快速抖动的纺锤波。")