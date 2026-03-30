import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import mne
# import pandas as pd # 已移除，无需安装
from tensorflow.keras import layers, models, Sequential
from pathlib import Path

# ================= 1. 路径配置 (请确保这些路径真实存在) =================
CORE_CODE_DIR = Path(__file__).resolve().parents[2] / "02_核心代码"
if str(CORE_CODE_DIR) not in sys.path:
    sys.path.append(str(CORE_CODE_DIR))
from bootstrap_paths import PROJECT_ROOT
from path_utils import pick_existing_path

base_dir = str(PROJECT_ROOT)
raw_dir = pick_existing_path(
    os.path.join(base_dir, "04_原始数据", "Raw_edf"),
    os.path.join(base_dir, "Raw_edf"),
)       # 原始数据
asr_dir = pick_existing_path(
    os.path.join(base_dir, "05_模型输出", "ASR_result"),
    os.path.join(base_dir, "ASR_result"),
)    # Matlab ASR 结果
model_path = pick_existing_path(
    os.path.join(base_dir, "03_训练模型", "denoise_model.h5"),
    os.path.join(base_dir, "denoise_model.h5"),
)
output_img_dir = os.path.join(base_dir, "Comparison_Images") # 图片保存位置

if not os.path.exists(output_img_dir):
    os.makedirs(output_img_dir)

TARGET_SFREQ = 256 

# ================= 2. 网络结构定义 (务必保留) =================
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

# ================= 3. 加载模型 =================
print("🧠 加载 AI 模型...")
custom_objects = {'SEBlock': SEBlock, 'Res_BasicBlock': Res_BasicBlock, 'BasicBlockall': BasicBlockall}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# ================= 4. 批量处理逻辑 =================
# 只找 *PSG.edf，自动跳过 Hypnogram
raw_files = glob.glob(os.path.join(raw_dir, "*PSG.edf"))
print(f"📂 在 Raw_edf 中发现 {len(raw_files)} 个受试者数据")

report_list = []

for idx, raw_path in enumerate(raw_files):
    filename = os.path.basename(raw_path) # 例如: SC4001E0-PSG.edf
    
    # 提取受试者ID (用于显示)
    subject_id = filename.split('-')[0] # SC4001E0
    
    print(f"\n[{idx+1}/{len(raw_files)}] 正在匹配受试者: {subject_id} ({filename}) ...")
    
    # --- 🔥 核心修复点：文件名构造逻辑 🔥 ---
    # Matlab 生成的文件名逻辑是: [原始文件名(不带后缀)]_fixed_clean.set
    # 1. 去掉 .edf 后缀 -> SC4001E0-PSG
    name_without_ext = os.path.splitext(filename)[0] 
    # 2. 加上后缀 -> SC4001E0-PSG_fixed_clean.set
    asr_filename = f"{name_without_ext}_fixed_clean.set"
    
    asr_path = os.path.join(asr_dir, asr_filename)
    
    # 检查 ASR 文件是否存在
    if not os.path.exists(asr_path):
        print(f"   ⚠️ 跳过：未找到对应的 ASR 文件")
        print(f"      期待路径: {asr_path}")
        report_list.append({"Subject": subject_id, "Status": "Skipped (No ASR File)"})
        continue
        
    try:
        # --- A. 读取数据 ---
        # 读 Raw
        raw = mne.io.read_raw_edf(raw_path, preload=True, verbose=False)
        if raw.info['sfreq'] != TARGET_SFREQ: raw.resample(TARGET_SFREQ)
        eeg_picks = mne.pick_channels(raw.info['ch_names'], include=['EEG Fpz-Cz'])
        if len(eeg_picks) == 0: eeg_picks = [0]
        data_raw = raw.get_data(picks=eeg_picks)[0]
        
        # 读 ASR
        raw_asr = mne.io.read_raw_eeglab(asr_path, preload=True, verbose=False)
        if raw_asr.info['sfreq'] != TARGET_SFREQ: raw_asr.resample(TARGET_SFREQ)
        data_asr = raw_asr.get_data(picks=[0])[0]
        
        # 对齐长度
        min_len = min(len(data_raw), len(data_asr))
        data_raw = data_raw[:min_len]
        data_asr = data_asr[:min_len]
        
        # --- B. AI 模型推理 (只取前 2000 个片段以节省时间) ---
        segment_len = 512
        n_segments = min(len(data_raw) // segment_len, 2000)
        
        input_data = []
        scales = []
        
        for i in range(n_segments):
            seg = data_raw[i*segment_len : (i+1)*segment_len]
            std_val = np.std(seg)
            if std_val == 0: std_val = 1
            input_data.append(seg / std_val)
            scales.append(std_val)
            
        input_data = np.array(input_data).reshape(-1, 512, 1)
        
        # 推理
        preds = model.predict(input_data, batch_size=256, verbose=0)
        preds = np.squeeze(preds)
        
        # --- C. 计算简单指标 (寻找差异最大的片段画图) ---
        # 找方差最大的 3 个片段 (通常是最脏的片段)
        noisy_indices = np.argsort([np.var(x) for x in input_data])[-3:]
        
        # --- D. 画图并保存 ---
        plt.figure(figsize=(15, 10))
        for plot_i, seg_idx in enumerate(noisy_indices):
            plt.subplot(3, 1, plot_i+1)
            
            start = seg_idx * segment_len
            end = (seg_idx + 1) * segment_len
            
            # 还原微伏
            uV_raw = data_raw[start:end] * 1e6
            uV_asr = data_asr[start:end] * 1e6
            uV_ai  = preds[seg_idx] * scales[seg_idx] * 1e6
            
            plt.plot(uV_raw, color='gray', alpha=0.5, label='Raw')
            plt.plot(uV_asr, color='blue', linestyle='--', alpha=0.8, label='ASR')
            plt.plot(uV_ai,  color='red', linewidth=1.5, label='Ours')
            
            plt.title(f"Subject {subject_id} - High Noise Segment {seg_idx}")
            plt.ylabel("Amplitude (uV)")
            if plot_i == 0: plt.legend(loc="upper right")
            
        plt.tight_layout()
        save_img_path = os.path.join(output_img_dir, f"{subject_id}_Comparison.png")
        plt.savefig(save_img_path)
        plt.close() # 关闭画布
        
        print(f"   ✅ 对比图已保存: {save_img_path}")
        report_list.append({"Subject": subject_id, "Status": "Success"})
        
    except Exception as e:
        print(f"   ❌ 处理失败: {e}")
        report_list.append({"Subject": subject_id, "Status": "Failed"})

# ================= 5. 简单文字报告 =================
print("\n" + "="*50)
print(f"{'Subject':<20} | {'Status':<15}")
print("-" * 40)
for item in report_list:
    print(f"{item['Subject']:<20} | {item['Status']:<15}")
print("="*50)
print(f"图片已全部保存在: {output_img_dir}")