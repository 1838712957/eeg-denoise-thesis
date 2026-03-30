import os
import glob
import numpy as np
import tensorflow as tf
import mne
import csv
import time
import sys
from tensorflow.keras import layers, models, Sequential
from pathlib import Path

# ================= 1. 路径配置 =================
CORE_CODE_DIR = Path(__file__).resolve().parents[2] / "02_核心代码"
if str(CORE_CODE_DIR) not in sys.path:
    sys.path.append(str(CORE_CODE_DIR))
from bootstrap_paths import PROJECT_ROOT
from path_utils import pick_existing_path

base_dir = str(PROJECT_ROOT)
raw_dir = pick_existing_path(
    os.path.join(base_dir, "04_原始数据", "Raw_edf"),
    os.path.join(base_dir, "Raw_edf"),
)
asr_dir = pick_existing_path(
    os.path.join(base_dir, "05_模型输出", "ASR_result"),
    os.path.join(base_dir, "ASR_result"),
)
model_path = pick_existing_path(
    os.path.join(base_dir, "03_训练模型", "denoise_model.h5"),
    os.path.join(base_dir, "denoise_model.h5"),
)
output_csv_path = os.path.join(base_dir, "Quantitative_Results.csv") 

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

# ================= 3. 加载模型 =================
print("🧠 加载 AI 模型...")
custom_objects = {'SEBlock': SEBlock, 'Res_BasicBlock': Res_BasicBlock, 'BasicBlockall': BasicBlockall}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# ================= 4. 定义评估指标函数 =================
def calculate_metrics(raw, cleaned):
    if np.std(raw) == 0 or np.std(cleaned) == 0: return 0, 0
    # NRR: 能量减少比
    var_raw = np.var(raw)
    var_clean = np.var(cleaned)
    nrr = (var_raw - var_clean) / var_raw * 100
    # CC: 相关系数
    cc = np.corrcoef(raw, cleaned)[0, 1]
    return nrr, cc

# ================= 5. 批量计算循环 =================
raw_files = glob.glob(os.path.join(raw_dir, "*PSG.edf"))
print(f"📂 发现 {len(raw_files)} 个受试者，开始计算指标...")

results = []
totals = {"NRR_ASR": 0, "NRR_Ours": 0, "CC_ASR": 0, "CC_Ours": 0, "Time": 0}

for idx, raw_path in enumerate(raw_files):
    filename = os.path.basename(raw_path)
    subject_id = filename.split('-')[0]
    
    print(f"\n[{idx+1}/{len(raw_files)}] 计算中: {subject_id} ...", end="")
    
    name_without_ext = os.path.splitext(filename)[0]
    asr_filename = f"{name_without_ext}_fixed_clean.set"
    asr_path = os.path.join(asr_dir, asr_filename)
    
    if not os.path.exists(asr_path):
        print(" [跳过: 缺 ASR 文件]")
        continue
        
    try:
        # A. 读取
        raw = mne.io.read_raw_edf(raw_path, preload=True, verbose=False)
        if raw.info['sfreq'] != TARGET_SFREQ: raw.resample(TARGET_SFREQ)
        eeg_picks = mne.pick_channels(raw.info['ch_names'], include=['EEG Fpz-Cz'])
        if len(eeg_picks) == 0: eeg_picks = [0]
        data_raw = raw.get_data(picks=eeg_picks)[0]
        
        raw_asr = mne.io.read_raw_eeglab(asr_path, preload=True, verbose=False)
        if raw_asr.info['sfreq'] != TARGET_SFREQ: raw_asr.resample(TARGET_SFREQ)
        data_asr = raw_asr.get_data(picks=[0])[0]
        
        # 对齐
        min_len = min(len(data_raw), len(data_asr))
        data_raw = data_raw[:min_len]
        data_asr = data_asr[:min_len]
        
        # B. 推理 (前2000段)
        segment_len = 512
        n_segments = min(len(data_raw) // segment_len, 2000)
        
        input_data = []
        scales = []
        for i in range(n_segments):
            seg = data_raw[i*segment_len : (i+1)*segment_len]
            std = np.std(seg)
            if std == 0: std = 1
            input_data.append(seg / std)
            scales.append(std)
            
        input_tensor = np.array(input_data).reshape(-1, 512, 1)
        
        t0 = time.time()
        preds = model.predict(input_tensor, batch_size=256, verbose=0)
        t_inference = time.time() - t0
        
        preds = np.squeeze(preds)
        
        # 还原
        data_ai_full = []
        data_raw_full = []
        data_asr_full = []
        
        for i in range(n_segments):
            data_ai_full.extend(preds[i] * scales[i])
            start = i*segment_len
            end = (i+1)*segment_len
            data_raw_full.extend(data_raw[start:end])
            data_asr_full.extend(data_asr[start:end])
            
        flat_raw = np.array(data_raw_full)
        flat_asr = np.array(data_asr_full)
        flat_ai  = np.array(data_ai_full)
        
        # C. 指标
        nrr_asr, cc_asr = calculate_metrics(flat_raw, flat_asr)
        nrr_ours, cc_ours = calculate_metrics(flat_raw, flat_ai)
        
        print(f" [完成]")
        
        row = {
            "Subject": subject_id,
            "NRR_ASR(%)": round(nrr_asr, 2),
            "NRR_Ours(%)": round(nrr_ours, 2),
            "CC_Raw_ASR": round(cc_asr, 4),
            "CC_Raw_Ours": round(cc_ours, 4),
            "Time(s)": round(t_inference, 2)
        }
        results.append(row)
        
        totals["NRR_ASR"] += nrr_asr
        totals["NRR_Ours"] += nrr_ours
        totals["CC_ASR"] += cc_asr
        totals["CC_Ours"] += cc_ours
        totals["Time"] += t_inference
        
    except Exception as e:
        print(f" [失败: {e}]")

# ================= 6. 打印并保存表格 =================
if len(results) > 0:
    count = len(results)
    avg_row = {
        "Subject": "AVERAGE",
        "NRR_ASR(%)": round(totals["NRR_ASR"]/count, 2),
        "NRR_Ours(%)": round(totals["NRR_Ours"]/count, 2),
        "CC_Raw_ASR": round(totals["CC_ASR"]/count, 4),
        "CC_Raw_Ours": round(totals["CC_Ours"]/count, 4),
        "Time(s)": round(totals["Time"]/count, 2)
    }
    results.append(avg_row)
    
    # 打印表格
    print("\n" + "="*90)
    print(f"{'Subject':<10} | {'NRR_ASR(%)':<12} | {'NRR_Ours(%)':<12} | {'CC_ASR':<10} | {'CC_Ours':<10} | {'Time(s)':<8}")
    print("-" * 90)
    for r in results:
        # ✅ 这里修复了闭合括号的问题
        print(f"{r['Subject']:<10} | {r['NRR_ASR(%)']:<12} | {r['NRR_Ours(%)']:<12} | {r['CC_Raw_ASR']:<10} | {r['CC_Raw_Ours']:<10} | {r['Time(s)']:<8}")
    print("="*90)
    
    # 保存 CSV
    headers = ["Subject", "NRR_ASR(%)", "NRR_Ours(%)", "CC_Raw_ASR", "CC_Raw_Ours", "Time(s)"]
    try:
        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✅ 表格已保存至: {output_csv_path}")
    except Exception as e:
        print(f"❌ 保存 CSV 失败: {e}")

else:
    print("❌ 没有生成任何结果，请检查 ASR 文件是否匹配。")