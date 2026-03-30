import os
import glob
import numpy as np
import tensorflow as tf
import mne
import csv
import time
import h5py
import shutil
import json
import sys
import yasa
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from tensorflow.keras import layers, models, Sequential
from sklearn.metrics import accuracy_score
from pathlib import Path

# ================= 0. 屏蔽干扰信息 =================
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 基础配置 =================
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
output_csv_path = os.path.join(base_dir, "Final_Staging_Comparison.csv") 
output_img_dir = os.path.join(base_dir, "Staging_Plots")

if not os.path.exists(output_img_dir): os.makedirs(output_img_dir)
TARGET_SFREQ = 256

# ================= 2. 网络结构定义 (缺一不可) =================
class SEBlock(layers.Layer):
    def __init__(self, channels=32, reduction=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.channels, self.reduction = channels, reduction
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
        self.kernelsize, self.stride, self.use_se, self.se_reduction = kernelsize, stride, use_se, se_reduction
        self.bblock = Sequential([
            layers.Conv1D(32, kernelsize, strides=stride, padding="same"),
            layers.BatchNormalization(), layers.ReLU(),
            layers.Conv1D(16, kernelsize, strides=1, padding="same"),
            layers.BatchNormalization(), layers.ReLU(),
            layers.Conv1D(32, kernelsize, strides=1, padding="same"),
            layers.BatchNormalization(), layers.ReLU()
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

# === 之前漏掉的类，现在补上了！ ===
class BasicBlockall(layers.Layer):
    def __init__(self, stride=1, use_se=False, se_reduction=16, **kwargs):
        super(BasicBlockall, self).__init__(**kwargs)
        self.stride, self.use_se, self.se_reduction = stride, use_se, se_reduction
        self.bblock3 = Sequential([Res_BasicBlock(3, use_se=use_se), Res_BasicBlock(3, use_se=use_se)])
        self.bblock5 = Sequential([Res_BasicBlock(5, use_se=use_se), Res_BasicBlock(5, use_se=use_se)])
        self.bblock7 = Sequential([Res_BasicBlock(7, use_se=use_se), Res_BasicBlock(7, use_se=use_se)])
    def call(self, inputs):
        return tf.concat([self.bblock3(inputs), self.bblock5(inputs), self.bblock7(inputs)], axis=-1)
    def get_config(self):
        config = super(BasicBlockall, self).get_config()
        config.update({"stride": self.stride, "use_se": self.use_se, "se_reduction": self.se_reduction})
        return config

# ================= 3. 模型修复模块 =================
def clean_config_dict(config):
    if isinstance(config, dict):
        if 'class_name' in config and config['class_name'] == 'BatchNormalization':
            if 'config' in config and 'synchronized' in config['config']:
                del config['config']['synchronized']
        for key, value in config.items():
            clean_config_dict(value)
    elif isinstance(config, list):
        for item in config:
            clean_config_dict(item)

def load_model_with_fix(original_path):
    temp_path = original_path.replace(".h5", "_vis_fixed.h5")
    shutil.copyfile(original_path, temp_path)
    try:
        with h5py.File(temp_path, 'r+') as f:
            if 'model_config' in f.attrs:
                config_str = f.attrs['model_config']
                if isinstance(config_str, bytes): config_str = config_str.decode('utf-8')
                fixed_str = config_str.replace('"batch_shape":', '"batch_input_shape":')
                try:
                    config_json = json.loads(fixed_str)
                    clean_config_dict(config_json)
                    f.attrs['model_config'] = json.dumps(config_json).encode('utf-8')
                except: pass
    except: pass

    class FakeDTypePolicy:
        def __init__(self, *args, **kwargs):
            self._name = "float32"
            self._compute_dtype = "float32"
            self._variable_dtype = "float32"
        @property
        def name(self): return self._name
        @property
        def compute_dtype(self): return self._compute_dtype
        @property
        def variable_dtype(self): return self._variable_dtype
        @classmethod
        def from_config(cls, config): return cls()
        def get_config(self): return {"name": "float32"}

    try:
        # 现在这里一定能找到 BasicBlockall 了
        custom_objects = {'SEBlock': SEBlock, 'Res_BasicBlock': Res_BasicBlock, 
                          'BasicBlockall': BasicBlockall, 'DTypePolicy': FakeDTypePolicy}
        model = tf.keras.models.load_model(temp_path, custom_objects=custom_objects, compile=False)
        try: os.remove(temp_path)
        except: pass
        return model
    except Exception as e:
        print(f"❌ 模型加载失败: {e}"); exit()

# ================= 4. YASA 分期函数 (含翻译机) =================
def run_yasa_staging(raw_data_array, sfreq):
    info = mne.create_info(ch_names=['EEG'], sfreq=sfreq, ch_types=['eeg'])
    raw_tmp = mne.io.RawArray(raw_data_array.reshape(1, -1), info, verbose=False)
    sls = yasa.SleepStaging(raw_tmp, eeg_name="EEG")
    
    # 预测并转数字
    hypno_pred = sls.predict()
    hypno_int = yasa.hypno_str_to_int(hypno_pred)
    return hypno_int

# ================= 5. 画图函数 =================
def plot_staging_comparison(hypno_true, hypno_raw, hypno_asr, hypno_ours, acc_dict, sid, save_path):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    stages = ['Wake', 'N1', 'N2', 'N3', 'REM']
    time_axis = np.arange(len(hypno_true)) * 30 / 60 

    def plot_hypno(ax, hypno, title, color):
        if hypno is None: return
        hypno = np.array(hypno, dtype=int)
        ax.step(time_axis, hypno, where='post', color=color, linewidth=1.5)
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(stages)
        ax.invert_yaxis()
        ax.set_title(title, loc='left', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plot_hypno(axes[0], hypno_true, f"Ground Truth (Doctor)", 'black')
    plot_hypno(axes[1], hypno_raw, f"Raw Data (Acc: {acc_dict['Raw']}%)", 'gray')
    
    title_asr = f"ASR Cleaned (Acc: {acc_dict['ASR']}%)" if hypno_asr is not None else "ASR Cleaned (No File)"
    plot_hypno(axes[2], hypno_asr if hypno_asr is not None else np.zeros_like(hypno_true), title_asr, 'blue')
    
    plot_hypno(axes[3], hypno_ours, f"Ours ResCNN (Acc: {acc_dict['Ours']}%)", 'red')
    
    axes[3].set_xlabel("Time (minutes)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# ================= 6. 主程序 =================
if not os.path.exists(model_path): print("❌ 找不到模型！"); exit()
model = load_model_with_fix(model_path)
print("✅ 模型就绪！开始生成最终对比图...")

raw_files = glob.glob(os.path.join(raw_dir, "*PSG.edf"))
results = []
STAGE_MAPPING = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2, 'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4, 'Movement time': -1, 'Sleep stage ?': -1}

for idx, raw_path in enumerate(raw_files):
    fname = os.path.basename(raw_path)
    prefix = fname[:6] 
    print(f"\n[{idx+1}/{len(raw_files)}] 正在处理: {prefix} ...")
    
    # 智能寻找标签
    hypno_files = glob.glob(os.path.join(raw_dir, f"{prefix}*Hypnogram.edf"))
    if not hypno_files: print("   ⚠️ 缺标签"); continue
    hypno_path = hypno_files[0]
    
    asr_path = os.path.join(asr_dir, f"{os.path.splitext(fname)[0]}_fixed_clean.set")
    
    try:
        # 1. 准备数据
        raw = mne.io.read_raw_edf(raw_path, preload=True, verbose=False)
        if raw.info['sfreq'] != TARGET_SFREQ: raw.resample(TARGET_SFREQ)
        data_raw = raw.get_data(picks=[0])[0]
        
        data_asr = None
        if os.path.exists(asr_path):
            try:
                raw_a = mne.io.read_raw_eeglab(asr_path, preload=True, verbose=False)
                if raw_a.info['sfreq'] != TARGET_SFREQ: raw_a.resample(TARGET_SFREQ)
                data_asr = raw_a.get_data(picks=[0])[0]
            except: pass

        # 2. 模型去噪
        print("   ⚡ 模型去噪中...", end="")
        slen = 512
        n_seg = len(data_raw) // slen
        batch_in, scales = [], []
        for i in range(n_seg):
            seg = data_raw[i*slen : (i+1)*slen]
            std = np.std(seg) if np.std(seg)!=0 else 1.0
            batch_in.append(seg/std); scales.append(std)
        p = model.predict(np.array(batch_in).reshape(-1, slen, 1), batch_size=256, verbose=0)
        data_ours = np.array([p[i].flatten() * scales[i] for i in range(n_seg)]).flatten()
        print(" [完成]")
        
        # 3. YASA 分期
        print("   🧠 AI 分期中...", end="")
        L_final = len(data_ours)
        h_raw = run_yasa_staging(data_raw[:L_final], TARGET_SFREQ)
        h_ours = run_yasa_staging(data_ours, TARGET_SFREQ)
        h_asr = run_yasa_staging(data_asr[:L_final], TARGET_SFREQ) if data_asr is not None else None
        print(" [完成]")
        
        # 4. 真实标签提取
        annot = mne.read_annotations(hypno_path)
        raw.set_annotations(annot, emit_warning=False)
        events, _ = mne.events_from_annotations(raw, event_id=STAGE_MAPPING, chunk_duration=30., verbose=False)
        h_true = events[:, 2]
        
        # 5. 对齐与计算
        L = min(len(h_true), len(h_raw), len(h_ours))
        if h_asr is not None: L = min(L, len(h_asr))
        
        valid = h_true[:L] != -1
        acc_dict = {'Raw': 0, 'ASR': 0, 'Ours': 0}
        
        if np.sum(valid) > 0:
            acc_dict['Raw'] = round(accuracy_score(h_true[:L][valid], h_raw[:L][valid]) * 100, 2)
            acc_dict['Ours'] = round(accuracy_score(h_true[:L][valid], h_ours[:L][valid]) * 100, 2)
            if h_asr is not None:
                acc_dict['ASR'] = round(accuracy_score(h_true[:L][valid], h_asr[:L][valid]) * 100, 2)
        
        print(f"   🏆 准确率: Raw={acc_dict['Raw']}% -> Ours={acc_dict['Ours']}%")
        
        # 6. 画图
        img_path = os.path.join(output_img_dir, f"Staging_{prefix}.png")
        plot_staging_comparison(h_true[:L], h_raw[:L], h_asr[:L] if h_asr is not None else None, h_ours[:L], acc_dict, prefix, img_path)
        
        results.append({'Subject': prefix, **acc_dict})

    except Exception as e:
        print(f"   ❌ 失败: {e}")
        import traceback; traceback.print_exc()

if results:
    pd.DataFrame(results).to_csv(output_csv_path, index=False)
    print(f"\n✅ 完成！请打开 {output_img_dir} 查看图片")