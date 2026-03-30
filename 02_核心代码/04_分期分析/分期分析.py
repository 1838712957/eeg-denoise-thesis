import os
import glob
import numpy as np
import tensorflow as tf
import mne
import yasa
import pandas as pd
import warnings
import sys
from tensorflow.keras import layers, models, Sequential
from sklearn.metrics import accuracy_score
from pathlib import Path

# ================= 0. 配置 =================
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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

# 模型路径
model_path = pick_existing_path(
    os.path.join(base_dir, "03_训练模型", "denoise_model_v2_final.h5"),
    os.path.join(base_dir, "denoise_model_v2_final.h5"),
) 
TARGET_SFREQ = 256

# ================= 1. 网络结构定义 (必须与 Colab 训练时完全一致) =================
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

# ⚠️ 关键补充：手动构建模型结构
def build_model():
    inp = layers.Input(shape=(512, 1))
    x = layers.Conv1D(32, 7, padding='same')(inp)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = BasicBlockall(use_se=True)(x) # 注意：这里要和训练时一致
    out = layers.Conv1D(1, 7, padding='same')(x)
    return models.Model(inp, out)

# ================= 2. 核心加载函数 (只读权重) =================
def load_weights_safely(path):
    print(f"📂 正在初始化模型结构...")
    model = build_model() # 1. 先在本地造一个空壳
    
    print(f"⚖️ 正在注入权重: {os.path.basename(path)}")
    try:
        # 2. 只读取权重，忽略版本不兼容的配置信息
        model.load_weights(path)
        print("✅ 权重加载成功！")
        return model
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        print("请检查网络结构定义是否与 Colab 训练脚本完全一致。")
        exit()

# ================= 3. YASA 工具 =================
def run_yasa(data, sfreq):
    info = mne.create_info(['EEG'], sfreq, ['eeg'])
    raw_tmp = mne.io.RawArray(data.reshape(1, -1), info, verbose=False)
    sls = yasa.SleepStaging(raw_tmp, eeg_name="EEG")
    hypno_pred = sls.predict()
    return yasa.hypno_str_to_int(hypno_pred)

# ================= 4. 主程序 =================
if not os.path.exists(model_path):
    print(f"❌ 找不到模型文件: {model_path}")
    exit()

# 使用新方法加载
model = load_weights_safely(model_path)

raw_files = glob.glob(os.path.join(raw_dir, "*PSG.edf"))
STAGE_MAPPING = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2, 'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4, 'Movement time': -1, 'Sleep stage ?': -1}
STAGE_NAMES = {0: "Wake (清醒)", 1: "N1 (浅睡)", 2: "N2 (中睡)", 3: "N3 (深睡)", 4: "REM (做梦)"}

print(f"\n🚀 开始 V2.0 模型效果分析...")
print(f"{'受试者':<10} | {'阶段':<12} | {'原始准确率':<10} | {'V2.0 准确率':<12} | {'变化':<10}")
print("-" * 75)

for raw_path in raw_files:
    fname = os.path.basename(raw_path)
    prefix = fname[:6]
    hypno_files = glob.glob(os.path.join(raw_dir, f"{prefix}*Hypnogram.edf"))
    if not hypno_files: continue
    
    try:
        # 1. 读数据
        raw = mne.io.read_raw_edf(raw_path, preload=True, verbose=False)
        if raw.info['sfreq'] != TARGET_SFREQ: raw.resample(TARGET_SFREQ)
        data_raw = raw.get_data(picks=[0])[0]
        
        # 2. 预测
        slen = 512
        n_seg = len(data_raw) // slen
        batch_in = []
        scales = []
        for i in range(n_seg):
            seg = data_raw[i*slen : (i+1)*slen]
            std = np.std(seg) if np.std(seg)!=0 else 1.0
            batch_in.append(seg/std); scales.append(std)
            
        p = model.predict(np.array(batch_in).reshape(-1, slen, 1), batch_size=256, verbose=0)
        data_clean = np.array([p[i].flatten() * scales[i] for i in range(n_seg)]).flatten()
        
        # 3. YASA
        L = len(data_clean)
        h_raw = run_yasa(data_raw[:L], TARGET_SFREQ)
        h_clean = run_yasa(data_clean, TARGET_SFREQ)
        
        # 4. 真值
        annot = mne.read_annotations(hypno_files[0])
        raw.set_annotations(annot, emit_warning=False)
        events, _ = mne.events_from_annotations(raw, event_id=STAGE_MAPPING, chunk_duration=30., verbose=False)
        h_true = events[:, 2]
        
        # 5. 统计
        L_final = min(len(h_true), len(h_raw))
        for stage_val in [0, 1, 2, 3, 4]:
            mask = (h_true[:L_final] == stage_val)
            if np.sum(mask) < 10: continue
            
            acc_raw = accuracy_score(h_true[:L_final][mask], h_raw[:L_final][mask]) * 100
            acc_clean = accuracy_score(h_true[:L_final][mask], h_clean[:L_final][mask]) * 100
            diff = acc_clean - acc_raw
            
            diff_str = f"{diff:+.1f}%"
            if diff > 0: diff_str = f"🔺 {diff_str}"
            elif diff < -5: diff_str = f"🔻 {diff_str}"
            else: diff_str = f"🔸 {diff_str}"
            
            print(f"{prefix:<10} | {STAGE_NAMES[stage_val]:<12} | {acc_raw:<10.1f}% | {acc_clean:<12.1f}% | {diff_str}")
            
    except Exception as e:
        print(f"❌ {prefix} 出错: {e}")