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
import warnings
from tensorflow.keras import layers, models, Sequential
from sklearn.metrics import accuracy_score
from pathlib import Path

# ================= 0. 屏蔽警告 =================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
# 屏蔽 sklearn 版本警告
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

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
model_path = pick_existing_path(
    os.path.join(base_dir, "03_训练模型", "denoise_model.h5"),
    os.path.join(base_dir, "denoise_model.h5"),
)
output_csv_path = os.path.join(base_dir, "Staging_Comparison_Result.csv") 

TARGET_SFREQ = 256

# ================= 2. 模型修复模块 (保持不变) =================
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
    temp_path = original_path.replace(".h5", "_temp_fixed.h5")
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
        custom_objects = {'SEBlock': SEBlock, 'Res_BasicBlock': Res_BasicBlock, 
                          'BasicBlockall': BasicBlockall, 'DTypePolicy': FakeDTypePolicy}
        model = tf.keras.models.load_model(temp_path, custom_objects=custom_objects, compile=False)
        try: os.remove(temp_path)
        except: pass
        return model
    except Exception as e:
        print(f"❌ 模型加载失败: {e}"); exit()

# ================= 3. 网络结构 (保持不变) =================
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

# ================= 4. YASA 分期函数 =================
def run_yasa_staging(raw_data_array, sfreq):
    # 构造临时 MNE Raw 对象
    info = mne.create_info(ch_names=['EEG'], sfreq=sfreq, ch_types=['eeg'])
    raw_tmp = mne.io.RawArray(raw_data_array.reshape(1, -1), info, verbose=False)
    # 自动分期
    sls = yasa.SleepStaging(raw_tmp, eeg_name="EEG")
    hypno = sls.predict()
    return hypno

# ================= 5. 主程序 =================
if not os.path.exists(model_path):
    print("❌ 找不到模型文件！"); exit()

print("🧠 正在加载模型...")
model = load_model_with_fix(model_path)
print("✅ 模型就绪！开始批量处理...")

raw_files = glob.glob(os.path.join(raw_dir, "*PSG.edf"))
results = []

# 定义 Sleep-EDF 标签到数字的映射字典
# Sleep-EDF 通常用 'Sleep stage W', '1', '2' 等
# 映射目标: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
STAGE_MAPPING = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3, # N4 合并入 N3
    'Sleep stage R': 4,
    'Movement time': -1, # 伪差/未分期
    'Sleep stage ?': -1
}

for idx, raw_path in enumerate(raw_files):
    fname = os.path.basename(raw_path)
    subject_prefix = fname[:6] 
    
    print(f"\n[{idx+1}/{len(raw_files)}] 处理受试者: {subject_prefix} ...")
    
    # 智能寻找标签文件
    search_pattern = os.path.join(raw_dir, f"{subject_prefix}*Hypnogram.edf")
    found_hypnos = glob.glob(search_pattern)
    
    if len(found_hypnos) > 0:
        hypno_path = found_hypnos[0]
        has_label = True
    else:
        hypno_path = None
        has_label = False
        print(f"   ⚠️ 未找到 {subject_prefix} 的标签文件")

    try:
        # 1. 读取数据
        raw = mne.io.read_raw_edf(raw_path, preload=True, verbose=False)
        if raw.info['sfreq'] != TARGET_SFREQ: raw.resample(TARGET_SFREQ)
        data_raw = raw.get_data(picks=[0])[0]
        
        # 2. 前测 (原始数据分期)
        print("   🔍 1. 识别原始数据分期...", end="")
        pred_before = run_yasa_staging(data_raw, TARGET_SFREQ)
        print(" [完成]")

        # 3. 干预 (模型去噪)
        print("   ⚡ 2. 执行深度去噪...", end="")
        slen = 512
        n_seg = len(data_raw) // slen
        batch_in, scales = [], []
        for i in range(n_seg):
            seg = data_raw[i*slen : (i+1)*slen]
            std = np.std(seg) if np.std(seg)!=0 else 1.0
            batch_in.append(seg/std); scales.append(std)
        
        preds_out = model.predict(np.array(batch_in).reshape(-1, slen, 1), batch_size=256, verbose=0)
        data_clean = np.array([preds_out[i].flatten() * scales[i] for i in range(n_seg)]).flatten()
        print(" [完成]")

        # 4. 后测 (去噪后分期)
        print("   🔍 3. 识别修复后分期...", end="")
        pred_after = run_yasa_staging(data_clean, TARGET_SFREQ)
        print(" [完成]")

        # 5. 计算结果 (这里是修复的核心部分)
        res_dict = {"Subject": subject_prefix}
        if has_label:
            # === 修复点：使用 MNE 读取 EDF 标签，而不是 yasa.load_profusion ===
            annot = mne.read_annotations(hypno_path)
            raw.set_annotations(annot, emit_warning=False)
            
            # 使用 MNE 将注释切分为 30s 的事件，并根据字典映射为数字
            events, _ = mne.events_from_annotations(
                raw, 
                event_id=STAGE_MAPPING, 
                chunk_duration=30.,
                verbose=False
            )
            
            # 提取第三列作为真实标签 (0, 1, 2...)
            hypno_true = events[:, 2]
            # =============================================================
            
            # 对齐长度 (取三者最短)
            L = min(len(hypno_true), len(pred_before), len(pred_after))
            
            # 过滤掉标签为 -1 (Movement time/Unknown) 的片段，不参与计分
            valid_mask = hypno_true[:L] != -1
            
            if np.sum(valid_mask) > 0:
                acc_before = accuracy_score(hypno_true[:L][valid_mask], pred_before[:L][valid_mask]) * 100
                acc_after = accuracy_score(hypno_true[:L][valid_mask], pred_after[:L][valid_mask]) * 100
                improvement = acc_after - acc_before
                
                res_dict["Acc_Before(%)"] = round(acc_before, 2)
                res_dict["Acc_After(%)"] = round(acc_after, 2)
                res_dict["Improvement(%)"] = round(improvement, 2)
                
                print(f"   🏆 结果: 准确率 {res_dict['Acc_Before(%)']}% -> {res_dict['Acc_After(%)']}% (提升 {res_dict['Improvement(%)']}%)")
            else:
                print("   ⚠️ 有效标签数量不足，跳过计算")
        else:
            print("   ⚠️ 无标签，跳过计算")
            
        results.append(res_dict)

    except Exception as e:
        print(f"   ❌ 出错: {e}")
        # 打印详细错误方便调试
        import traceback
        traceback.print_exc()

if results:
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"\n📄 结果已保存: {output_csv_path}")