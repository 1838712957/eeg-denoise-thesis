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
from tensorflow.keras import layers, models, Sequential
from pathlib import Path

# ========================================================================
# 0. 核弹级模型修复 (JSON 深度清洗版)
# ========================================================================
def clean_config_dict(config):
    """递归遍历配置字典，删除所有不兼容的 'synchronized' 参数"""
    if isinstance(config, dict):
        # 如果是 BatchNormalization 层，删除 synchronized 键
        if 'class_name' in config and config['class_name'] == 'BatchNormalization':
            if 'config' in config and 'synchronized' in config['config']:
                print("   ✂️ 已切除 BatchNormalization 层的 'synchronized' 参数")
                del config['config']['synchronized']
        
        # 递归处理所有子项
        for key, value in config.items():
            clean_config_dict(value)
    elif isinstance(config, list):
        for item in config:
            clean_config_dict(item)

def load_model_with_deep_clean(original_path):
    temp_path = original_path.replace(".h5", "_deep_fixed.h5")
    print(f"🔧 正在执行模型深度修复...\n   源文件: {original_path}")
    
    # 1. 复制文件
    shutil.copyfile(original_path, temp_path)
    
    try:
        with h5py.File(temp_path, 'r+') as f:
            if 'model_config' in f.attrs:
                # 读取原始配置
                config_str = f.attrs['model_config']
                if isinstance(config_str, bytes): 
                    config_str = config_str.decode('utf-8')
                
                # --- 第一步：字符串级暴力替换 (解决 batch_shape) ---
                fixed_str = config_str.replace('"batch_shape":', '"batch_input_shape":')
                fixed_str = fixed_str.replace("'batch_shape':", "'batch_input_shape':")
                
                # --- 第二步：JSON 级精准手术 (解决 synchronized) ---
                try:
                    config_json = json.loads(fixed_str)
                    clean_config_dict(config_json) # 递归清洗
                    final_str = json.dumps(config_json)
                    
                    # 写回文件
                    f.attrs['model_config'] = final_str.encode('utf-8')
                    print("   ✅ 模型配置文件深度清洗完成！")
                except json.JSONDecodeError:
                    print("   ⚠️ JSON 解析失败，回退到字符串替换模式...")
                    # 如果解析失败，尝试暴力替换字符串
                    final_str = fixed_str.replace('"synchronized": true,', '')
                    final_str = final_str.replace('"synchronized": false,', '')
                    f.attrs['model_config'] = final_str.encode('utf-8')

    except Exception as e:
        print(f"⚠️ 修复过程警告: {e}")

    # 2. 定义伪装类 (解决 DTypePolicy)
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

    # 3. 加载清洗后的模型
    try:
        custom_objects = {
            'SEBlock': SEBlock, 
            'Res_BasicBlock': Res_BasicBlock, 
            'BasicBlockall': BasicBlockall,
            'DTypePolicy': FakeDTypePolicy,
            'Float32': FakeDTypePolicy
        }
        
        # compile=False 是关键，不加载优化器状态
        model = tf.keras.models.load_model(temp_path, custom_objects=custom_objects, compile=False)
        print("✅ 模型加载成功！(已清除临时文件)")
        
        try: os.remove(temp_path)
        except: pass
        return model
    except Exception as e:
        print(f"❌ 最终加载失败: {e}")
        # 不删除临时文件，方便调试
        exit()

# ========================================================================
# 1. 路径与配置
# ========================================================================
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
output_csv_path = os.path.join(base_dir, "Quantitative_Results_Final.csv") 
model_path = pick_existing_path(
    os.path.join(base_dir, "03_训练模型", "denoise_model.h5"),
    os.path.join(base_dir, "denoise_model.h5"),
)
TARGET_SFREQ = 256 

# ========================================================================
# 2. 网络结构定义
# ========================================================================
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

# ========================================================================
# 3. 运行逻辑
# ========================================================================
if not os.path.exists(model_path):
    print(f"❌ 找不到模型文件: {model_path}")
    exit()

# 使用深度清洗加载模型
model = load_model_with_deep_clean(model_path)

def calculate_metrics(raw, cleaned):
    if np.std(raw) == 0 or np.std(cleaned) == 0: return 0, 0
    var_raw, var_clean = np.var(raw), np.var(cleaned)
    nrr = (var_raw - var_clean) / var_raw * 100 
    cc = np.corrcoef(raw, cleaned)[0, 1] 
    return nrr, cc

raw_files = glob.glob(os.path.join(raw_dir, "*PSG.edf"))
print(f"📂 发现 {len(raw_files)} 个受试者，开始批量处理...")

results = []
for idx, raw_path in enumerate(raw_files):
    fname = os.path.basename(raw_path)
    sid = fname.split('-')[0]
    print(f"[{idx+1}/{len(raw_files)}] 处理中: {sid} ...", end="")
    
    asr_path = os.path.join(asr_dir, f"{os.path.splitext(fname)[0]}_fixed_clean.set")
    if not os.path.exists(asr_path):
        print(" [跳过: 缺ASR对比文件]")
        continue
        
    try:
        raw_m = mne.io.read_raw_edf(raw_path, preload=True, verbose=False)
        if raw_m.info['sfreq'] != TARGET_SFREQ: raw_m.resample(TARGET_SFREQ)
        d_raw = raw_m.get_data(picks=[0])[0]
        
        raw_a = mne.io.read_raw_eeglab(asr_path, preload=True, verbose=False)
        if raw_a.info['sfreq'] != TARGET_SFREQ: raw_a.resample(TARGET_SFREQ)
        d_asr = raw_a.get_data(picks=[0])[0]
        
        L = min(len(d_raw), len(d_asr))
        d_raw, d_asr = d_raw[:L], d_asr[:L]
        
        slen, nseg = 512, min(L // 512, 1000)
        batch, scales = [], []
        for i in range(nseg):
            s = d_raw[i*slen:(i+1)*slen]
            std = np.std(s) if np.std(s) != 0 else 1.0
            batch.append(s/std); scales.append(std)
            
        t0 = time.time()
        p = model.predict(np.array(batch).reshape(-1, 512, 1), batch_size=128, verbose=0)
        t_inf = time.time() - t0
        d_ai = np.array([p[i].flatten() * scales[i] for i in range(nseg)]).flatten()
        
        n_a, c_a = calculate_metrics(d_raw[:len(d_ai)], d_asr[:len(d_ai)])
        n_o, c_o = calculate_metrics(d_raw[:len(d_ai)], d_ai)
        
        results.append({
            "Subject": sid, 
            "NRR_ASR(%)": round(n_a, 2), 
            "NRR_Ours(%)": round(n_o, 2), 
            "CC_Raw_ASR": round(c_a, 4), 
            "CC_Raw_Ours": round(c_o, 4), 
            "Time(s)": round(t_inf, 2)
        })
        print(f" [完成] Ours NRR: {round(n_o, 1)}%")
    except Exception as e:
        print(f" [失败: {e}]")

if results:
    with open(output_csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)
    print(f"\n✅ 成果表格已保存至: {output_csv_path}")