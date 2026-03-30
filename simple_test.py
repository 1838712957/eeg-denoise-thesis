"""
简化版EEG去噪测试脚本
"""
import os
import numpy as np
import tensorflow as tf
import warnings
from tensorflow.keras import layers, models, Sequential
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 50)
print("EEG去噪算法简化测试")
print("=" * 50)

# 路径配置
PROJECT_ROOT = Path(__file__).resolve().parent
model_path = PROJECT_ROOT / "03_训练模型" / "去噪模型v2最终版.h5"

# 网络结构定义
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

def build_model():
    inp = layers.Input(shape=(512, 1))
    x = layers.Conv1D(32, 7, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = BasicBlockall(use_se=True)(x)
    out = layers.Conv1D(1, 7, padding='same')(x)
    return models.Model(inp, out)

# 1. 测试模型加载
print("\n[1/3] 测试模型加载...")
print(f"模型路径: {model_path}")
print(f"文件存在: {model_path.exists()}")

try:
    model = build_model()
    model.load_weights(str(model_path))
    print("✅ 模型加载成功!")
    model.summary()
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit()

# 2. 测试推理
print("\n[2/3] 测试模型推理...")
try:
    # 生成模拟EEG信号 (含噪声)
    np.random.seed(42)
    t = np.linspace(0, 2, 512)  # 2秒, 256Hz
    
    # 模拟干净EEG (Delta + Theta + Alpha)
    clean_eeg = (0.5 * np.sin(2 * np.pi * 2 * t) +    # Delta (2 Hz)
                 0.3 * np.sin(2 * np.pi * 6 * t) +    # Theta (6 Hz)
                 0.2 * np.sin(2 * np.pi * 10 * t))    # Alpha (10 Hz)
    
    # 添加噪声
    noise = 0.3 * np.random.randn(512)
    noisy_eeg = clean_eeg + noise
    
    # 标准化
    std_val = np.std(noisy_eeg)
    input_signal = noisy_eeg / std_val
    
    # 推理
    input_tensor = input_signal.reshape(1, 512, 1)
    output_tensor = model.predict(input_tensor, verbose=0)
    output_signal = output_tensor.flatten() * std_val
    
    # 计算指标
    nrr = (np.var(noisy_eeg) - np.var(output_signal)) / np.var(noisy_eeg) * 100
    cc = np.corrcoef(noisy_eeg, output_signal)[0, 1]
    cc_clean = np.corrcoef(clean_eeg, output_signal)[0, 1]
    
    print(f"✅ 推理成功!")
    print(f"   - 输入形状: {input_tensor.shape}")
    print(f"   - 输出形状: {output_tensor.shape}")
    print(f"   - 噪声减少率 (NRR): {nrr:.2f}%")
    print(f"   - 与原始信号相关系数: {cc:.4f}")
    print(f"   - 与干净信号相关系数: {cc_clean:.4f}")
    
except Exception as e:
    print(f"❌ 推理失败: {e}")
    import traceback
    traceback.print_exc()

# 3. 测试批量推理
print("\n[3/3] 测试批量推理...")
try:
    batch_size = 10
    batch_input = np.random.randn(batch_size, 512, 1)
    batch_output = model.predict(batch_input, verbose=0)
    print(f"✅ 批量推理成功!")
    print(f"   - 批量大小: {batch_size}")
    print(f"   - 输出形状: {batch_output.shape}")
    
except Exception as e:
    print(f"❌ 批量推理失败: {e}")

print("\n" + "=" * 50)
print("✅ 所有测试完成!")
print("=" * 50)