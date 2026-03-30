"""
简单测试：验证模型是否能正确加载
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("[1] 导入TensorFlow...")
import tensorflow as tf
print(f"    TensorFlow版本: {tf.__version__}")

print("\n[2] 检查模型文件...")
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent

denoise_model_path = PROJECT_ROOT / "03_训练模型" / "去噪模型v2最终版.h5"
deepsleepnet_path = PROJECT_ROOT / "03_训练模型" / "DeepSleepNet裁判模型.h5"

print(f"    去噪模型路径: {denoise_model_path}")
print(f"    文件存在: {denoise_model_path.exists()}")

print(f"    DeepSleepNet路径: {deepsleepnet_path}")
print(f"    文件存在: {deepsleepnet_path.exists()}")

print("\n[3] 加载去噪模型...")
from tensorflow.keras import layers, models, Sequential

# 定义去噪模型
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
        config.update({"kernelsize": self.kernelsize, "stride": self.stride, "use_se": self.use_se, "se_reduction": self.se_reduction})
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

def build_denoise_model():
    inp = layers.Input(shape=(512, 1))
    x = layers.Conv1D(32, 7, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = BasicBlockall(use_se=True)(x)
    out = layers.Conv1D(1, 7, padding='same')(x)
    return models.Model(inp, out)

denoise_model = build_denoise_model()
denoise_model.load_weights(str(denoise_model_path))
print("    去噪模型加载成功!")

print("\n[4] 加载DeepSleepNet模型...")
from tensorflow.keras.models import load_model
deepsleepnet = load_model(str(deepsleepnet_path), compile=False)
print("    DeepSleepNet加载成功!")

print("\n[5] 测试推理...")
import numpy as np
test_input = np.random.randn(1, 512, 1).astype(np.float32)
test_output = denoise_model.predict(test_input, verbose=0)
print(f"    去噪模型输出形状: {test_output.shape}")

test_input2 = np.random.randn(1, 3000, 1).astype(np.float32)
test_output2 = deepsleepnet.predict(test_input2, verbose=0)
print(f"    DeepSleepNet输出形状: {test_output2.shape}")

print("\n[OK] 所有模型加载测试通过!")