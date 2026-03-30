"""
EEG去噪测试脚本 - 结果写入文件
"""
import os
import numpy as np
import warnings
from pathlib import Path

# 设置环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

# 输出文件
output_file = Path(__file__).resolve().parent / "test_result.txt"

def log(msg, f):
    print(msg)
    f.write(msg + "\n")

with open(output_file, 'w', encoding='utf-8') as f:
    log("=" * 50, f)
    log("EEG去噪算法测试", f)
    log("=" * 50, f)
    
    # 1. 导入TensorFlow
    log("\n[1/4] 导入TensorFlow...", f)
    try:
        import tensorflow as tf
        log(f"✅ TensorFlow版本: {tf.__version__}", f)
    except Exception as e:
        log(f"❌ TensorFlow导入失败: {e}", f)
        exit()
    
    # 2. 导入其他依赖
    log("\n[2/4] 导入其他依赖...", f)
    try:
        from tensorflow.keras import layers, models, Sequential
        import mne
        log(f"✅ MNE版本: {mne.__version__}", f)
    except Exception as e:
        log(f"❌ 依赖导入失败: {e}", f)
        exit()
    
    # 3. 定义模型结构
    log("\n[3/4] 定义模型结构...", f)
    
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

    log("✅ 模型结构定义完成", f)
    
    # 4. 加载模型
    log("\n[4/4] 加载模型权重...", f)
    PROJECT_ROOT = Path(__file__).resolve().parent
    model_path = PROJECT_ROOT / "03_训练模型" / "去噪模型v2最终版.h5"
    log(f"模型路径: {model_path}", f)
    log(f"文件存在: {model_path.exists()}", f)
    
    try:
        model = build_model()
        log("✅ 模型结构构建完成", f)
        
        model.load_weights(str(model_path))
        log("✅ 模型权重加载成功!", f)
    except Exception as e:
        log(f"❌ 模型加载失败: {e}", f)
        import traceback
        log(traceback.format_exc(), f)
        exit()
    
    # 5. 测试推理
    log("\n[5/5] 测试推理...", f)
    try:
        # 生成模拟EEG信号
        np.random.seed(42)
        t = np.linspace(0, 2, 512)
        
        # 模拟干净EEG
        clean_eeg = (0.5 * np.sin(2 * np.pi * 2 * t) +
                     0.3 * np.sin(2 * np.pi * 6 * t) +
                     0.2 * np.sin(2 * np.pi * 10 * t))
        
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
        
        log("✅ 推理成功!", f)
        log(f"   输入形状: {input_tensor.shape}", f)
        log(f"   输出形状: {output_tensor.shape}", f)
        log(f"   噪声减少率 (NRR): {nrr:.2f}%", f)
        log(f"   与原始信号相关系数: {cc:.4f}", f)
        log(f"   与干净信号相关系数: {cc_clean:.4f}", f)
        
    except Exception as e:
        log(f"❌ 推理失败: {e}", f)
        import traceback
        log(traceback.format_exc(), f)
    
    log("\n" + "=" * 50, f)
    log("测试完成!", f)
    log("=" * 50, f)

print(f"\n结果已保存到: {output_file}")