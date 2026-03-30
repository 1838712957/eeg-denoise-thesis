"""
消融实验 - 模型训练脚本
训练所有模型变体并保存
"""
import os
import numpy as np
import warnings
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential, callbacks, optimizers
from tensorflow.keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "03_训练模型"
OUTPUT_DIR = PROJECT_ROOT / "05_处理结果" / "消融实验"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================= 模型定义 =================

class SEBlock(layers.Layer):
    """SE注意力模块"""
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
    """残差基础块"""
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
        if use_se:
            self.se = SEBlock(32, se_reduction)
    
    def call(self, inputs):
        out = self.bblock(inputs)
        if self.use_se:
            out = self.se(out)
        return layers.add([out, inputs])
    
    def get_config(self):
        config = super(Res_BasicBlock, self).get_config()
        config.update({"kernelsize": self.kernelsize, "stride": self.stride, 
                       "use_se": self.use_se, "se_reduction": self.se_reduction})
        return config


class BasicBlockall(layers.Layer):
    """多尺度并行卷积块"""
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


class SingleScaleBlock(layers.Layer):
    """单一尺度块（只用kernel_size=3）"""
    def __init__(self, use_se=False, se_reduction=16, **kwargs):
        super(SingleScaleBlock, self).__init__(**kwargs)
        self.use_se, self.se_reduction = use_se, se_reduction
        self.block = Sequential([
            Res_BasicBlock(3, use_se=use_se), 
            Res_BasicBlock(3, use_se=use_se),
            Res_BasicBlock(3, use_se=use_se)
        ])
    
    def call(self, inputs):
        return self.block(inputs)
    
    def get_config(self):
        config = super(SingleScaleBlock, self).get_config()
        config.update({"use_se": self.use_se, "se_reduction": self.se_reduction})
        return config


# ================= 模型构建函数 =================

def build_baseline_model():
    """Baseline: 基础1D CNN，无残差、无多尺度、无注意力"""
    inp = layers.Input(shape=(512, 1))
    x = layers.Conv1D(32, 7, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(32, 7, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(16, 7, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    out = layers.Conv1D(1, 7, padding='same')(x)
    return models.Model(inp, out, name="Baseline")


def build_model_without_se():
    """V4 w/o SE: 移除SE注意力机制"""
    inp = layers.Input(shape=(512, 1))
    x = layers.Conv1D(32, 7, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = BasicBlockall(use_se=False)(x)
    out = layers.Conv1D(1, 7, padding='same')(x)
    return models.Model(inp, out, name="V4_wo_SE")


def build_model_single_scale():
    """V4 Single Scale: 单一尺度（只用kernel_size=3）"""
    inp = layers.Input(shape=(512, 1))
    x = layers.Conv1D(32, 7, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = SingleScaleBlock(use_se=True)(x)
    out = layers.Conv1D(1, 7, padding='same')(x)
    return models.Model(inp, out, name="V4_Single_Scale")


def build_model_complete():
    """V4 Complete: 完整V4模型"""
    inp = layers.Input(shape=(512, 1))
    x = layers.Conv1D(32, 7, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = BasicBlockall(use_se=True)(x)
    out = layers.Conv1D(1, 7, padding='same')(x)
    return models.Model(inp, out, name="V4_Complete")


# ================= 损失函数 =================

def rmse_loss(y_true, y_pred):
    """RMSE损失"""
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


def combined_loss(y_true, y_pred):
    """组合损失: RMSE + 频域一致性"""
    # 时域RMSE
    rmse = K.sqrt(K.mean(K.square(y_true - y_pred)))
    
    # 频域一致性（简化版）
    y_true_fft = K.abs(tf.signal.rfft(tf.squeeze(y_true, axis=-1)))
    y_pred_fft = K.abs(tf.signal.rfft(tf.squeeze(y_pred, axis=-1)))
    freq_loss = K.mean(K.square(y_true_fft - y_pred_fft))
    
    return rmse + 0.1 * freq_loss


# ================= 数据生成器 =================

def generate_training_data(n_samples=10000, segment_len=512):
    """生成模拟训练数据"""
    print(f"生成 {n_samples} 个训练样本...")
    
    # 生成干净的EEG信号（模拟Delta波）
    clean_signals = []
    noisy_signals = []
    
    for i in range(n_samples):
        # 生成基础Delta波 (0.5-4Hz)
        t = np.linspace(0, 2, segment_len)  # 2秒
        
        # Delta波成分
        delta1 = np.sin(2 * np.pi * 1.0 * t) * np.random.uniform(0.5, 1.5)
        delta2 = np.sin(2 * np.pi * 2.0 * t) * np.random.uniform(0.3, 0.8)
        delta3 = np.sin(2 * np.pi * 0.5 * t) * np.random.uniform(0.2, 0.6)
        
        # 添加一些高频成分
        theta = np.sin(2 * np.pi * 6 * t) * np.random.uniform(0.1, 0.3)
        
        clean = delta1 + delta2 + delta3 + theta
        
        # 添加噪声（模拟EMG和EOG伪迹）
        # 高频噪声
        emg_noise = np.random.randn(segment_len) * np.random.uniform(0.1, 0.5)
        
        # 低频漂移
        drift = np.sin(2 * np.pi * 0.1 * t) * np.random.uniform(0.2, 0.5)
        
        # 瞬态伪迹
        transient = np.zeros(segment_len)
        n_transients = np.random.randint(0, 3)
        for _ in range(n_transients):
            pos = np.random.randint(0, segment_len - 50)
            transient[pos:pos+50] = np.random.randn(50) * np.random.uniform(0.5, 2.0)
        
        noisy = clean + emg_noise + drift + transient
        
        # 标准化
        std = np.std(clean)
        if std > 0:
            clean = clean / std
            noisy = noisy / std
        
        clean_signals.append(clean)
        noisy_signals.append(noisy)
    
    clean_signals = np.array(clean_signals).reshape(-1, segment_len, 1)
    noisy_signals = np.array(noisy_signals).reshape(-1, segment_len, 1)
    
    return noisy_signals, clean_signals


# ================= 训练函数 =================

def train_model(model, model_name, X_train, y_train, epochs=50, batch_size=64):
    """训练模型"""
    print(f"\n训练 {model_name}...")
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss=combined_loss,
        metrics=[rmse_loss]
    )
    
    # 回调函数
    callbacks_list = [
        callbacks.ModelCheckpoint(
            str(MODEL_DIR / f"{model_name}.h5"),
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        )
    ]
    
    # 训练
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1
    )
    
    return history


# ================= 主程序 =================

def main():
    print("=" * 80)
    print("消融实验 - 模型训练")
    print("=" * 80)
    
    # 生成训练数据
    print("\n[1] 生成训练数据...")
    X_train, y_train = generate_training_data(n_samples=10000)
    print(f"训练数据形状: {X_train.shape}")
    
    # 定义所有模型变体
    print("\n[2] 构建模型变体...")
    model_variants = {
        'Baseline': build_baseline_model(),
        'V4_wo_SE': build_model_without_se(),
        'V4_Single_Scale': build_model_single_scale(),
        'V4_Complete': build_model_complete()
    }
    
    for name, model in model_variants.items():
        print(f"  {name}: {model.count_params()} 参数")
    
    # 训练所有模型
    print("\n[3] 开始训练...")
    histories = {}
    
    for name, model in model_variants.items():
        print(f"\n{'='*60}")
        print(f"训练模型: {name}")
        print(f"{'='*60}")
        
        history = train_model(model, name, X_train, y_train, epochs=30)
        histories[name] = history
    
    # 输出训练结果汇总
    print("\n" + "=" * 80)
    print("训练完成！模型已保存到: " + str(MODEL_DIR))
    print("=" * 80)
    
    # 保存训练日志
    log_file = OUTPUT_DIR / "training_log.txt"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("消融实验训练日志\n")
        f.write("=" * 80 + "\n\n")
        
        for name, history in histories.items():
            f.write(f"\n{name}:\n")
            f.write(f"  最终训练损失: {history.history['loss'][-1]:.4f}\n")
            f.write(f"  最终验证损失: {history.history['val_loss'][-1]:.4f}\n")
            f.write(f"  参数数量: {model_variants[name].count_params()}\n")
    
    print(f"训练日志已保存到: {log_file}")


if __name__ == "__main__":
    main()
