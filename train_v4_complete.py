"""
单独训练V4_Complete模型
"""
import os
import numpy as np
import warnings
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, Sequential
from tensorflow.keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

PROJECT_ROOT = Path('.')
MODEL_DIR = PROJECT_ROOT / '03_训练模型'

# ================= 模型组件 =================

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


# ================= 损失函数 =================

def rmse_loss(y_true, y_pred):
    """RMSE损失"""
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


def combined_loss(y_true, y_pred):
    """组合损失: RMSE + 频域一致性"""
    rmse = K.sqrt(K.mean(K.square(y_true - y_pred)))
    y_true_fft = K.abs(tf.signal.rfft(tf.squeeze(y_true, axis=-1)))
    y_pred_fft = K.abs(tf.signal.rfft(tf.squeeze(y_pred, axis=-1)))
    freq_loss = K.mean(K.square(y_true_fft - y_pred_fft))
    return rmse + 0.1 * freq_loss


# ================= 数据生成 =================

def generate_training_data(n_samples=10000, segment_len=512):
    """生成模拟训练数据"""
    print(f"生成 {n_samples} 个训练样本...")
    
    clean_signals = []
    noisy_signals = []
    
    for i in range(n_samples):
        t = np.linspace(0, 2, segment_len)
        
        # Delta波成分
        delta1 = np.sin(2 * np.pi * 1.0 * t) * np.random.uniform(0.5, 1.5)
        delta2 = np.sin(2 * np.pi * 2.0 * t) * np.random.uniform(0.3, 0.8)
        delta3 = np.sin(2 * np.pi * 0.5 * t) * np.random.uniform(0.2, 0.6)
        theta = np.sin(2 * np.pi * 6 * t) * np.random.uniform(0.1, 0.3)
        
        clean = delta1 + delta2 + delta3 + theta
        
        # 噪声
        emg_noise = np.random.randn(segment_len) * np.random.uniform(0.1, 0.5)
        drift = np.sin(2 * np.pi * 0.1 * t) * np.random.uniform(0.2, 0.5)
        
        # 瞬态伪迹
        transient = np.zeros(segment_len)
        n_transients = np.random.randint(0, 3)
        for _ in range(n_transients):
            pos = np.random.randint(0, segment_len - 50)
            transient[pos:pos+50] = np.random.randn(50) * np.random.uniform(0.5, 2.0)
        
        noisy = clean + emg_noise + drift + transient
        
        std = np.std(clean)
        if std > 0:
            clean = clean / std
            noisy = noisy / std
        
        clean_signals.append(clean)
        noisy_signals.append(noisy)
    
    return np.array(noisy_signals).reshape(-1, segment_len, 1), np.array(clean_signals).reshape(-1, segment_len, 1)


# ================= 模型构建 =================

def build_v4_complete_model(input_shape=(512, 1)):
    """构建完整的V4模型（多尺度+SE注意力）"""
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, 7, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = BasicBlockall(use_se=True)(x)
    out = layers.Conv1D(1, 7, padding='same')(x)
    return models.Model(inp, out, name='V4_Complete')


def main():
    print("=" * 60)
    print("训练 V4_Complete 模型")
    print("=" * 60)
    
    # 生成数据
    print("\n生成训练数据...")
    X_train, y_train = generate_training_data(n_samples=10000)
    print(f"训练数据形状: {X_train.shape}")
    
    # 构建模型
    print("\n构建模型...")
    model = build_v4_complete_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=combined_loss,
        metrics=[rmse_loss]
    )
    model.summary()
    
    # 回调
    callbacks_list = [
        callbacks.ModelCheckpoint(
            str(MODEL_DIR / 'V4_Complete.h5'),
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True,
            monitor='val_loss'
        ),
        callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=5,
            monitor='val_loss',
            verbose=1
        )
    ]
    
    # 训练
    print("\n开始训练...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=64,
        callbacks=callbacks_list,
        verbose=1
    )
    
    print("\n训练完成!")
    print(f"最终验证损失: {min(history.history['val_loss']):.5f}")
    
    # 验证模型文件
    model_path = MODEL_DIR / 'V4_Complete.h5'
    if model_path.exists():
        print(f"✓ 模型已保存: {model_path}")
    else:
        print("✗ 模型保存失败")

if __name__ == '__main__':
    main()