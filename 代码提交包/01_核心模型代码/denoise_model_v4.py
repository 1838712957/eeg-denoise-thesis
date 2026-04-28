"""
V4去噪模型 - 完整实现
多尺度残差结构 + SE注意力机制 + Delta波保护损失函数

论文对应: 第三章 系统设计与实现
"""
import os
import numpy as np
import warnings
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, Sequential
from tensorflow.keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# 导入模型组件
from model_components import SEBlock, Res_BasicBlock, BasicBlockall


# ================= 损失函数 =================

def rmse_loss(y_true, y_pred):
    """RMSE损失"""
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


def delta_protection_loss(y_true, y_pred, fs=100, delta_low=0.5, delta_high=4.0):
    """
    Delta波保护损失函数 (基于FFT的可微频域约束)
    
    论文对应: 3.3 损失函数设计
    
    原理:
    1. 使用FFT将时域信号转换到频域
    2. 提取Delta频段(0.5-4Hz)的幅度谱
    3. 计算该频段的MSE损失
    
    参数:
        y_true: 真实信号
        y_pred: 预测信号
        fs: 采样率
        delta_low: Delta波下限频率
        delta_high: Delta波上限频率
    """
    # FFT变换
    fft_true = tf.signal.rfft(tf.squeeze(y_true, axis=-1))
    fft_pred = tf.signal.rfft(tf.squeeze(y_pred, axis=-1))
    
    # 计算频率分辨率
    n = tf.cast(tf.shape(y_true)[1], tf.float32)
    freq_res = fs / n
    
    # 计算Delta频段索引
    delta_low_idx = tf.cast(delta_low / freq_res, tf.int32)
    delta_high_idx = tf.cast(delta_high / freq_res, tf.int32)
    
    # 提取Delta频段幅度谱
    delta_true = tf.abs(fft_true[:, delta_low_idx:delta_high_idx])
    delta_pred = tf.abs(fft_pred[:, delta_low_idx:delta_high_idx])
    
    # 计算Delta频段MSE
    return K.mean(K.square(delta_true - delta_pred))


def combined_loss(y_true, y_pred, lambda_delta=0.1):
    """
    组合损失: RMSE + Delta波保护
    
    L_total = L_MSE + λ · L_Delta
    
    参数:
        y_true: 真实信号
        y_pred: 预测信号
        lambda_delta: Delta保护系数 (论文中使用0.1)
    """
    rmse = rmse_loss(y_true, y_pred)
    delta_loss = delta_protection_loss(y_true, y_pred)
    return rmse + lambda_delta * delta_loss


# ================= 模型构建 =================

def build_v4_model(input_length=3000, use_se=True):
    """
    构建V4去噪模型
    
    论文对应: 3.2 去噪网络结构设计
    
    网络结构:
    Input(3000, 1) -> Conv1D(32, 7) -> BN -> ReLU -> 
    BasicBlockall(use_se=True) -> Conv1D(96, 1) -> BN -> ReLU ->
    Conv1D(1, 7) -> Output
    
    参数:
        input_length: 输入信号长度 (默认3000，对应30秒@100Hz)
        use_se: 是否使用SE注意力
    """
    inputs = layers.Input(shape=(input_length, 1))
    
    # 初始卷积层
    x = layers.Conv1D(32, 7, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 多尺度残差块 + SE注意力
    x = BasicBlockall(use_se=use_se)(x)
    
    # 通道调整 (96 -> 32)
    x = layers.Conv1D(32, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 输出层
    outputs = layers.Conv1D(1, 7, padding='same')(x)
    
    # 残差连接
    outputs = layers.Add()([inputs, outputs])
    
    model = models.Model(inputs=inputs, outputs=outputs, name='V4_DenoiseModel')
    return model


def build_baseline_model(input_length=3000):
    """
    构建基础CNN模型 (用于消融实验对比)
    
    论文对应: 4.3 消融实验 Baseline
    """
    inputs = layers.Input(shape=(input_length, 1))
    
    # 简单的编码器-解码器结构
    # 编码器
    x = layers.Conv1D(32, 7, padding='same', activation='relu')(inputs)
    x = layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    
    # 解码器
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv1D(1, 7, padding='same')(x)
    
    # 残差连接
    outputs = layers.Add()([inputs, x])
    
    model = models.Model(inputs=inputs, outputs=outputs, name='Baseline_DenoiseModel')
    return model


# ================= 训练函数 =================

def train_model(model, train_data, val_data, epochs=100, batch_size=32, lr=0.001):
    """
    训练模型
    
    参数:
        model: 待训练模型
        train_data: (x_train, y_train)
        val_data: (x_val, y_val)
        epochs: 训练轮数
        batch_size: 批量大小
        lr: 学习率
    """
    x_train, y_train = train_data
    x_val, y_val = val_data
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=lambda y_true, y_pred: combined_loss(y_true, y_pred, lambda_delta=0.1),
        metrics=['mse']
    )
    
    # 回调函数
    callbacks_list = [
        callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
    
    # 训练
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1
    )
    
    return history


# ================= 评估指标 =================

def calc_rrmse(clean, est):
    """
    相对均方根误差 (RRMSE)
    越小越好
    
    RRMSE = RMS(error) / RMS(clean)
    """
    return np.sqrt(np.mean((clean - est) ** 2)) / np.sqrt(np.mean(clean ** 2))


def calc_cc(clean, est):
    """
    皮尔逊相关系数 (CC)
    越接近1越好
    """
    return np.corrcoef(clean.flatten(), est.flatten())[0, 1]


def calc_delta_energy_preservation(clean, est, fs=100):
    """
    Delta波能量保持率
    越大越好
    
    论文对应: N3期Delta波保护效果评估
    """
    from scipy.signal import butter, filtfilt
    
    # 带通滤波提取Delta波段
    nyq = fs / 2
    low = 0.5 / nyq
    high = 4.0 / nyq
    b, a = butter(2, [low, high], btype='band')
    
    clean_delta = filtfilt(b, a, clean.flatten())
    est_delta = filtfilt(b, a, est.flatten())
    
    # 计算能量
    energy_clean = np.sum(clean_delta ** 2)
    energy_est = np.sum(est_delta ** 2)
    
    return energy_est / energy_clean * 100


# ================= 主函数 =================

if __name__ == "__main__":
    print("=" * 60)
    print("V4去噪模型 - 多尺度残差 + SE注意力 + Delta波保护")
    print("=" * 60)
    
    # 构建模型
    model = build_v4_model(input_length=3000, use_se=True)
    model.summary()
    
    print("\n模型参数量:", model.count_params())