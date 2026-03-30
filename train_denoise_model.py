"""
EEG去噪模型训练脚本
使用DREAMS数据集 (Raw_edf 2) 从subject6开始训练
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential, callbacks
import matplotlib.pyplot as plt
from pathlib import Path
import mne
from scipy.signal import welch
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).resolve().parent

# ==================== 模型定义 ====================
class SEBlock(layers.Layer):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels=32, reduction=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
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
        return {"channels": self.channels, "reduction": self.reduction}


class Res_BasicBlock(layers.Layer):
    """残差基本块"""
    def __init__(self, kernelsize, stride=1, use_se=False, **kwargs):
        super(Res_BasicBlock, self).__init__(**kwargs)
        self.kernelsize = kernelsize
        self.use_se = use_se
        
        self.bblock = Sequential([
            layers.Conv1D(32, kernelsize, strides=stride, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv1D(16, kernelsize, strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv1D(32, kernelsize, strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        
        if use_se:
            self.se = SEBlock(32)
    
    def call(self, inputs):
        out = self.bblock(inputs)
        if self.use_se:
            out = self.se(out)
        return layers.add([out, inputs])
    
    def get_config(self):
        return {"kernelsize": self.kernelsize, "use_se": self.use_se}


class BasicBlockall(layers.Layer):
    """多尺度残差块"""
    def __init__(self, use_se=False, **kwargs):
        super(BasicBlockall, self).__init__(**kwargs)
        self.use_se = use_se
        
        self.bblock3 = Sequential([
            Res_BasicBlock(3, use_se=use_se),
            Res_BasicBlock(3, use_se=use_se)
        ])
        self.bblock5 = Sequential([
            Res_BasicBlock(5, use_se=use_se),
            Res_BasicBlock(5, use_se=use_se)
        ])
        self.bblock7 = Sequential([
            Res_BasicBlock(7, use_se=use_se),
            Res_BasicBlock(7, use_se=use_se)
        ])
    
    def call(self, inputs):
        return tf.concat([
            self.bblock3(inputs),
            self.bblock5(inputs),
            self.bblock7(inputs)
        ], axis=-1)
    
    def get_config(self):
        return {"use_se": self.use_se}


def build_denoise_model(input_len=512):
    """构建去噪模型"""
    inp = layers.Input(shape=(input_len, 1))
    
    # 编码器
    x = layers.Conv1D(32, 7, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 多尺度残差块
    x = BasicBlockall(use_se=True)(x)
    
    # 解码器
    x = layers.Conv1D(1, 7, padding='same')(x)
    
    return models.Model(inp, x)


# ==================== 数据加载 ====================
def load_dreams_data(data_dir, start_subject=6, end_subject=20, segment_len=512):
    """
    加载DREAMS数据集
    
    Args:
        data_dir: 数据目录
        start_subject: 起始受试者编号
        end_subject: 结束受试者编号
        segment_len: 信号分段长度
    
    Returns:
        X: 原始信号片段
        X_clean: 干净信号片段（使用带通滤波）
    """
    print(f"加载DREAMS数据集: subject{start_subject} - subject{end_subject}")
    
    all_segments = []
    all_clean_segments = []
    
    for subj_id in range(start_subject, end_subject + 1):
        edf_file = data_dir / f"subject{subj_id}.edf"
        
        if not edf_file.exists():
            print(f"  警告: {edf_file} 不存在，跳过")
            continue
        
        print(f"  加载 subject{subj_id}...")
        
        try:
            # 读取EDF文件
            raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
            
            # 获取采样率
            sfreq = raw.info['sfreq']
            
            # 获取EEG数据（取第一个通道）
            data = raw.get_data()[0]
            
            # 重采样到256Hz（如果需要）
            if sfreq != 256:
                print(f"    重采样: {sfreq}Hz -> 256Hz")
                raw.resample(256)
                data = raw.get_data()[0]
                sfreq = 256
            
            # 分段
            n_segments = len(data) // segment_len
            
            for i in range(n_segments):
                segment = data[i*segment_len:(i+1)*segment_len]
                
                # 标准化
                std = np.std(segment)
                if std > 0:
                    segment_norm = segment / std
                else:
                    continue
                
                # 创建"干净"信号：带通滤波 (0.5-30Hz)
                # 使用MNE进行滤波
                raw_segment = mne.io.RawArray(
                    segment_norm.reshape(1, -1),
                    mne.create_info(['EEG'], sfreq, ch_types='eeg'),
                    verbose=False
                )
                raw_segment.filter(0.5, 30, verbose=False)
                clean_segment = raw_segment.get_data()[0]
                
                all_segments.append(segment_norm)
                all_clean_segments.append(clean_segment)
        
        except Exception as e:
            print(f"  错误: 加载 subject{subj_id} 失败: {e}")
            continue
    
    X = np.array(all_segments).reshape(-1, segment_len, 1)
    X_clean = np.array(all_clean_segments).reshape(-1, segment_len, 1)
    
    print(f"总共加载 {len(X)} 个信号片段")
    
    return X, X_clean


# ==================== 自定义损失函数 ====================
def combined_loss(y_true, y_pred):
    """组合损失：MSE + 频域损失"""
    # MSE损失
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # 频域损失（保持频谱结构）
    y_true_fft = tf.signal.rfft(tf.squeeze(y_true, axis=-1))
    y_pred_fft = tf.signal.rfft(tf.squeeze(y_pred, axis=-1))
    
    # 幅度谱损失
    true_mag = tf.abs(y_true_fft)
    pred_mag = tf.abs(y_pred_fft)
    spectral_loss = tf.reduce_mean(tf.square(true_mag - pred_mag))
    
    return mse_loss + 0.1 * spectral_loss


# ==================== 训练函数 ====================
def train_model():
    """训练去噪模型"""
    print("=" * 60)
    print("EEG去噪模型训练")
    print("=" * 60)
    
    # 配置
    DATA_DIR = PROJECT_ROOT / "04_原始数据" / "Raw_edf 2"
    OUTPUT_DIR = PROJECT_ROOT / "03_训练模型"
    SEGMENT_LEN = 512
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    X, X_clean = load_dreams_data(DATA_DIR, start_subject=6, end_subject=20, segment_len=SEGMENT_LEN)
    
    # 划分训练集和验证集
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, X_clean, test_size=0.2, random_state=42
    )
    
    print(f"训练集: {len(X_train)} 片段")
    print(f"验证集: {len(X_val)} 片段")
    
    # 2. 构建模型
    print("\n[2/5] 构建模型...")
    model = build_denoise_model(SEGMENT_LEN)
    model.summary()
    
    # 3. 编译模型
    print("\n[3/5] 编译模型...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=combined_loss,
        metrics=['mae']
    )
    
    # 4. 设置回调
    print("\n[4/5] 设置回调...")
    callback_list = [
        callbacks.ModelCheckpoint(
            str(OUTPUT_DIR / "denoise_model_dreams_best.h5"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.CSVLogger(
            str(OUTPUT_DIR / "training_log.csv"),
            separator=',',
            append=False
        )
    ]
    
    # 5. 训练模型
    print("\n[5/5] 开始训练...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callback_list,
        verbose=1
    )
    
    # 保存最终模型
    final_model_path = OUTPUT_DIR / "denoise_model_dreams_final.h5"
    model.save(str(final_model_path))
    print(f"\n最终模型已保存: {final_model_path}")
    
    # 绘制训练曲线
    plot_training_history(history, OUTPUT_DIR)
    
    # 评估模型
    evaluate_model(model, X_val, y_val, OUTPUT_DIR)
    
    return model, history


def plot_training_history(history, output_dir):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    axes[0].plot(history.history['loss'], label='训练损失')
    axes[0].plot(history.history['val_loss'], label='验证损失')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('训练损失曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE曲线
    axes[1].plot(history.history['mae'], label='训练MAE')
    axes[1].plot(history.history['val_mae'], label='验证MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('训练MAE曲线')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存: {output_dir / 'training_curves.png'}")


def evaluate_model(model, X_val, y_val, output_dir):
    """评估模型"""
    print("\n" + "=" * 40)
    print("模型评估")
    print("=" * 40)
    
    # 预测
    y_pred = model.predict(X_val, verbose=0)
    
    # 计算指标
    mse = np.mean((y_val - y_pred) ** 2)
    mae = np.mean(np.abs(y_val - y_pred))
    
    # 计算相关系数
    correlations = []
    for i in range(len(y_val)):
        corr = np.corrcoef(y_val[i].flatten(), y_pred[i].flatten())[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    avg_corr = np.mean(correlations)
    
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"平均相关系数: {avg_corr:.4f}")
    
    # 可视化几个样本
    visualize_samples(X_val, y_val, y_pred, output_dir)
    
    return {'mse': mse, 'mae': mae, 'correlation': avg_corr}


def visualize_samples(X_val, y_val, y_pred, output_dir, n_samples=5):
    """可视化去噪效果"""
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 3*n_samples))
    
    time_axis = np.arange(X_val.shape[1]) / 256
    
    for i in range(n_samples):
        # 原始信号
        axes[i, 0].plot(time_axis, X_val[i].flatten(), 'b-', linewidth=0.5)
        axes[i, 0].set_title(f'样本{i+1} - 原始信号')
        axes[i, 0].set_ylabel('幅度')
        axes[i, 0].grid(True, alpha=0.3)
        
        # 目标信号（带通滤波）
        axes[i, 1].plot(time_axis, y_val[i].flatten(), 'g-', linewidth=0.5)
        axes[i, 1].set_title(f'样本{i+1} - 目标信号')
        axes[i, 1].grid(True, alpha=0.3)
        
        # 预测信号
        axes[i, 2].plot(time_axis, y_pred[i].flatten(), 'r-', linewidth=0.5)
        axes[i, 2].set_title(f'样本{i+1} - 去噪结果')
        axes[i, 2].grid(True, alpha=0.3)
    
    axes[-1, 0].set_xlabel('时间 (s)')
    axes[-1, 1].set_xlabel('时间 (s)')
    axes[-1, 2].set_xlabel('时间 (s)')
    
    plt.tight_layout()
    plt.savefig(output_dir / "denoising_samples.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"去噪样本可视化已保存: {output_dir / 'denoising_samples.png'}")


if __name__ == "__main__":
    train_model()
