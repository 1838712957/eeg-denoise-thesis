"""
数据量缩放实验 - 测试不同训练数据量对模型性能的影响
用于论文4.4节：数据量测试
"""
import os, numpy as np, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import layers, Model
from pathlib import Path

print("="*60)
print("数据量缩放实验 - 测试训练数据量对模型性能的影响")
print("="*60)

# 使用合成EEG数据
print("\n[1] 生成合成EEG数据...")
all_clean = []
all_noisy = []

np.random.seed(42)
n_samples = 2000  # 生成2000个30秒epoch

for i in range(n_samples):
    t = np.linspace(0, 30, 3000)
    
    # 生成模拟睡眠EEG信号（包含Delta, Theta, Alpha, Beta波）
    epoch = np.zeros(3000)
    
    # Delta波 (0.5-4Hz) - 睡眠主要波率
    epoch += np.sin(2*np.pi*1.5*t) * 2.0
    epoch += np.sin(2*np.pi*2.5*t) * 1.5
    
    # Theta波 (4-8Hz)
    epoch += np.sin(2*np.pi*5*t) * 0.8
    
    # Alpha波 (8-12Hz)
    epoch += np.sin(2*np.pi*10*t) * 0.5
    
    # Beta波 (12-30Hz)
    epoch += np.sin(2*np.pi*20*t) * 0.2
    
    # 添加一些随机变化
    epoch += np.random.randn(3000) * 0.3
    
    # 归一化
    epoch = (epoch - np.mean(epoch)) / (np.std(epoch) + 1e-8)
    
    clean = epoch.copy()
    noisy = epoch.copy()
    
    # 添加高斯噪声
    noise_level = np.std(epoch) * 0.3
    noisy += np.random.randn(len(epoch)) * noise_level
    
    # 添加肌电伪迹 (EMG) - 简化版
    if np.random.rand() > 0.5:
        emg = np.random.randn(200) * 0.5
        start = 500
        noisy[start:start+200] += emg
    
    # 添加眼动伪迹 (EOG) - 简化版
    if np.random.rand() > 0.5:
        eog = np.random.randn(100) * np.std(epoch) * 1.0
        start = 1000
        noisy[start:start+100] += eog
    
    all_clean.append(clean)
    all_noisy.append(noisy)

print(f"  生成 {len(all_clean)} 个样本")

X_noisy = np.array(all_noisy)
X_clean = np.array(all_clean)

# 打乱数据
idx = np.random.permutation(len(X_noisy))
X_noisy = X_noisy[idx]
X_clean = X_clean[idx]

print(f"  总样本数: {len(X_noisy)}")

# 划分训练集和验证集 (90%训练, 10%验证)
n_train = int(len(X_noisy) * 0.9)
X_train_noisy = X_noisy[:n_train]
X_train_clean = X_clean[:n_train]
X_val_noisy = X_noisy[n_train:]
X_val_clean = X_clean[n_train:]

print(f"  训练集: {len(X_train_noisy)}, 验证集: {len(X_val_noisy)}")

# 添加通道维度
X_train_noisy = X_train_noisy.reshape(-1, 3000, 1)
X_train_clean = X_train_clean.reshape(-1, 3000, 1)
X_val_noisy = X_val_noisy.reshape(-1, 3000, 1)
X_val_clean = X_val_clean.reshape(-1, 3000, 1)

# 构建简化版U-Net模型（加快训练速度）
def build_simple_unet(input_shape=(3000, 1)):
    inputs = layers.Input(shape=input_shape)
    
    # 编码器
    x = layers.Conv1D(32, 7, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(32, 7, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x1 = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x1)
    
    x = layers.Conv1D(64, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(64, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x2 = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x2)
    
    # 瓶颈
    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # 解码器
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(64, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Add()([x, x2])
    
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32, 7, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Add()([x, x1])
    
    outputs = layers.Conv1D(1, 7, padding='same')(x)
    
    model = Model(inputs, outputs)
    return model

# 计算RRMSE
def compute_rrmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2)) / np.sqrt(np.mean(y_true**2)) * 100

# 计算相关系数
def compute_cc(y_true, y_pred):
    return np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]

# 数据量比例
data_ratios = [0.1, 0.3, 0.5, 1.0]

results = []

print("\n[2] 开始数据量缩放实验...")
print("-" * 60)

for ratio in data_ratios:
    n_samples = int(len(X_train_noisy) * ratio)
    print(f"\n>>> 训练数据比例: {ratio*100:.0f}% (样本数: {n_samples})")
    
    # 取对应比例的训练数据
    X_sub_noisy = X_train_noisy[:n_samples]
    X_sub_clean = X_train_clean[:n_samples]
    
    # 构建并训练模型
    model = build_simple_unet()
    model.compile(optimizer='adam', loss='mse')
    
    # 训练（减少epoch数以加快速度）
    history = model.fit(
        X_sub_noisy, X_sub_clean,
        validation_data=(X_val_noisy, X_val_clean),
        epochs=20,
        batch_size=16,
        verbose=0
    )
    
    # 评估
    val_loss = history.history['val_loss'][-1]
    val_pred = model.predict(X_val_noisy, verbose=0)
    
    rrmse = compute_rrmse(X_val_clean, val_pred)
    cc = compute_cc(X_val_clean, val_pred)
    
    print(f"    验证损失: {val_loss:.6f}")
    print(f"    RRMSE: {rrmse:.2f}%")
    print(f"    CC: {cc:.4f}")
    
    results.append({
        'ratio': ratio,
        'n_samples': n_samples,
        'val_loss': val_loss,
        'rrmse': rrmse,
        'cc': cc
    })

# 保存结果
print("\n" + "="*60)
print("实验结果汇总")
print("="*60)
print(f"{'数据比例':<10} {'样本数':<10} {'验证损失':<12} {'RRMSE(%)':<12} {'CC':<10}")
print("-"*60)
for r in results:
    print(f"{r['ratio']*100:>5.0f}%    {r['n_samples']:<10} {r['val_loss']:<12.6f} {r['rrmse']:<12.2f} {r['cc']:<10.4f}")

# 保存到文件
result_file = '05_处理结果/数据量测试结果.txt'
with open(result_file, 'w', encoding='utf-8') as f:
    f.write("数据量缩放实验结果\n")
    f.write("="*60 + "\n\n")
    f.write(f"{'数据比例':<10} {'样本数':<10} {'验证损失':<12} {'RRMSE(%)':<12} {'CC':<10}\n")
    f.write("-"*60 + "\n")
    for r in results:
        f.write(f"{r['ratio']*100:>5.0f}%    {r['n_samples']:<10} {r['val_loss']:<12.6f} {r['rrmse']:<12.2f} {r['cc']:<10.4f}\n")

print(f"\n结果已保存到: {result_file}")