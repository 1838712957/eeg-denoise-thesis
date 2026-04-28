"""
鲁棒性分析实验 - 按伪迹严重程度分组测试
按Peak-to-Peak Amplitude分成：严重伪迹、中度伪迹、干净信号
"""
import os
import numpy as np
import warnings
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "05_处理结果" / "消融实验"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================= 数据生成 =================
def generate_eeg_data(n_samples=500, length=3000, seed=42):
    """生成模拟EEG数据"""
    np.random.seed(seed)
    fs = 100
    
    clean_signals = []
    noisy_signals = []
    
    for i in range(n_samples):
        t = np.arange(length) / fs
        stage = np.random.choice(['N1', 'N2', 'N3', 'REM', 'Wake'], p=[0.1, 0.4, 0.2, 0.2, 0.1])
        
        if stage == 'N3':
            clean = np.sin(2 * np.pi * 1.5 * t) * 2.0 + np.sin(2 * np.pi * 3 * t) * 1.0
        elif stage == 'N2':
            clean = np.sin(2 * np.pi * 13 * t) * 0.5 + np.sin(2 * np.pi * 0.7 * t) * 0.8
        elif stage == 'N1':
            clean = np.sin(2 * np.pi * 6 * t) * 0.8
        elif stage == 'REM':
            clean = np.sin(2 * np.pi * 15 * t) * 0.4 + np.sin(2 * np.pi * 20 * t) * 0.3
        else:
            clean = np.sin(2 * np.pi * 10 * t) * 0.3 + np.sin(2 * np.pi * 20 * t) * 0.2
        
        clean += np.sin(2 * np.pi * 0.2 * t) * 0.3
        
        # 生成不同严重程度的噪声
        noise_level = np.random.choice(['severe', 'moderate', 'clean'], p=[0.3, 0.4, 0.3])
        
        if noise_level == 'severe':
            noise = np.sin(2 * np.pi * 0.8 * t) * 1.5 + np.random.normal(0, 0.5, length)
        elif noise_level == 'moderate':
            noise = np.sin(2 * np.pi * 0.8 * t) * 0.8 + np.random.normal(0, 0.25, length)
        else:
            noise = np.random.normal(0, 0.1, length)
        
        noisy = clean + noise
        
        clean = clean / (np.max(np.abs(clean)) + 1e-8)
        noisy = noisy / (np.max(np.abs(noisy)) + 1e-8)
        
        clean_signals.append(clean)
        noisy_signals.append(noisy)
    
    return np.array(clean_signals), np.array(noisy_signals)

# ================= 模型定义 =================
class SEBlock(layers.Layer):
    def __init__(self, channels=32, reduction=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.channels, self.reduction = channels, reduction
        self.gap = layers.GlobalAveragePooling1D()
        self.fc1 = layers.Dense(max(channels // reduction, 4), activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')
        self.reshape = layers.Reshape((1, channels))
    
    def call(self, inputs):
        x = self.gap(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.reshape(x)
        return inputs * x

def build_v4_complete(input_shape=(3000, 1)):
    """V4完整版"""
    inputs = layers.Input(shape=input_shape)
    
    conv3 = layers.Conv1D(32, 3, padding='same', activation='relu')(inputs)
    conv5 = layers.Conv1D(32, 5, padding='same', activation='relu')(inputs)
    conv7 = layers.Conv1D(32, 7, padding='same', activation='relu')(inputs)
    
    x = layers.Concatenate()([conv3, conv5, conv7])
    x = SEBlock(channels=96)(x)
    
    residual = layers.Conv1D(96, 1, padding='same')(inputs)
    x = layers.Add()([x, residual])
    x = layers.Activation('relu')(x)
    
    outputs = layers.Conv1D(1, 1, padding='same', activation='tanh')(x)
    
    return models.Model(inputs, outputs)

# ================= 评估指标 =================
def calculate_metrics(clean, denoised):
    """计算评估指标"""
    rmse = np.sqrt(np.mean((clean - denoised) ** 2))
    clean_std = np.std(clean)
    rrmse = rmse / (clean_std + 1e-8) * 100
    
    cc = np.corrcoef(clean.flatten(), denoised.flatten())[0, 1]
    
    from scipy import signal
    fs = 100
    b, a = signal.butter(4, [0.5, 4], btype='band', fs=fs)
    
    clean_delta = signal.filtfilt(b, a, clean.flatten())
    denoised_delta = signal.filtfilt(b, a, denoised.flatten())
    
    clean_energy = np.sum(clean_delta ** 2)
    denoised_energy = np.sum(denoised_delta ** 2)
    
    delta_preservation = (denoised_energy / (clean_energy + 1e-8)) * 100
    
    return rrmse, cc, delta_preservation

def calculate_ptp(signal):
    """计算峰峰值 (Peak-to-Peak)"""
    return np.max(signal) - np.min(signal)

# ================= 主实验 =================
def run_robustness_analysis():
    """运行鲁棒性分析"""
    print("=" * 60)
    print("鲁棒性分析实验 - 按伪迹严重程度分组")
    print("=" * 60)
    
    # 生成测试数据
    print("\n[1/4] 生成测试数据...")
    n_test = 300
    np.random.seed(200)
    clean_test, noisy_test = generate_eeg_data(n_samples=n_test, length=3000, seed=200)
    
    clean_test = clean_test.reshape(-1, 3000, 1)
    noisy_test = noisy_test.reshape(-1, 3000, 1)
    
    print(f"  测试样本数: {n_test}")
    
    # 计算每个样本的峰峰值
    print("\n[2/4] 按峰峰值分组...")
    ptp_values = []
    for i in range(len(noisy_test)):
        ptp = calculate_ptp(noisy_test[i].flatten())
        ptp_values.append(ptp)
    
    ptp_values = np.array(ptp_values)
    
    # 按阈值分组
    # 严重伪迹: PTP > 2.5
    # 中度伪迹: 1.5 < PTP <= 2.5
    # 干净信号: PTP <= 1.5
    
    severe_idx = np.where(ptp_values > 2.5)[0]
    moderate_idx = np.where((ptp_values > 1.5) & (ptp_values <= 2.5))[0]
    clean_idx = np.where(ptp_values <= 1.5)[0]
    
    print(f"  严重伪迹组: {len(severe_idx)} 样本 (PTP > 2.5)")
    print(f"  中度伪迹组: {len(moderate_idx)} 样本 (1.5 < PTP <= 2.5)")
    print(f"  干净信号组: {len(clean_idx)} 样本 (PTP <= 1.5)")
    
    # 训练模型
    print("\n[3/4] 训练模型...")
    n_train = 500
    np.random.seed(42)
    clean_train, noisy_train = generate_eeg_data(n_samples=n_train, length=3000, seed=42)
    clean_train = clean_train.reshape(-1, 3000, 1)
    noisy_train = noisy_train.reshape(-1, 3000, 1)
    
    model = build_v4_complete()
    model.compile(optimizer='adam', loss='mse')
    model.fit(noisy_train, clean_train, epochs=10, batch_size=32, validation_split=0.15, verbose=0)
    
    # 评估各组
    print("\n[4/4] 评估各组去噪效果...")
    
    groups = {
        '严重伪迹 (PTP>2.5)': severe_idx,
        '中度伪迹 (1.5<PTP≤2.5)': moderate_idx,
        '干净信号 (PTP≤1.5)': clean_idx
    }
    
    results = {}
    
    for group_name, indices in groups.items():
        if len(indices) == 0:
            continue
            
        clean_group = clean_test[indices]
        noisy_group = noisy_test[indices]
        
        denoised = model.predict(noisy_group, verbose=0)
        rrmse, cc, delta = calculate_metrics(clean_group, denoised)
        
        results[group_name] = {
            'n_samples': len(indices),
            'RRMSE': rrmse,
            'CC': cc,
            'Delta': delta,
            'avg_ptp': np.mean(ptp_values[indices])
        }
        
        print(f"\n  {group_name}:")
        print(f"    样本数: {len(indices)}")
        print(f"    平均PTP: {np.mean(ptp_values[indices]):.2f}")
        print(f"    RRMSE: {rrmse:.2f}%")
        print(f"    CC: {cc:.4f}")
        print(f"    Delta保持率: {delta:.2f}%")
    
    # 保存结果
    output_file = OUTPUT_DIR / "robustness_analysis_results.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("鲁棒性分析实验结果\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("实验方法:\n")
        f.write("- 按Peak-to-Peak Amplitude(峰峰值)分组\n")
        f.write("- 严重伪迹: PTP > 2.5\n")
        f.write("- 中度伪迹: 1.5 < PTP <= 2.5\n")
        f.write("- 干净信号: PTP <= 1.5\n\n")
        
        f.write("实验结果:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'分组':<30} {'样本数':<10} {'RRMSE(%)':<12} {'CC':<12} {'Delta(%)':<12}\n")
        f.write("-" * 70 + "\n")
        
        for name, metrics in results.items():
            f.write(f"{name:<30} {metrics['n_samples']:<10} {metrics['RRMSE']:<12.2f} {metrics['CC']:<12.4f} {metrics['Delta']:<12.2f}\n")
        
        f.write("-" * 70 + "\n\n")
        
        f.write("分析:\n")
        
        # 比较严重伪迹和干净信号
        severe = results.get('严重伪迹 (PTP>2.5)', None)
        clean = results.get('干净信号 (PTP≤1.5)', None)
        
        if severe and clean:
            f.write(f"1. 严重伪迹 vs 干净信号:\n")
            f.write(f"   - RRMSE差异: {severe['RRMSE'] - clean['RRMSE']:.2f}%\n")
            f.write(f"   - CC差异: {severe['CC'] - clean['CC']:.4f}\n")
            f.write(f"   - Delta差异: {severe['Delta'] - clean['Delta']:.2f}%\n\n")
            
            # 判断鲁棒性
            cc_drop = (clean['CC'] - severe['CC']) / clean['CC'] * 100
            f.write(f"2. 鲁棒性评估:\n")
            f.write(f"   - CC下降率: {cc_drop:.1f}%\n")
            
            if cc_drop < 20:
                f.write(f"   - 结论: 模型在不同噪声水平下表现稳定，鲁棒性良好\n")
            else:
                f.write(f"   - 结论: 模型在严重伪迹时性能下降明显\n")
    
    print(f"\n结果已保存到: {output_file}")
    
    # 打印结果
    print("\n" + "=" * 60)
    print("鲁棒性分析结果汇总")
    print("=" * 60)
    print(f"{'分组':<30} {'样本数':<10} {'RRMSE(%)':<12} {'CC':<12} {'Delta(%)':<12}")
    print("-" * 74)
    for name, metrics in results.items():
        print(f"{name:<30} {metrics['n_samples']:<10} {metrics['RRMSE']:<12.2f} {metrics['CC']:<12.4f} {metrics['Delta']:<12.2f}")
    print("-" * 74)
    
    return results

if __name__ == "__main__":
    results = run_robustness_analysis()