"""
消融实验脚本
对比不同模型变体的去噪效果和下游任务保护能力

模型变体：
1. Baseline: 基础1D CNN，无残差、无多尺度、无注意力
2. V4 w/o SE: 移除SE注意力机制
3. V4 Single Scale: 单一尺度（只用kernel_size=3）
4. V4 Complete: 完整V4模型

评价指标：
- 时域: RRMSE, CC
- 下游任务: N3召回率, 整体准确率
- 频域: Delta能量保持率
"""
import os
import numpy as np
import warnings
from pathlib import Path
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
import mne
from scipy import signal

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
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


# ================= 评价指标 =================

def calculate_rrmse(clean, denoised):
    """计算相对均方根误差"""
    mse = np.mean((clean - denoised) ** 2)
    power = np.mean(clean ** 2)
    if power == 0:
        return 0
    return np.sqrt(mse / power) * 100


def calculate_cc(clean, denoised):
    """计算皮尔逊相关系数"""
    clean_flat = clean.flatten()
    denoised_flat = denoised.flatten()
    if np.std(clean_flat) == 0 or np.std(denoised_flat) == 0:
        return 0
    return np.corrcoef(clean_flat, denoised_flat)[0, 1]


def calculate_delta_power(data, sfreq=100):
    """计算Delta频段(0.5-4Hz)功率"""
    freqs, psd = signal.welch(data, sfreq, nperseg=min(256, len(data)))
    delta_mask = (freqs >= 0.5) & (freqs <= 4)
    delta_power = np.trapz(psd[delta_mask], freqs[delta_mask])
    return delta_power


def calculate_delta_preservation(original, denoised, sfreq=100):
    """计算Delta能量保持率"""
    orig_delta = calculate_delta_power(original, sfreq)
    denoised_delta = calculate_delta_power(denoised, sfreq)
    if orig_delta == 0:
        return 100
    return (denoised_delta / orig_delta) * 100


# ================= 睡眠分期 =================

TARGET_SFREQ = 100
EPOCH_LEN = 3000

def run_deepsleepnet_staging(data, sfreq, model):
    """使用DeepSleepNet进行睡眠分期"""
    if sfreq != TARGET_SFREQ:
        n_samples = int(len(data) * TARGET_SFREQ / sfreq)
        data = np.interp(np.linspace(0, len(data), n_samples), np.arange(len(data)), data)
    
    n_epochs = len(data) // EPOCH_LEN
    if n_epochs == 0:
        return np.array([])
    
    epochs = []
    for i in range(n_epochs):
        epoch = data[i*EPOCH_LEN : (i+1)*EPOCH_LEN]
        if len(epoch) == EPOCH_LEN:
            epochs.append(epoch)
    
    if len(epochs) == 0:
        return np.array([])
    
    epochs = np.array(epochs)
    for i in range(len(epochs)):
        std = np.std(epochs[i])
        if std > 0:
            epochs[i] = epochs[i] / std
    
    epochs = epochs.reshape(-1, EPOCH_LEN, 1)
    predictions = model.predict(epochs, batch_size=32, verbose=0)
    return np.argmax(predictions, axis=1)


def apply_denoising(data, model, segment_len=512):
    """应用去噪模型"""
    n_segments = len(data) // segment_len
    if n_segments == 0:
        return data
    
    batch_in = []
    scales = []
    
    for i in range(n_segments):
        seg = data[i*segment_len : (i+1)*segment_len]
        std = np.std(seg) if np.std(seg) != 0 else 1.0
        batch_in.append(seg / std)
        scales.append(std)
    
    input_tensor = np.array(batch_in).reshape(-1, segment_len, 1)
    predictions = model.predict(input_tensor, batch_size=128, verbose=0)
    predictions = np.squeeze(predictions)
    
    data_clean = np.array([predictions[i].flatten() * scales[i] for i in range(n_segments)]).flatten()
    return data_clean


def parse_dreams_hypno(hypno_path):
    """解析DREAMS数据集的标签文件"""
    stages = []
    with open(hypno_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('[') and not line.startswith('#'):
                try:
                    label = int(line)
                    if label == 5:
                        stages.append(-1)
                    elif 0 <= label <= 4:
                        stages.append(label)
                    else:
                        stages.append(-1)
                except ValueError:
                    continue
    return np.array(stages)


# ================= 主程序 =================

def main():
    print("=" * 80)
    print("消融实验：模型变体对比分析")
    print("=" * 80)
    
    # 加载DeepSleepNet裁判模型
    print("\n[1] 加载DeepSleepNet裁判模型...")
    deepsleepnet = tf.keras.models.load_model(
        str(PROJECT_ROOT / "03_训练模型" / "DeepSleepNet裁判模型.h5"), 
        compile=False
    )
    print("[OK] DeepSleepNet加载成功")
    
    # 定义模型配置
    model_configs = [
        {'name': 'Baseline', 'build_fn': build_baseline_model, 'weights': 'Baseline.h5'},
        {'name': 'V4_wo_SE', 'build_fn': build_model_without_se, 'weights': 'V4_wo_SE.h5'},
        {'name': 'V4_Single_Scale', 'build_fn': build_model_single_scale, 'weights': 'V4_Single_Scale.h5'},
        {'name': 'V4_Complete', 'build_fn': build_model_complete, 'weights': 'V4_Complete.h5'}
    ]
    
    # 加载所有模型
    print("\n[2] 加载所有训练好的模型...")
    models_dict = {}
    for config in model_configs:
        model_path = PROJECT_ROOT / "03_训练模型" / config['weights']
        if model_path.exists():
            print(f"  加载 {config['name']}...")
            model = config['build_fn']()
            model.load_weights(str(model_path))
            models_dict[config['name']] = model
            print(f"    [OK] {config['name']} 加载成功")
        else:
            print(f"    [跳过] {config['name']} 权重文件不存在")
    
    print(f"\n  共加载 {len(models_dict)} 个模型")
    
    # 数据路径
    data_dir = PROJECT_ROOT / "04_原始数据" / "Raw_edf 2"
    edf_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.edf') and f.startswith('subject')])
    
    # 收集所有结果
    all_results = {name: {'rrmse': [], 'cc': [], 'delta_pres': [], 'accuracy': [], 'n3_recall': []} 
                   for name in models_dict.keys()}
    all_results['Original'] = {'accuracy': [], 'n3_recall': []}
    
    print(f"\n[3] 处理 {len(edf_files)} 个EDF文件...")
    
    for edf_file in edf_files[:5]:  # 限制处理5个文件以加快速度
        print(f"\n  处理 {edf_file}...")
        try:
            raw = mne.io.read_raw_edf(str(data_dir / edf_file), preload=True, verbose=False)
            sfreq = raw.info['sfreq']
            
            # 获取EEG通道 - 使用与batch_dreams_staging相同的方法
            eeg_picks = mne.pick_channels(raw.info['ch_names'], include=['EEG'], ordered=False)
            if len(eeg_picks) == 0:
                eeg_picks = [0]
            data = raw.get_data(picks=eeg_picks)[0]
            
            # 获取标签
            subj_num = edf_file.replace('subject', '').replace('.edf', '')
            hypno_file = f"HypnogramAASM_subject{subj_num}.txt"
            hypno_path = data_dir / hypno_file
            if not hypno_path.exists():
                continue
            gt_stages = parse_dreams_hypno(hypno_path)
            
            # 原始数据分期
            orig_stages = run_deepsleepnet_staging(data, sfreq, deepsleepnet)
            min_len = min(len(orig_stages), len(gt_stages))
            if min_len == 0:
                continue
            
            valid_mask = gt_stages[:min_len] != -1
            orig_acc = accuracy_score(gt_stages[:min_len][valid_mask], orig_stages[:min_len][valid_mask])
            all_results['Original']['accuracy'].append(orig_acc)
            
            # N3召回率
            n3_mask = gt_stages[:min_len] == 3
            if np.any(n3_mask):
                n3_recall = np.mean(orig_stages[:min_len][n3_mask] == 3)
                all_results['Original']['n3_recall'].append(n3_recall)
            
            # 对每个模型进行评估
            for model_name, model in models_dict.items():
                denoised = apply_denoising(data, model)
                
                # 时域指标
                min_len_data = min(len(data), len(denoised))
                rrmse = calculate_rrmse(data[:min_len_data], denoised[:min_len_data])
                cc = calculate_cc(data[:min_len_data], denoised[:min_len_data])
                delta_pres = calculate_delta_preservation(data[:min_len_data], denoised[:min_len_data], sfreq)
                
                all_results[model_name]['rrmse'].append(rrmse)
                all_results[model_name]['cc'].append(cc)
                all_results[model_name]['delta_pres'].append(delta_pres)
                
                # 分期指标
                denoised_stages = run_deepsleepnet_staging(denoised, sfreq, deepsleepnet)
                min_len2 = min(len(denoised_stages), len(gt_stages))
                if min_len2 > 0:
                    valid_mask2 = gt_stages[:min_len2] != -1
                    acc = accuracy_score(gt_stages[:min_len2][valid_mask2], denoised_stages[:min_len2][valid_mask2])
                    all_results[model_name]['accuracy'].append(acc)
                    
                    n3_mask2 = gt_stages[:min_len2] == 3
                    if np.any(n3_mask2):
                        n3_recall = np.mean(denoised_stages[:min_len2][n3_mask2] == 3)
                        all_results[model_name]['n3_recall'].append(n3_recall)
            
            print(f"    [OK] 完成")
            
        except Exception as e:
            print(f"    [错误] {e}")
            continue
    
    # 计算平均值并输出结果
    print("\n" + "=" * 80)
    print("消融实验结果")
    print("=" * 80)
    
    results_lines = []
    results_lines.append("\n模型变体对比表:")
    results_lines.append("-" * 80)
    results_lines.append(f"{'模型':<20} {'RRMSE(%)':<12} {'CC':<10} {'Delta保持(%)':<15} {'准确率(%)':<12} {'N3召回(%)':<12}")
    results_lines.append("-" * 80)
    
    # 原始数据基准
    orig_acc = np.mean(all_results['Original']['accuracy']) * 100 if all_results['Original']['accuracy'] else 0
    orig_n3 = np.mean(all_results['Original']['n3_recall']) * 100 if all_results['Original']['n3_recall'] else 0
    results_lines.append(f"{'Original (基准)':<20} {'-':<12} {'-':<10} {'-':<15} {orig_acc:<12.2f} {orig_n3:<12.2f}")
    
    # 各模型结果
    for model_name in ['Baseline', 'V4_wo_SE', 'V4_Single_Scale', 'V4_Complete']:
        if model_name in all_results:
            r = all_results[model_name]
            rrmse = np.mean(r['rrmse']) if r['rrmse'] else 0
            cc = np.mean(r['cc']) if r['cc'] else 0
            delta = np.mean(r['delta_pres']) if r['delta_pres'] else 0
            acc = np.mean(r['accuracy']) * 100 if r['accuracy'] else 0
            n3 = np.mean(r['n3_recall']) * 100 if r['n3_recall'] else 0
            results_lines.append(f"{model_name:<20} {rrmse:<12.2f} {cc:<10.4f} {delta:<15.2f} {acc:<12.2f} {n3:<12.2f}")
    
    results_lines.append("-" * 80)
    
    # 打印和保存
    result_text = "\n".join(results_lines)
    print(result_text)
    
    with open(OUTPUT_DIR / "ablation_results.txt", 'w', encoding='utf-8') as f:
        f.write("消融实验结果\n")
        f.write("=" * 80 + "\n")
        f.write(result_text)
    
    print(f"\n结果已保存到: {OUTPUT_DIR / 'ablation_results.txt'}")


if __name__ == "__main__":
    main()
