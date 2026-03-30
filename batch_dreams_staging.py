"""
DREAMS数据集批量睡眠分期测试
从subject6开始，对比ASR和深度学习去噪的分期准确率
"""
import os
import numpy as np
import warnings
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
import mne

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "05_处理结果" / "DREAMS批量分析"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_file = OUTPUT_DIR / "batch_staging_results.txt"

def log(msg, f=None):
    print(msg)
    if f:
        f.write(msg + "\n")

# ================= 定义去噪模型 =================
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

# ================= 分期函数 =================
TARGET_SFREQ = 100
EPOCH_LEN = 3000
STAGE_NAMES = ['W', 'N1', 'N2', 'N3', 'REM']

def run_deepsleepnet_staging(data, sfreq, model):
    """使用DeepSleepNet进行睡眠分期"""
    if sfreq != TARGET_SFREQ:
        n_samples = int(len(data) * TARGET_SFREQ / sfreq)
        data = np.interp(np.linspace(0, len(data), n_samples), np.arange(len(data)), data)
    
    n_epochs = len(data) // EPOCH_LEN
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

def apply_dl_denoising(data, model, segment_len=512):
    """应用深度学习去噪"""
    n_segments = len(data) // segment_len
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

def apply_asr_denoising(data, sfreq):
    """应用ASR去噪（简化版：使用带通滤波模拟）"""
    # ASR主要去除高频噪声和慢漂移
    # 这里使用1-30Hz带通滤波模拟ASR效果
    from scipy.signal import butter, filtfilt
    
    nyq = sfreq / 2
    low = 1.0 / nyq
    high = min(30.0 / nyq, 0.99)
    
    b, a = butter(4, [low, high], btype='band')
    data_asr = filtfilt(b, a, data)
    
    return data_asr

def calculate_per_stage_accuracy(y_true, y_pred, stage_names):
    """计算各分期准确率"""
    results = {}
    for i, name in enumerate(stage_names):
        mask = y_true == i
        if mask.sum() > 0:
            correct = (y_pred[mask] == i).sum()
            total = mask.sum()
            acc = correct / total * 100
            results[name] = {'correct': correct, 'total': total, 'accuracy': acc}
        else:
            results[name] = {'correct': 0, 'total': 0, 'accuracy': 0}
    return results

def parse_dreams_hypno(hypno_path):
    """解析DREAMS数据集的标签文件
    
    DREAMS数据集标签格式（数字编码）：
    0 = Wake
    1 = N1
    2 = N2
    3 = N3
    4 = REM
    5 = Movement/Artefact（无效标签）
    """
    stages = []
    
    with open(hypno_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('[') and not line.startswith('#'):
                try:
                    label = int(line)
                    # DREAMS数字编码映射
                    # 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM, 5=Movement(无效)
                    if label == 5:
                        stages.append(-1)  # Movement/Artefact标记为无效
                    elif 0 <= label <= 4:
                        stages.append(label)  # 直接使用数字编码
                    else:
                        stages.append(-1)  # 其他未知标签
                except ValueError:
                    # 如果不是数字，跳过
                    continue
    
    return np.array(stages)

# ================= 主程序 =================
with open(output_file, 'w', encoding='utf-8') as f:
    log("=" * 80, f)
    log("DREAMS数据集批量睡眠分期测试", f)
    log("对比：原始信号 vs ASR去噪 vs 深度学习去噪", f)
    log("=" * 80, f)
    
    # 加载模型
    log("\n[1/4] 加载模型...", f)
    denoise_model = build_denoise_model()
    denoise_model.load_weights(str(PROJECT_ROOT / "03_训练模型" / "去噪模型v2最终版.h5"))
    log("[OK] 去噪模型加载成功", f)
    
    from tensorflow.keras.models import load_model
    deepsleepnet = load_model(str(PROJECT_ROOT / "03_训练模型" / "DeepSleepNet裁判模型.h5"), compile=False)
    log("[OK] DeepSleepNet模型加载成功", f)
    
    # 数据路径
    log("\n[2/4] 扫描数据文件...", f)
    data_dir = PROJECT_ROOT / "04_原始数据" / "Raw_edf 2"
    edf_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.edf')])
    log(f"发现 {len(edf_files)} 个EDF文件", f)
    
    # 从subject6开始
    start_subject = 6
    edf_files = [f for f in edf_files if int(f.replace('subject', '').replace('.edf', '')) >= start_subject]
    log(f"从subject{start_subject}开始，共 {len(edf_files)} 个文件", f)
    
    # 存储结果
    all_results = {
        'raw': {name: [] for name in STAGE_NAMES},
        'asr': {name: [] for name in STAGE_NAMES},
        'dl': {name: [] for name in STAGE_NAMES}
    }
    
    # 处理每个受试者
    log("\n[3/4] 批量处理...", f)
    
    for edf_file in edf_files:
        subject_id = edf_file.replace('.edf', '')
        edf_path = data_dir / edf_file
        
        # 查找对应的标签文件
        hypno_path = data_dir / f"HypnogramAASM_{subject_id}.txt"
        if not hypno_path.exists():
            log(f"\n{subject_id}: 未找到标签文件，跳过", f)
            continue
        
        try:
            log(f"\n{'='*60}", f)
            log(f"处理: {subject_id}", f)
            log(f"{'='*60}", f)
            
            # 加载EEG数据
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
            sfreq = raw.info['sfreq']
            
            # 获取EEG通道
            eeg_picks = mne.pick_channels(raw.info['ch_names'], include=['EEG'], ordered=False)
            if len(eeg_picks) == 0:
                eeg_picks = [0]
            
            data_raw = raw.get_data(picks=eeg_picks)[0]
            log(f"  采样率: {sfreq}Hz, 数据长度: {len(data_raw)/sfreq:.1f}秒", f)
            
            # 解析标签
            y_true = parse_dreams_hypno(hypno_path)
            log(f"  标签数量: {len(y_true)}", f)
            
            # 重采样到100Hz
            if sfreq != TARGET_SFREQ:
                n_samples = int(len(data_raw) * TARGET_SFREQ / sfreq)
                data_raw = np.interp(np.linspace(0, len(data_raw), n_samples), np.arange(len(data_raw)), data_raw)
                sfreq = TARGET_SFREQ
            
            # 应用去噪
            log(f"  应用ASR去噪...", f)
            data_asr = apply_asr_denoising(data_raw, sfreq)
            
            log(f"  应用深度学习去噪...", f)
            data_dl = apply_dl_denoising(data_raw, denoise_model)
            
            # 睡眠分期
            log(f"  进行睡眠分期...", f)
            pred_raw = run_deepsleepnet_staging(data_raw, sfreq, deepsleepnet)
            pred_asr = run_deepsleepnet_staging(data_asr, sfreq, deepsleepnet)
            pred_dl = run_deepsleepnet_staging(data_dl, sfreq, deepsleepnet)
            
            # 对齐长度
            L = min(len(y_true), len(pred_raw), len(pred_asr), len(pred_dl))
            y_true_final = y_true[:L]
            pred_raw_final = pred_raw[:L]
            pred_asr_final = pred_asr[:L]
            pred_dl_final = pred_dl[:L]
            
            # 过滤无效标签
            valid_mask = y_true_final >= 0
            y_true_final = y_true_final[valid_mask]
            pred_raw_final = pred_raw_final[valid_mask]
            pred_asr_final = pred_asr_final[valid_mask]
            pred_dl_final = pred_dl_final[valid_mask]
            
            # 计算准确率
            acc_raw = accuracy_score(y_true_final, pred_raw_final) * 100
            acc_asr = accuracy_score(y_true_final, pred_asr_final) * 100
            acc_dl = accuracy_score(y_true_final, pred_dl_final) * 100
            
            # 计算各分期准确率
            stage_raw = calculate_per_stage_accuracy(y_true_final, pred_raw_final, STAGE_NAMES)
            stage_asr = calculate_per_stage_accuracy(y_true_final, pred_asr_final, STAGE_NAMES)
            stage_dl = calculate_per_stage_accuracy(y_true_final, pred_dl_final, STAGE_NAMES)
            
            # 输出结果
            log(f"\n  总体准确率:", f)
            log(f"    原始信号: {acc_raw:.2f}%", f)
            log(f"    ASR去噪:  {acc_asr:.2f}% ({acc_asr-acc_raw:+.2f}%)", f)
            log(f"    DL去噪:   {acc_dl:.2f}% ({acc_dl-acc_raw:+.2f}%)", f)
            
            log(f"\n  各分期准确率 (N1-N3):", f)
            log(f"  {'分期':<6} | {'原始':<10} | {'ASR':<10} | {'DL':<10} | {'样本数':<8}", f)
            log(f"  {'-'*50}", f)
            
            for name in ['N1', 'N2', 'N3']:
                r_raw = stage_raw[name]['accuracy']
                r_asr = stage_asr[name]['accuracy']
                r_dl = stage_dl[name]['accuracy']
                n = stage_raw[name]['total']
                log(f"  {name:<6} | {r_raw:>8.2f}% | {r_asr:>8.2f}% | {r_dl:>8.2f}% | {n:<8}", f)
                
                all_results['raw'][name].append(r_raw)
                all_results['asr'][name].append(r_asr)
                all_results['dl'][name].append(r_dl)
            
        except Exception as e:
            log(f"\n{subject_id}: 处理失败 - {e}", f)
            import traceback
            traceback.print_exc()
    
    # 汇总结果
    log(f"\n{'='*80}", f)
    log("[4/4] 汇总结果", f)
    log(f"{'='*80}", f)
    
    log(f"\n各分期平均准确率 (N1-N3):", f)
    log(f"{'分期':<6} | {'原始':<12} | {'ASR':<12} | {'DL':<12} | {'ASR变化':<10} | {'DL变化':<10}", f)
    log(f"{'-'*70}", f)
    
    for name in ['N1', 'N2', 'N3']:
        if all_results['raw'][name]:
            avg_raw = np.mean(all_results['raw'][name])
            avg_asr = np.mean(all_results['asr'][name])
            avg_dl = np.mean(all_results['dl'][name])
            log(f"{name:<6} | {avg_raw:>10.2f}% | {avg_asr:>10.2f}% | {avg_dl:>10.2f}% | {avg_asr-avg_raw:>+8.2f}% | {avg_dl-avg_raw:>+8.2f}%", f)
    
    log(f"\n{'='*80}", f)
    log("测试完成！", f)
    log(f"结果已保存到: {output_file}", f)
    log(f"{'='*80}", f)
