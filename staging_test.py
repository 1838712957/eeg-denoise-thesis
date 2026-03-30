"""
睡眠分期准确率测试脚本
测试去噪前后睡眠分期准确率变化
"""
import os
import glob
import numpy as np
import warnings
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

output_file = Path(__file__).resolve().parent / "staging_result.txt"

def log(msg, f):
    try:
        print(msg)
    except:
        pass
    f.write(msg + "\n")

with open(output_file, 'w', encoding='utf-8') as f:
    log("=" * 70, f)
    log("睡眠分期准确率测试", f)
    log("=" * 70, f)
    
    # 1. 导入依赖
    log("\n[1/6] 导入依赖...", f)
    import tensorflow as tf
    from tensorflow.keras import layers, models, Sequential
    import mne
    import yasa
    from sklearn.metrics import accuracy_score
    log(f"[OK] TensorFlow: {tf.__version__}", f)
    log(f"[OK] MNE: {mne.__version__}", f)
    log(f"[OK] YASA: {yasa.__version__}", f)
    
    # 2. 定义去噪模型
    log("\n[2/6] 定义去噪模型...", f)
    
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

    log("[OK] 去噪模型定义完成", f)
    
    # 3. 加载去噪模型
    log("\n[3/6] 加载去噪模型...", f)
    PROJECT_ROOT = Path(__file__).resolve().parent
    model_path = PROJECT_ROOT / "03_训练模型" / "去噪模型v2最终版.h5"
    
    denoise_model = build_model()
    denoise_model.load_weights(str(model_path))
    log("[OK] 去噪模型加载成功", f)
    
    # 4. 定义YASA分期函数
    log("\n[4/6] 定义分期函数...", f)
    
    TARGET_SFREQ = 256
    STAGE_MAPPING = {
        'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2,
        'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4,
        'Movement time': -1, 'Sleep stage ?': -1
    }
    
    def run_yasa_staging(data, sfreq):
        """使用YASA进行睡眠分期"""
        info = mne.create_info(['EEG'], sfreq, ['eeg'])
        raw_tmp = mne.io.RawArray(data.reshape(1, -1), info, verbose=False)
        sls = yasa.SleepStaging(raw_tmp, eeg_name="EEG")
        hypno_pred = sls.predict()
        return yasa.hypno_str_to_int(hypno_pred)
    
    log("[OK] 分期函数定义完成", f)
    
    # 5. 加载数据
    log("\n[5/6] 加载EEG数据...", f)
    raw_dir = PROJECT_ROOT / "04_原始数据" / "Raw_edf"
    raw_files = glob.glob(str(raw_dir / "*PSG.edf"))
    log(f"发现 {len(raw_files)} 个受试者数据", f)
    
    # 6. 测试分期准确率
    log("\n[6/6] 测试分期准确率...", f)
    log("-" * 70, f)
    log(f"{'受试者':<10} | {'原始准确率':<12} | {'去噪后准确率':<12} | {'变化':<10}", f)
    log("-" * 70, f)
    
    all_results = []
    
    for raw_path in raw_files[:5]:  # 测试前5个
        fname = os.path.basename(raw_path)
        subject_id = fname.split('-')[0]
        
        # 查找标签文件
        hypno_files = glob.glob(str(raw_dir / f"{subject_id}*Hypnogram.edf"))
        if not hypno_files:
            log(f"{subject_id:<10} | 未找到标签文件", f)
            continue
        
        try:
            # 读取EEG数据
            raw = mne.io.read_raw_edf(raw_path, preload=True, verbose=False)
            if raw.info['sfreq'] != TARGET_SFREQ:
                raw.resample(TARGET_SFREQ)
            
            eeg_picks = mne.pick_channels(raw.info['ch_names'], include=['EEG Fpz-Cz'])
            if len(eeg_picks) == 0:
                eeg_picks = [0]
            data_raw = raw.get_data(picks=eeg_picks)[0]
            
            # 读取真值标签
            annot = mne.read_annotations(hypno_files[0])
            raw.set_annotations(annot, emit_warning=False)
            events, _ = mne.events_from_annotations(raw, event_id=STAGE_MAPPING, chunk_duration=30., verbose=False)
            y_true = events[:, 2]
            
            # 去噪处理
            SEGMENT_LEN = 512
            n_segments = len(data_raw) // SEGMENT_LEN
            
            batch_in = []
            scales = []
            for i in range(n_segments):
                seg = data_raw[i*SEGMENT_LEN : (i+1)*SEGMENT_LEN]
                std = np.std(seg) if np.std(seg) != 0 else 1.0
                batch_in.append(seg / std)
                scales.append(std)
            
            input_tensor = np.array(batch_in).reshape(-1, SEGMENT_LEN, 1)
            predictions = denoise_model.predict(input_tensor, batch_size=128, verbose=0)
            predictions = np.squeeze(predictions)
            
            data_clean = np.array([predictions[i].flatten() * scales[i] for i in range(n_segments)]).flatten()
            
            # YASA分期
            L = min(len(y_true) * TARGET_SFREQ * 30, len(data_raw), len(data_clean))
            
            h_raw = run_yasa_staging(data_raw[:L], TARGET_SFREQ)
            h_clean = run_yasa_staging(data_clean[:L], TARGET_SFREQ)
            
            # 对齐长度
            L_final = min(len(y_true), len(h_raw), len(h_clean))
            
            # 计算准确率
            acc_raw = accuracy_score(y_true[:L_final], h_raw[:L_final]) * 100
            acc_clean = accuracy_score(y_true[:L_final], h_clean[:L_final]) * 100
            diff = acc_clean - acc_raw
            
            diff_str = f"{diff:+.2f}%"
            if diff > 0:
                diff_str = f"[UP] {diff_str}"
            elif diff < -5:
                diff_str = f"[DOWN] {diff_str}"
            else:
                diff_str = f"[SAME] {diff_str}"
            
            log(f"{subject_id:<10} | {acc_raw:<12.2f}% | {acc_clean:<12.2f}% | {diff_str}", f)
            
            all_results.append({
                'subject': subject_id,
                'acc_raw': acc_raw,
                'acc_clean': acc_clean,
                'diff': diff
            })
            
        except Exception as e:
            log(f"{subject_id:<10} | 处理失败: {e}", f)
    
    # 汇总
    log("-" * 70, f)
    if all_results:
        avg_raw = np.mean([r['acc_raw'] for r in all_results])
        avg_clean = np.mean([r['acc_clean'] for r in all_results])
        avg_diff = avg_clean - avg_raw
        log(f"{'平均':<10} | {avg_raw:<12.2f}% | {avg_clean:<12.2f}% | {avg_diff:+.2f}%", f)
    
    log("\n" + "=" * 70, f)
    log("[OK] 分期测试完成!", f)
    log("=" * 70, f)

print(f"\n结果已保存到: {output_file}")