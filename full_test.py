"""
完整EEG去噪测试脚本 - 使用真实数据
"""
import os
import glob
import numpy as np
import warnings
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

output_file = Path(__file__).resolve().parent / "full_test_result.txt"

def log(msg, f):
    print(msg)
    f.write(msg + "\n")

with open(output_file, 'w', encoding='utf-8') as f:
    log("=" * 60, f)
    log("EEG去噪算法完整测试 - 真实数据", f)
    log("=" * 60, f)
    
    # 1. 导入依赖
    log("\n[1/5] 导入依赖...", f)
    import tensorflow as tf
    from tensorflow.keras import layers, models, Sequential
    import mne
    log(f"✅ TensorFlow: {tf.__version__}", f)
    log(f"✅ MNE: {mne.__version__}", f)
    
    # 2. 定义模型
    log("\n[2/5] 定义模型结构...", f)
    
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
    
    # 3. 加载模型
    log("\n[3/5] 加载模型权重...", f)
    PROJECT_ROOT = Path(__file__).resolve().parent
    model_path = PROJECT_ROOT / "03_训练模型" / "去噪模型v2最终版.h5"
    
    model = build_model()
    model.load_weights(str(model_path))
    log(f"✅ 模型加载成功: {model_path.name}", f)
    
    # 4. 加载真实EEG数据
    log("\n[4/5] 加载真实EEG数据...", f)
    raw_dir = PROJECT_ROOT / "04_原始数据" / "Raw_edf"
    raw_files = glob.glob(str(raw_dir / "*PSG.edf"))
    log(f"发现 {len(raw_files)} 个受试者数据文件", f)
    
    TARGET_SFREQ = 256
    SEGMENT_LEN = 512
    
    # 5. 处理每个受试者
    log("\n[5/5] 测试去噪效果...", f)
    log("-" * 60, f)
    log(f"{'受试者':<12} | {'片段数':<8} | {'NRR (%)':<10} | {'CC':<8}", f)
    log("-" * 60, f)
    
    all_results = []
    
    for idx, raw_path in enumerate(raw_files[:5]):  # 测试前5个
        subject_id = os.path.basename(raw_path).split('-')[0]
        
        try:
            # 读取数据
            raw = mne.io.read_raw_edf(raw_path, preload=True, verbose=False)
            if raw.info['sfreq'] != TARGET_SFREQ:
                raw.resample(TARGET_SFREQ)
            
            eeg_picks = mne.pick_channels(raw.info['ch_names'], include=['EEG Fpz-Cz'])
            if len(eeg_picks) == 0:
                eeg_picks = [0]
            data_raw = raw.get_data(picks=eeg_picks)[0]
            
            # 切片
            n_segments = min(len(data_raw) // SEGMENT_LEN, 200)
            
            # 准备输入
            batch_in = []
            scales = []
            for i in range(n_segments):
                seg = data_raw[i*SEGMENT_LEN : (i+1)*SEGMENT_LEN]
                std = np.std(seg) if np.std(seg) != 0 else 1.0
                batch_in.append(seg / std)
                scales.append(std)
            
            # 推理
            input_tensor = np.array(batch_in).reshape(-1, SEGMENT_LEN, 1)
            predictions = model.predict(input_tensor, batch_size=128, verbose=0)
            predictions = np.squeeze(predictions)
            
            # 还原信号
            data_clean = np.array([predictions[i].flatten() * scales[i] for i in range(n_segments)]).flatten()
            data_raw_segment = data_raw[:len(data_clean)]
            
            # 计算指标
            var_raw, var_clean = np.var(data_raw_segment), np.var(data_clean)
            nrr = (var_raw - var_clean) / var_raw * 100
            cc = np.corrcoef(data_raw_segment, data_clean)[0, 1]
            
            log(f"{subject_id:<12} | {n_segments:<8} | {nrr:<10.2f} | {cc:<8.4f}", f)
            
            all_results.append({'subject': subject_id, 'nrr': nrr, 'cc': cc})
            
        except Exception as e:
            log(f"{subject_id:<12} | 处理失败: {e}", f)
    
    # 汇总
    log("-" * 60, f)
    if all_results:
        avg_nrr = np.mean([r['nrr'] for r in all_results])
        avg_cc = np.mean([r['cc'] for r in all_results])
        log(f"{'平均':<12} | {'':<8} | {avg_nrr:<10.2f} | {avg_cc:<8.4f}", f)
    
    log("\n" + "=" * 60, f)
    log("✅ 测试完成!", f)
    log("=" * 60, f)

print(f"\n结果已保存到: {output_file}")