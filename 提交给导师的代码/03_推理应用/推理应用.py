import streamlit as st
import numpy as np
import pandas as pd
import os
import tempfile
import mne
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import welch
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from sklearn.metrics import confusion_matrix, accuracy_score
import sys
from pathlib import Path

# 解决画图时的中文乱码问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="脑电去噪在线推理引擎", layout="wide")
st.title("⚡ 脑电去噪端到端在线推理与临床评估引擎")

CORE_CODE_DIR = Path(__file__).resolve().parents[2] / "02_核心代码"
if str(CORE_CODE_DIR) not in sys.path:
    sys.path.append(str(CORE_CODE_DIR))
from bootstrap_paths import PROJECT_ROOT
base_dir = str(PROJECT_ROOT)

# ================= 初始化Session State =================
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# ================= 1. 核心模型全局缓存区 =================
@st.cache_resource
def load_deep_models():
    v4_model_path = os.path.join(base_dir, "03_训练模型", "V4最优去噪模型.h5") 
    referee_model_path = os.path.join(base_dir, "03_训练模型", "DeepSleepNet裁判模型.h5")
    
    v4_model = load_model(v4_model_path, compile=False) if os.path.exists(v4_model_path) else None
    referee_model = load_model(referee_model_path, compile=False) if os.path.exists(referee_model_path) else None
    
    return v4_model, referee_model

v4_model, referee_model = load_deep_models()

if v4_model and referee_model:
    st.sidebar.success("✅ 深度学习推理核心已就绪")
else:
    st.sidebar.error(f"❌ 模型加载失败！")

# ================= 2. Grad-CAM 核心函数 =================
def compute_gradcam_1d(model, input_signal, class_idx=None):
    """计算1D信号的Grad-CAM注意力热力图"""
    last_conv_layer = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            last_conv_layer = layer
            break
    
    if last_conv_layer is None:
        return np.ones(input_signal.shape[1]) / input_signal.shape[1]
    
    grad_model = Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_signal)
        if class_idx is None:
            class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap_np = heatmap.numpy()
    original_len = len(heatmap_np)
    target_len = input_signal.shape[1]
    if original_len != target_len:
        heatmap_np = np.interp(np.linspace(0, original_len-1, target_len), 
                               np.arange(original_len), heatmap_np)
    return heatmap_np

def find_victim_epochs(y_true, y_pred_raw, y_pred_v4, target_class=3):
    """找出受害样本：原始正确分类N3，去噪后被误判"""
    victim_indices = []
    for i in range(len(y_true)):
        if y_true[i] == target_class and y_pred_raw[i] == target_class and y_pred_v4[i] != target_class:
            victim_indices.append(i)
    return victim_indices

# ================= 3. Delta能量量化函数 =================
def compute_delta_energy_loss(X_raw, X_denoised, fs=100):
    """计算Delta频段(0.5-4Hz)能量损失"""
    flat_raw = X_raw.flatten()
    flat_v4 = X_denoised.flatten()
    
    freqs_raw, psd_raw = welch(flat_raw, fs, nperseg=1000)
    freqs_v4, psd_v4 = welch(flat_v4, fs, nperseg=1000)
    
    delta_mask = (freqs_raw >= 0.5) & (freqs_raw <= 4)
    delta_energy_raw = np.trapz(psd_raw[delta_mask], freqs_raw[delta_mask])
    delta_energy_v4 = np.trapz(psd_v4[delta_mask], freqs_v4[delta_mask])
    energy_loss_pct = (delta_energy_raw - delta_energy_v4) / delta_energy_raw * 100
    
    return delta_energy_raw, delta_energy_v4, energy_loss_pct

# ================= 4. 数据处理与推理函数 =================
def process_data_and_infer(edf_file, txt_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_edf:
        tmp_edf.write(edf_file.getvalue())
        tmp_edf_path = tmp_edf.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_txt:
        tmp_txt.write(txt_file.getvalue())
        tmp_txt_path = tmp_txt.name

    try:
        raw = mne.io.read_raw_edf(tmp_edf_path, preload=True, verbose=False)
        if raw.info['sfreq'] != 100: raw.resample(100)
        ch_name = next((c for c in raw.ch_names if 'C3' in c or 'CZ' in c), raw.ch_names[0])
        raw_data = raw.get_data(picks=ch_name)[0]

        n_epochs = len(raw_data) // 3000
        X_raw = raw_data[:n_epochs * 3000].reshape(n_epochs, 3000, 1)
        X_raw = (X_raw - np.mean(X_raw, axis=1, keepdims=True)) / (np.std(X_raw, axis=1, keepdims=True) + 1e-8)

        df = pd.read_csv(tmp_txt_path, header=None, skiprows=1)
        labels_raw = df.values.flatten()[:n_epochs * 6]
        labels_30s, _ = stats.mode(labels_raw.reshape((n_epochs, 6)), axis=1, keepdims=False)
        y_true = labels_30s.flatten()
        y_true[y_true == 4] = 3
        y_true[y_true == 5] = 4
        valid_idx = (y_true >= 0) & (y_true <= 4)
        
        X_valid = X_raw[valid_idx]
        y_valid = y_true[valid_idx]

        n_segments = 5
        segment_len = 512
        n_epochs = len(X_valid)
        
        X_v4_denoised = np.zeros_like(X_valid)
        all_segments = []
        segment_map = []
        
        for epoch_idx in range(n_epochs):
            epoch_signal = X_valid[epoch_idx].flatten()
            for seg_idx in range(n_segments):
                start = seg_idx * segment_len
                end = start + segment_len
                all_segments.append(epoch_signal[start:end])
                segment_map.append((epoch_idx, seg_idx))
        
        all_segments = np.array(all_segments).reshape(-1, segment_len, 1)
        denoised_segments = v4_model.predict(all_segments, batch_size=128, verbose=0)
        
        for i, (epoch_idx, seg_idx) in enumerate(segment_map):
            start = seg_idx * segment_len
            end = start + segment_len
            X_v4_denoised[epoch_idx, start:end, 0] = denoised_segments[i].flatten()
        
        X_v4_denoised[:, n_segments * segment_len:, 0] = X_valid[:, n_segments * segment_len:, 0]
        
        pred_probs_raw = referee_model.predict(X_valid, batch_size=32, verbose=0)
        y_pred_raw = np.argmax(pred_probs_raw, axis=1)
        
        pred_probs_v4 = referee_model.predict(X_v4_denoised, batch_size=32, verbose=0)
        y_pred_v4 = np.argmax(pred_probs_v4, axis=1)

        return X_valid, X_v4_denoised, y_valid, y_pred_raw, y_pred_v4

    finally:
        os.remove(tmp_edf_path)
        os.remove(tmp_txt_path)

# ================= 5. 界面交互舱 =================
st.header("📥 第一步：输入临床脑电与标签数据")
col1, col2 = st.columns(2)
with col1:
    uploaded_edf = st.file_uploader("上传原始受试者脑电 (.edf)", type=["edf"])
with col2:
    uploaded_txt = st.file_uploader("上传对应睡眠分期标签 (.txt)", type=["txt"])

st.header("⚙️ 第二步：启动端到端洗稿与