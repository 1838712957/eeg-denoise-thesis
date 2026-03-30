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

st.header("⚙️ 第二步：启动端到端洗稿与审判")
if uploaded_edf and uploaded_txt and v4_model and referee_model:
    if st.button("🚀 启动 V4 深度去噪与特征实测", use_container_width=True):
        with st.spinner('算法正在全速运转中...'):
            X_raw, X_v4, y_true, pred_raw, pred_v4 = process_data_and_infer(uploaded_edf, uploaded_txt)
            st.session_state.processed_data = {
                'X_raw': X_raw,
                'X_v4': X_v4,
                'y_true': y_true,
                'pred_raw': pred_raw,
                'pred_v4': pred_v4
            }
    
    if st.session_state.processed_data is not None:
        data = st.session_state.processed_data
        X_raw = data['X_raw']
        X_v4 = data['X_v4']
        y_true = data['y_true']
        pred_raw = data['pred_raw']
        pred_v4 = data['pred_v4']
        
        cm_raw = confusion_matrix(y_true, pred_raw, labels=[0, 1, 2, 3, 4])
        cm_v4 = confusion_matrix(y_true, pred_v4, labels=[0, 1, 2, 3, 4])
        
        acc_raw = accuracy_score(y_true, pred_raw) * 100
        acc_v4 = accuracy_score(y_true, pred_v4) * 100
        
        n3_raw = (cm_raw[3, 3] / np.sum(cm_raw[3, :]) * 100) if np.sum(cm_raw[3, :]) > 0 else 0
        n3_v4 = (cm_v4[3, 3] / np.sum(cm_v4[3, :]) * 100) if np.sum(cm_v4[3, :]) > 0 else 0

        st.success("推理完毕！以下为当前上传受试者的即时特征保真度体检报告：")
        
        st.subheader("📊 诊断任务战报：特征丢失的定量惩罚")
        c1, c2, c3 = st.columns(3)
        c1.metric("整体分期准确率", f"{acc_raw:.2f}%", f"{acc_v4 - acc_raw:.2f}% (V4后)", delta_color="inverse")
        c2.metric("N3 核心慢波召回率", f"{n3_raw:.2f}%", f"{n3_v4 - n3_raw:.2f}% (V4后)", delta_color="inverse")
        
        st.subheader("📉 Delta能量量化：慢波特征损失分析")
        delta_raw, delta_v4, delta_loss_pct = compute_delta_energy_loss(X_raw, X_v4)
        
        d1, d2, d3 = st.columns(3)
        d1.metric("原始Delta能量", f"{delta_raw:.2e}", "0.5-4Hz")
        d2.metric("去噪后Delta能量", f"{delta_v4:.2e}", "0.5-4Hz")
        if delta_loss_pct > 30:
            d3.metric("Delta能量损失", f"{delta_loss_pct:.1f}%", "⚠️ 严重损失!", delta_color="inverse")
        else:
            d3.metric("Delta能量损失", f"{delta_loss_pct:.1f}%")
        st.info(f"💡 V4去噪导致Delta慢波频段能量损失 **{delta_loss_pct:.1f}%**，这是N3识别率下降的直接物理证据。")
        
        st.subheader("🔎 频域物证：功率谱密度对比")
        fs = 100
        flat_raw = X_raw.flatten()
        flat_v4 = X_v4.flatten()
        freqs, psd_raw = welch(flat_raw, fs, nperseg=1000)
        _, psd_v4 = welch(flat_v4, fs, nperseg=1000)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(freqs, 10 * np.log10(psd_raw), label='原始信号', color='#4c72b0', linewidth=2)
        ax.plot(freqs, 10 * np.log10(psd_v4), label='V4去噪信号', color='#dd8452', linewidth=2)
        ax.axvspan(0.5, 4, color='gray', alpha=0.15, label='Delta区 (0.5-4Hz)')
        ax.set_xlim(0, 30)
        ax.set_title("频域对比：过度平滑与能量塌陷", fontsize=14)
        ax.set_xlabel("频率 (Hz)")
        ax.set_ylabel("功率 (dB/Hz)")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig)
        
        st.subheader("🔥 Grad-CAM注意力分析：受害样本检测")
        victim_indices = find_victim_epochs(y_true, pred_raw, pred_v4, target_class=3)
        
        if len(victim_indices) > 0:
            st.warning(f"发现 {len(victim_indices)} 个受害样本 (N3→其他)")
            n_show = min(3, len(victim_indices))
            stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
            
            for i, idx in enumerate(victim_indices[:n_show]):
                st.markdown(f"**受害样本 #{i+1} (Epoch {idx})**")
                true_label = y_true[idx]
                raw_pred = pred_raw[idx]
                v4_pred = pred_v4[idx]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"原始信号: 真实={stage_names[true_label]}, 预测={stage_names[raw_pred]} ✅")
                with col2:
                    st.markdown(f"去噪信号: 真实={stage_names[true_label]}, 预测={stage_names[v4_pred]} ❌")
                
                try:
                    signal_raw = X_raw[idx:idx+1]
                    signal_v4 = X_v4[idx:idx+1]
                    cam_raw = compute_gradcam_1d(referee_model, signal_raw, class_idx=3)
                    cam_v4 = compute_gradcam_1d(referee_model, signal_v4, class_idx=3)
                    
                    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
                    t = np.arange(3000) / 100
                    signal_raw_flat = signal_raw.flatten()
                    signal_v4_flat = signal_v4.flatten()
                    
                    ax1.plot(t, signal_raw_flat, color='blue', alpha=0.7, label='原始EEG')
                    cam_raw_norm = (cam_raw - np.min(cam_raw)) / (np.max(cam_raw) - np.min(cam_raw) + 1e-8)
                    for j in range(len(t)-1):
                        if cam_raw_norm[j] > 0.3:
                            ax1.axvspan(t[j], t[j+1], alpha=cam_raw_norm[j]*0.5, color='red')
                    ax1.set_title(f'原始信号Grad-CAM (预测: {stage_names[raw_pred]})', fontsize=12)
                    ax1.set_xlabel('时间 (s)')
                    ax1.set_ylabel('幅度')
                    ax1.legend(loc='upper right')
                    ax1.set_xlim(0, 30)
                    
                    ax2.plot(t, signal_v4_flat, color='orange', alpha=0.7, label='去噪EEG')
                    cam_v4_norm = (cam_v4 - np.min(cam_v4)) / (np.max(cam_v4) - np.min(cam_v4) + 1e-8)
                    for j in range(len(t)-1):
                        if cam_v4_norm[j] > 0.3:
                            ax2.axvspan(t[j], t[j+1], alpha=cam_v4_norm[j]*0.5, color='red')
                    ax2.set_title(f'去噪信号Grad-CAM (预测: {stage_names[v4_pred]}) - 注意力分散!', fontsize=12)
                    ax2.set_xlabel('时间 (s)')
                    ax2.set_ylabel('幅度')
                    ax2.legend(loc='upper right')
                    ax2.set_xlim(0, 30)
                    
                    plt.tight_layout()
                    st.pyplot(fig2)
                except Exception as e:
                    st.error(f"Grad-CAM计算失败: {e}")
        else:
            st.success("✅ 未发现N3期受害样本，去噪效果良好！")
        
        st.subheader("📋 分析总结")
        summary_data = {
            '指标': ['整体准确率变化', 'N3召回率变化', 'Delta能量损失', '受害样本数'],
            '数值': [f"{acc_v4 - acc_raw:.2f}%", f"{n3_v4 - n3_raw:.2f}%", f"{delta_loss_pct:.1f}%", str(len(victim_indices))]
        }
        st.table(pd.DataFrame(summary_data))
        
        # ================= 信号片段查看器 =================
        st.subheader("🔍 信号片段查看器")
        st.markdown("拖动滑块查看不同时间段的原始信号与去噪信号对比")
        
        n_total_epochs = len(X_raw)
        epoch_slider = st.slider("选择Epoch片段", min_value=0, max_value=n_total_epochs-1, value=0, step=1)
        
        stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
        current_true = y_true[epoch_slider]
        current_raw_pred = pred_raw[epoch_slider]
        current_v4_pred = pred_v4[epoch_slider]
        
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.metric("真实标签", stage_names[current_true])
        with info_col2:
            st.metric("原始信号预测", stage_names[current_raw_pred], 
                     delta="✅ 正确" if current_raw_pred == current_true else "❌ 错误")
        with info_col3:
            st.metric("去噪信号预测", stage_names[current_v4_pred],
                     delta="✅ 正确" if current_v4_pred == current_true else "❌ 错误")
        
        fig_viewer, axes_viewer = plt.subplots(2, 1, figsize=(14, 6))
        t_viewer = np.arange(3000) / 100
        signal_raw_viewer = X_raw[epoch_slider].flatten()
        signal_v4_viewer = X_v4[epoch_slider].flatten()
        
        axes_viewer[0].plot(t_viewer, signal_raw_viewer, color='#1f77b4', linewidth=0.8, alpha=0.9)
        axes_viewer[0].set_title(f'原始EEG信号 - Epoch {epoch_slider}', fontsize=12, fontweight='bold')
        axes_viewer[0].set_xlabel('时间 (s)')
        axes_viewer[0].set_ylabel('幅度')
        axes_viewer[0].set_xlim(0, 30)
        axes_viewer[0].grid(True, alpha=0.3)
        
        axes_viewer[1].plot(t_viewer, signal_v4_viewer, color='#ff7f0e', linewidth=0.8, alpha=0.9)
        axes_viewer[1].set_title(f'去噪EEG信号 - Epoch {epoch_slider}', fontsize=12, fontweight='bold')
        axes_viewer[1].set_xlabel('时间 (s)')
        axes_viewer[1].set_ylabel('幅度')
        axes_viewer[1].set_xlim(0, 30)
        axes_viewer[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig_viewer)
        
        st.markdown("**当前Epoch频谱对比:**")
        freqs_viewer, psd_raw_viewer = welch(signal_raw_viewer, 100, nperseg=256)
        _, psd_v4_viewer = welch(signal_v4_viewer, 100, nperseg=256)
        
        fig_psd_viewer, ax_psd_viewer = plt.subplots(figsize=(10, 4))
        ax_psd_viewer.semilogy(freqs_viewer, psd_raw_viewer, label='原始信号', color='#1f77b4', linewidth=1.5)
        ax_psd_viewer.semilogy(freqs_viewer, psd_v4_viewer, label='去噪信号', color='#ff7f0e', linewidth=1.5)
        ax_psd_viewer.axvspan(0.5, 4, color='purple', alpha=0.1, label='Delta')
        ax_psd_viewer.set_xlim(0, 30)
        ax_psd_viewer.set_xlabel('频率 (Hz)')
        ax_psd_viewer.set_ylabel('功率谱密度')
        ax_psd_viewer.set_title(f'Epoch {epoch_slider} 频谱对比')
        ax_psd_viewer.legend(loc='upper right', fontsize=8)
        ax_psd_viewer.grid(True, alpha=0.3)
        st.pyplot(fig_psd_viewer)

else:
    if not v4_model or not referee_model:
        st.warning("⚠️ 模型未正确加载")
    else:
        st.info("👈 等待数据注入。请上传 .edf 与标签文件。")
