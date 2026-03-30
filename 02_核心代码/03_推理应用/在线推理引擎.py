import streamlit as st
import numpy as np
import pandas as pd
import os
import tempfile
import mne
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import welch, resample
import tensorflow as tf
from tensorflow.keras.models import load_model
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
    st.sidebar.success("✅ 深度学习推理核心已驻留显存/内存待命")
else:
    st.sidebar.error(f"❌ 模型加载失败！请检查目录: {base_dir}")

# ================= 2. 数据处理与推理函数 =================
def process_data_and_infer(edf_file, txt_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_edf:
        tmp_edf.write(edf_file.getvalue())
        tmp_edf_path = tmp_edf.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_txt:
        tmp_txt.write(txt_file.getvalue())
        tmp_txt_path = tmp_txt.name

    try:
        raw = mne.io.read_raw_edf(tmp_edf_path, preload=True, verbose=False)
        ch_name = next((c for c in raw.ch_names if 'C3' in c or 'CZ' in c), raw.ch_names[0])

        # ================== 1) Raw 基线：对齐 Evaluate_N3_Fidelity.py 输入准备（100Hz -> 30s -> epoch z-score） ==================
        raw_100 = raw.copy()
        if raw_100.info['sfreq'] != 100:
            raw_100.resample(100)
        raw_data_100 = raw_100.get_data(picks=ch_name)[0]

        epoch_len = 30 * 100  # 3000
        n_epochs_raw = len(raw_data_100) // epoch_len
        X_raw = raw_data_100[:n_epochs_raw * epoch_len].reshape(n_epochs_raw, epoch_len, 1)
        mean_raw = np.mean(X_raw, axis=1, keepdims=True)
        std_raw = np.std(X_raw, axis=1, keepdims=True)
        std_raw[std_raw < 1e-8] = 1
        X_raw = (X_raw - mean_raw) / std_raw
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # ================== 2) V4 去噪：对齐 Run_V4_Inference.py（256Hz -> 512块逐块标准化/反还原 -> 再降采样到 100Hz） ==================
        raw_256 = raw.copy()
        if raw_256.info['sfreq'] != 256:
            raw_256.resample(256)
        raw_data_256 = raw_256.get_data(picks=ch_name)[0]

        slen = 512
        pad_len = int(np.ceil(len(raw_data_256) / slen) * slen) - len(raw_data_256)
        raw_padded = np.pad(raw_data_256, (0, pad_len), mode='constant')
        n_seg = len(raw_padded) // slen

        # 每个 512 块独立做 (x-m)/std，并在预测后做 pred*std+m 还原幅度
        chunks = raw_padded.reshape(n_seg, slen).astype(np.float32)
        m = chunks.mean(axis=1, keepdims=True)
        std = chunks.std(axis=1, keepdims=True)
        std_fixed = np.where(std == 0, 1.0, std).astype(np.float32)
        chunks_std = (chunks - m) / std_fixed
        X_v4_in = chunks_std[..., None]  # (n_seg, 512, 1)

        X_denoised_in = v4_model.predict(X_v4_in, batch_size=32, verbose=0)
        denoised_arr = np.asarray(X_denoised_in)
        if denoised_arr.ndim == 3:
            denoised_flat = denoised_arr[:, :, 0]
        else:
            denoised_flat = denoised_arr.reshape(n_seg, slen)

        denoised_chunks = denoised_flat * std_fixed + m
        denoised_256hz = denoised_chunks.reshape(-1)[:len(raw_data_256)]

        # 256Hz -> 100Hz
        num_samples_100hz = int(len(denoised_256hz) * 100 / 256)
        data_100hz = resample(denoised_256hz, num_samples_100hz)

        # 100Hz -> 30s epoch，并做 epoch 级 z-score 标准化（对齐 Run_V4_Inference.py stage 2）
        n_epochs_v4 = len(data_100hz) // epoch_len
        X_v4 = data_100hz[:n_epochs_v4 * epoch_len].reshape(n_epochs_v4, epoch_len, 1)
        mean_v4 = np.mean(X_v4, axis=1, keepdims=True)
        std_v4 = np.std(X_v4, axis=1, keepdims=True)
        std_v4[std_v4 < 1e-8] = 1
        X_v4 = (X_v4 - mean_v4) / std_v4
        X_v4 = np.nan_to_num(X_v4, nan=0.0, posinf=0.0, neginf=0.0)

        # 为了和标签对齐，取两路信号共同能覆盖的最小 epoch 数
        n_epochs = min(n_epochs_raw, n_epochs_v4)
        X_raw = X_raw[:n_epochs]
        X_v4 = X_v4[:n_epochs]

        # ================== 3) 标签：对齐 Evaluate_N3_Fidelity.py 的 5s->30s 多数投票（每 epoch 6 行） ==================
        df = pd.read_csv(tmp_txt_path, header=None, skiprows=1)
        labels_raw = df.values.flatten()[:n_epochs * 6]
        labels_reshaped = labels_raw.reshape(n_epochs, 6)
        mode_res = stats.mode(labels_reshaped, axis=1, keepdims=False)
        labels_30s = getattr(mode_res, "mode", mode_res[0])
        y_true = labels_30s.flatten().astype(int)

        # DREAMS(0..5) -> DeepSleepNet(0..4)
        y_true[y_true == 4] = 3  # N4 -> N3
        y_true[y_true == 5] = 4  # REM -> 4

        valid_idx = (y_true >= 0) & (y_true <= 4)
        X_valid = X_raw[valid_idx]
        X_v4_valid = X_v4[valid_idx]
        y_valid = y_true[valid_idx]

        # ================== 4) 裁判网络预测 ==================
        pred_probs_raw = referee_model.predict(X_valid, batch_size=32, verbose=0)
        y_pred_raw = np.argmax(pred_probs_raw, axis=1)

        pred_probs_v4 = referee_model.predict(X_v4_valid, batch_size=32, verbose=0)
        y_pred_v4 = np.argmax(pred_probs_v4, axis=1)

        return X_valid, X_v4_valid, y_valid, y_pred_raw, y_pred_v4

    finally:
        os.remove(tmp_edf_path)
        os.remove(tmp_txt_path)

# ================= 辅助函数：计算频段能量 =================
def calculate_band_power(freqs, psd, fmin, fmax):
    """利用梯形积分计算特定频段的总能量"""
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return np.trapz(psd[idx], freqs[idx])

# ================= 3. 界面交互舱 =================
st.header("📥 第一步：输入临床脑电与标签数据")
col1, col2 = st.columns(2)
with col1:
    uploaded_edf = st.file_uploader("上传原始受试者脑电 (.edf)", type=["edf"])
with col2:
    uploaded_txt = st.file_uploader("上传对应睡眠分期标签 (.txt)", type=["txt"])

st.header("⚙️ 第二步：启动端到端洗稿与审判")
if uploaded_edf and uploaded_txt and v4_model and referee_model:
    if st.button("🚀 启动 V4 深度去噪与特征实测"):
        
        with st.spinner('算法正在全速运转中...'):
            X_raw, X_v4, y_true, pred_raw, pred_v4 = process_data_and_infer(uploaded_edf, uploaded_txt)
            
            # 将数据存入 session_state 以便后续滑动条交互使用，避免重复推理
            st.session_state['X_raw'] = X_raw
            st.session_state['X_v4'] = X_v4
            st.session_state['y_true'] = y_true
            st.session_state['pred_raw'] = pred_raw
            st.session_state['pred_v4'] = pred_v4
            st.session_state['inference_done'] = True

if st.session_state.get('inference_done'):
    X_raw = st.session_state['X_raw']
    X_v4 = st.session_state['X_v4']
    y_true = st.session_state['y_true']
    pred_raw = st.session_state['pred_raw']
    pred_v4 = st.session_state['pred_v4']

    # 计算分类指标
    cm_raw = confusion_matrix(y_true, pred_raw, labels=[0, 1, 2, 3, 4])
    cm_v4 = confusion_matrix(y_true, pred_v4, labels=[0, 1, 2, 3, 4])
    
    acc_raw = accuracy_score(y_true, pred_raw) * 100
    acc_v4 = accuracy_score(y_true, pred_v4) * 100
    n1_raw = (cm_raw[1, 1] / np.sum(cm_raw[1, :]) * 100) if np.sum(cm_raw[1, :]) > 0 else 0
    n1_v4 = (cm_v4[1, 1] / np.sum(cm_v4[1, :]) * 100) if np.sum(cm_v4[1, :]) > 0 else 0
    n3_raw = (cm_raw[3, 3] / np.sum(cm_raw[3, :]) * 100) if np.sum(cm_raw[3, :]) > 0 else 0
    n3_v4 = (cm_v4[3, 3] / np.sum(cm_v4[3, :]) * 100) if np.sum(cm_v4[3, :]) > 0 else 0

    st.success("推理完毕！以下为特征保真度体检报告：")
    
    st.subheader("📊 1. 诊断任务战报：特征丢失的定量惩罚")
    c1, c2, c3 = st.columns(3)
    c1.metric("整体分期准确率 (Raw -> V4)", f"{acc_raw:.2f}%", f"{acc_v4 - acc_raw:.2f}% (V4后)", delta_color="inverse")
    c2.metric("N1 浅睡过渡期召回率", f"{n1_raw:.2f}%", f"{n1_v4 - n1_raw:.2f}% (V4后)", delta_color="inverse")
    c3.metric("N3 核心慢波召回率", f"{n3_raw:.2f}%", f"{n3_v4 - n3_raw:.2f}% (V4后)", delta_color="inverse")
    
    # 频域计算
    fs = 100
    flat_raw = X_raw.flatten()
    flat_v4 = X_v4.flatten()
    freqs, psd_raw = welch(flat_raw, fs, nperseg=1000)
    _, psd_v4 = welch(flat_v4, fs, nperseg=1000)

    # 能量流失计算
    bp_delta_raw = calculate_band_power(freqs, psd_raw, 0.5, 4)
    bp_delta_v4 = calculate_band_power(freqs, psd_v4, 0.5, 4)
    loss_delta = (bp_delta_raw - bp_delta_v4) / bp_delta_raw * 100

    bp_theta_raw = calculate_band_power(freqs, psd_raw, 4, 7)
    bp_theta_v4 = calculate_band_power(freqs, psd_v4, 4, 7)
    loss_theta = (bp_theta_raw - bp_theta_v4) / bp_theta_raw * 100

    bp_sigma_raw = calculate_band_power(freqs, psd_raw, 12, 15)
    bp_sigma_v4 = calculate_band_power(freqs, psd_v4, 12, 15)
    loss_sigma = (bp_sigma_raw - bp_sigma_v4) / bp_sigma_raw * 100

    st.subheader("⚠️ 2. 频段能量流失量化 (Energy Loss)")
    l1, l2, l3 = st.columns(3)
    l1.metric("Delta 慢波区 (0.5-4 Hz) 能量流失", f"-{loss_delta:.2f}%", "过度平滑导致 N3 暴跌的主因", delta_color="inverse")
    l2.metric("Theta 过渡区 (4-7 Hz) 能量流失", f"-{loss_theta:.2f}%", "影响浅睡期判读", delta_color="inverse")
    l3.metric("纺锤波区 (12-15 Hz) 能量流失", f"-{loss_sigma:.2f}%", "微小高频特征被强行抹杀", delta_color="inverse")

    st.subheader("🔎 3. 频域物证：脑电功率谱密度 (PSD) 塌陷图")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freqs, 10 * np.log10(psd_raw), label='原始信号 (Raw)', color='#4c72b0', linewidth=2)
    ax.plot(freqs, 10 * np.log10(psd_v4), label='V4 去噪信号', color='#dd8452', linewidth=2)
    ax.axvspan(0.5, 4, color='gray', alpha=0.15, label='N3: Delta 慢波区')
    ax.axvspan(4, 7, color='blue', alpha=0.1, label='N1: Theta 过渡区')
    ax.axvspan(12, 15, color='orange', alpha=0.15, label='N2: 纺锤波区')
    ax.set_xlim(0, 30)
    ax.set_xlabel("频率 (Hz)", fontsize=12)
    ax.set_ylabel("功率谱密度 Power (dB/Hz)", fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig)

    st.subheader("⏱️ 4. 交互式时域波形显微镜 (微观对比)")
    st.markdown("拖动下方滑块，观察 V4 是如何将原本起伏的生理特征拉平的（为方便对比，V4信号已向下偏移放置）。")
    
    max_epoch = X_raw.shape[0] - 1
    selected_epoch = st.slider("选择要查看的 30秒 脑电片段 (Epoch 索引)", min_value=0, max_value=max_epoch, value=15, step=1)
    
    fig_time, ax_time = plt.subplots(figsize=(14, 4))
    time_axis = np.linspace(0, 30, 3000)
    
    raw_epoch_data = X_raw[selected_epoch].flatten()
    v4_epoch_data = X_v4[selected_epoch].flatten()
    
    ax_time.plot(time_axis, raw_epoch_data, label=f"原始信号 Raw (真实分期: {y_true[selected_epoch]})", color='#4c72b0', alpha=0.9, linewidth=1)
    # 向下偏移 6 个单位，防止两条线糊在一起
    ax_time.plot(time_axis, v4_epoch_data - 6, label=f"V4 去噪后 (机器预测: {pred_v4[selected_epoch]})", color='#dd8452', linewidth=1)
    
    ax_time.set_title(f"Epoch {selected_epoch} 局部波形特征抹除对比", fontsize=14)
    ax_time.set_xlabel("时间 (秒)", fontsize=12)
    ax_time.set_yticks([]) # 隐藏 Y 轴刻度，因为已经做了偏移
    ax_time.legend(loc="upper right")
    ax_time.grid(True, linestyle='--', alpha=0.4)
    st.pyplot(fig_time)

else:
    if not v4_model or not referee_model:
        st.warning("⚠️ 模型未正确加载，请检查左侧控制台状态。")
    else:
        st.info("👈 等待数据注入。请在上方上传需要测试的 .edf 与标签文件。")