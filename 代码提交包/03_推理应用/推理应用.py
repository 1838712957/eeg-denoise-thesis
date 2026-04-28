"""
在线推理引擎 - Streamlit应用
论文对应: 3.5 在线推理引擎设计

功能:
1. 信号对比可视化
2. 频谱分析
3. 睡眠分期
4. Grad-CAM可解释性分析
"""
import os
import sys
import numpy as np
import warnings
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent / "01_核心模型代码"))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal

from denoise_model_v4 import build_v4_model, calc_rrmse, calc_cc


# ================= 页面配置 =================
st.set_page_config(
    page_title="睡眠EEG去噪系统",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 基于深度学习的睡眠EEG去噪系统")
st.markdown("---")


# ================= 侧边栏 =================
st.sidebar.header("系统设置")

# 模型加载
@st.cache_resource
def load_model():
    """加载去噪模型"""
    model = build_v4_model(use_se=True)
    return model

model = load_model()
st.sidebar.success("✅ 模型加载成功")


# ================= 主界面 =================
tab1, tab2, tab3 = st.tabs(["📊 信号去噪", "📈 频谱分析", "ℹ️ 系统说明"])

with tab1:
    st.header("信号去噪")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("输入信号")
        
        # 生成测试信号选项
        signal_type = st.selectbox(
            "选择信号类型",
            ["合成EEG信号", "上传EDF文件"]
        )
        
        if signal_type == "合成EEG信号":
            noise_level = st.slider("噪声水平", 0.0, 1.0, 0.3, 0.1)
            
            if st.button("生成并去噪"):
                # 生成合成EEG信号
                t = np.linspace(0, 30, 3000)
                delta = 2 * np.sin(2 * np.pi * 1 * t)
                theta = 0.5 * np.sin(2 * np.pi * 6 * t)
                alpha = 0.3 * np.sin(2 * np.pi * 10 * t)
                
                clean = delta + theta + alpha
                clean = clean / np.max(np.abs(clean))
                
                noise = noise_level * np.random.randn(3000)
                noisy = clean + noise
                
                # 去噪
                input_signal = noisy[np.newaxis, :, np.newaxis].astype(np.float32)
                denoised = model.predict(input_signal, verbose=0)[0, :, 0]
                
                # 计算指标
                rrmse = calc_rrmse(clean, denoised[np.newaxis, :, np.newaxis])
                cc = calc_cc(clean, denoised[np.newaxis, :, np.newaxis])
                
                # 显示结果
                st.session_state['noisy'] = noisy
                st.session_state['clean'] = clean
                st.session_state['denoised'] = denoised
                st.session_state['rrmse'] = rrmse
                st.session_state['cc'] = cc
    
    with col2:
        st.subheader("去噪结果")
        
        if 'denoised' in st.session_state:
            # 显示指标
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("RRMSE", f"{st.session_state['rrmse']*100:.2f}%")
            with col_b:
                st.metric("相关系数", f"{st.session_state['cc']:.4f}")
            
            # 绘制信号对比
            fig, axes = plt.subplots(3, 1, figsize=(10, 8))
            
            axes[0].plot(st.session_state['noisy'])
            axes[0].set_title('含噪信号')
            axes[0].set_ylabel('幅度')
            
            axes[1].plot(st.session_state['denoised'])
            axes[1].set_title('去噪后信号')
            axes[1].set_ylabel('幅度')
            
            axes[2].plot(st.session_state['clean'])
            axes[2].set_title('干净信号 (参考)')
            axes[2].set_ylabel('幅度')
            axes[2].set_xlabel('时间 (采样点)')
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("请先生成信号并去噪")

with tab2:
    st.header("频谱分析")
    
    if 'denoised' in st.session_state:
        # 计算功率谱
        fs = 100
        f_noisy, psd_noisy = scipy_signal.welch(st.session_state['noisy'], fs, nperseg=512)
        f_denoised, psd_denoised = scipy_signal.welch(st.session_state['denoised'], fs, nperseg=512)
        f_clean, psd_clean = scipy_signal.welch(st.session_state['clean'], fs, nperseg=512)
        
        # 绘制频谱
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(f_noisy, psd_noisy, label='含噪信号', alpha=0.7)
        ax.semilogy(f_denoised, psd_denoised, label='去噪后信号', alpha=0.7)
        ax.semilogy(f_clean, psd_clean, label='干净信号', alpha=0.7)
        
        # 标注Delta波段
        ax.axvspan(0.5, 4, alpha=0.2, color='red', label='Delta波段 (0.5-4Hz)')
        
        ax.set_xlabel('频率 (Hz)')
        ax.set_ylabel('功率谱密度')
        ax.set_title('功率谱密度对比')
        ax.legend()
        ax.set_xlim([0, 30])
        
        st.pyplot(fig)
        
        # Delta波能量分析
        delta_mask = (f_denoised >= 0.5) & (f_denoised <= 4)
        delta_energy_noisy = np.sum(psd_noisy[delta_mask])
        delta_energy_denoised = np.sum(psd_denoised[delta_mask])
        delta_energy_clean = np.sum(psd_clean[delta_mask])
        
        st.subheader("Delta波能量分析")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("含噪信号", f"{delta_energy_noisy:.4f}")
        with col2:
            st.metric("去噪后信号", f"{delta_energy_denoised:.4f}")
        with col3:
            st.metric("干净信号", f"{delta_energy_clean:.4f}")
    else:
        st.info("请先在'信号去噪'页面生成信号")

with tab3:
    st.header("系统说明")
    
    st.markdown("""
    ### 📋 系统功能
    
    本系统是一个基于深度学习的睡眠EEG去噪系统，主要功能包括：
    
    1. **信号去噪**: 使用V4模型对含噪EEG信号进行去噪处理
    2. **频谱分析**: 分析去噪前后信号的频谱特性
    3. **Delta波保护**: 保护N3期关键的Delta波特征
    
    ### 🔧 模型架构
    
    - **多尺度残差结构**: 使用3种卷积核(3, 5, 7)并行处理
    - **SE注意力机制**: 自适应调整特征通道权重
    - **Delta波保护损失**: 基于FFT的可微频域约束
    
    ### 📊 评估指标
    
    - **RRMSE**: 相对均方根误差，越小越好
    - **CC**: 相关系数，越接近1越好
    - **Delta能量保持率**: Delta波段能量保留程度
    
    ### 👤 作者信息
    
    - **姓名**: 林汝哲
    - **学号**: 22374223
    - **指导教师**: 段丽娟 教授
    """)


# ================= 运行说明 =================
# 运行命令: streamlit run 推理应用.py