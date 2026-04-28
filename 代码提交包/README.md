# 基于深度学习的睡眠EEG信号去噪方法研究

**作者**: 林汝哲  
**学号**: 22374223  
**指导教师**: 邢玲副教授

## 项目简介

本项目提出了一种基于深度学习的睡眠EEG信号去噪方法，旨在提高睡眠分期任务的准确性。主要贡献包括：

1. 设计了多尺度残差卷积神经网络去噪模型
2. 提出了Delta波能量保持损失函数
3. 在Sleep-EDF和DREAMS数据集上验证了方法的有效性

## 目录结构

```
代码提交包/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖
│
├── 01_核心模型代码/
│   ├── model_components.py      # 模型组件 (SEBlock, ResBlock等)
│   └── denoise_model_v4.py      # V4去噪模型完整实现
│
├── 02_核心实验代码/
│   ├── 实验1_SleepEDF去噪效果/
│   │   └── sleep_edf_denoising.py
│   ├── 实验2_消融实验/
│   │   └── ablation_study.py
│   ├── 实验3_睡眠分期对比/
│   │   └── staging_comparison.py
│   ├── 实验4_鲁棒性分析/
│   │   └── robustness_analysis.py
│   ├── 实验5_N3召回率分析/
│   │   └── n3_recall_analysis.py
│   ├── 实验6_数据量实验/
│   │   └── train_data_scaling.py
│   ├── 实验7_对比分析/
│   │   └── comparative_analysis.py
│   └── 实验8_GradCAM分析/
│       └── gradcam_analysis.py
│
└── 03_推理应用/
    └── 推理应用.py              # Streamlit在线推理系统
```

## 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```

主要依赖：
- Python 3.8+
- TensorFlow 2.x
- NumPy, SciPy, scikit-learn
- Streamlit (用于推理应用)

## 运行实验

### 实验1: Sleep-EDF去噪效果
```bash
python 代码提交包/02_核心实验代码/实验1_SleepEDF去噪效果/sleep_edf_denoising.py
```

### 实验2: 消融实验
```bash
python 代码提交包/02_核心实验代码/实验2_消融实验/ablation_study.py
```

### 实验3: 睡眠分期对比
```bash
python 代码提交包/02_核心实验代码/实验3_睡眠分期对比/staging_comparison.py
```

### 实验4: 鲁棒性分析
```bash
python 代码提交包/02_核心实验代码/实验4_鲁棒性分析/robustness_analysis.py
```

### 实验5: N3召回率分析
```bash
python 代码提交包/02_核心实验代码/实验5_N3召回率分析/n3_recall_analysis.py
```

### 实验6: 数据量实验
```bash
python 代码提交包/02_核心实验代码/实验6_数据量实验/train_data_scaling.py
```

### 实验7: 对比分析
```bash
python 代码提交包/02_核心实验代码/实验7_对比分析/comparative_analysis.py
```

### 实验8: GradCAM分析
```bash
python 代码提交包/02_核心实验代码/实验8_GradCAM分析/gradcam_analysis.py
```

### 运行推理应用
```bash
streamlit run 代码提交包/03_推理应用/推理应用.py
```

## 实验结果摘要

| 实验 | 主要指标 | 结果 |
|------|----------|------|
| Sleep-EDF去噪 | RRMSE | 18.5% |
| 睡眠分期对比 | 准确率提升 | +5.2% |
| N3召回率 | 召回率提升 | +12.3% |
| 消融实验 | SE模块贡献 | +2.1% |
| 鲁棒性分析 | 高噪声下性能 | 稳定 |
| 数据量实验 | 最优数据量 | 80%训练集 |

## 模型架构

V4去噪模型采用编码器-解码器结构：
- **编码器**: 多尺度卷积 + SE注意力
- **解码器**: 转置卷积 + 跳跃连接
- **损失函数**: MSE + Delta波能量保持

## 联系方式

如有问题，请联系: linruzhe@bjut.edu.cn