# 基于深度学习的睡眠EEG信号去噪方法研究

**作者**: 林汝哲  
**学号**: 22374223  
**指导教师**: 段丽娟教授

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
- MNE (EEG数据处理)
- Streamlit (用于推理应用)

---

## 实验复现步骤

### 准备工作

1. **下载数据集**
   - DREAMS数据集: https://zenodo.org/record/2650142
   - Sleep-EDF数据集: https://physionet.org/content/sleep-edfx/1.0.0/

2. **放置数据**
   ```
   项目根目录/
   ├── 04_原始数据/
   │   ├── Raw_edf 2/           # DREAMS数据集
   │   │   ├── subject1.edf
   │   │   ├── HypnogramAASM_subject1.txt
   │   │   └── ...
   │   └── sleep-edf-database/  # Sleep-EDF数据集
   ```

3. **预训练模型**
   ```
   项目根目录/
   ├── 03_训练模型/
   │   ├── V4_Paper_Denoiser.h5         # MSR-Denoiser (论文核心模型)
   │   ├── DeepSleepNet_SleepEDF_Raw.h5 # 睡眠分期裁判模型
   │   ├── Baseline.h5                  # 基础1D-CNN (消融对照)
   │   ├── V4_wo_SE.h5                  # 无SE注意力 (消融对照)
   │   ├── V4_RealEEG_Complete.h5      # 真实EEG训练完整版
   │   └── V4_RealEEG_wo_SE.h5         # 真实EEG训练无SE版
   ```

---

### 实验1: Sleep-EDF去噪效果

**目的**: 评估模型在真实睡眠EEG数据上的去噪性能

**运行命令**:
```bash
cd 项目根目录
python 代码提交包/02_核心实验代码/实验1_SleepEDF去噪效果/sleep_edf_denoising.py
```

**输出结果**:
- RRMSE (相对均方根误差)
- CC (相关系数)
- Delta波能量保持率

---

### 实验2: 消融实验

**目的**: 验证MSR-Denoiser各核心模块的独立贡献

**数据集**: Sleep-EDF测试集 (SC4022 + SC4031)
- 使用DeepSleepNet作为分期评估器
- 每个消融变体应用于整夜记录，分期结果与专家标注对比

**模型配置 (累积叠加)**:
1. Baseline CNN: 基础1D-CNN (无多尺度、无注意力、无跳跃连接)
2. + Multi-scale: 添加多尺度并行卷积分支 (k=3,5,7)
3. + SE attention: 添加SE通道注意力机制
4. + Delta loss: 替换MSE为Delta保护+频带加权损失
5. MSR-Denoiser (Full): 完整架构 (全部模块)

**运行命令**:
```bash
cd 项目根目录
python test_paper_denoiser.py
```

**输出文件**: `05_处理结果/`

**实验结果 (表4.3)**:
| Configuration | CC ↑ | N3 Recall ↑ | Δ CC | Δ N3 |
|-------------|------|-------------|------|------|
| Baseline CNN | 0.72 | 42.1% | — | — |
| + Multi-scale | 0.85 | 58.3% | +0.13 | +16.2 pp |
| + SE attention | 0.91 | 67.5% | +0.06 | +9.2 pp |
| + Delta loss | 0.94 | 79.1% | +0.03 | +11.6 pp |
| MSR-Denoiser (Full) | 0.96 | 79.1% | +0.02 | +0.0 pp |

**核心发现**:
- 每个组件均正向贡献，无回退
- Delta损失贡献最大N3提升 (+11.6 pp)
- 多尺度分支贡献最大CC提升 (+0.13)
- 架构提供容量，损失函数提供动机，两者缺一不可

---

### 实验3: 睡眠分期对比

**目的**: 比较去噪前后睡眠分期性能

**运行命令**:
```bash
cd 项目根目录
python 代码提交包/02_核心实验代码/实验3_睡眠分期对比/staging_comparison.py
```

**对比方法**:
- 原始信号 (无去噪)
- 传统带通滤波
- 本文方法

---

### 实验4: 鲁棒性分析

**目的**: 测试模型在不同噪声水平下的性能

**运行命令**:
```bash
cd 项目根目录
python 代码提交包/02_核心实验代码/实验4_鲁棒性分析/robustness_analysis.py
```

**噪声分组**:
- 严重伪迹 (Peak-to-Peak > 200μV)
- 中度伪迹 (100μV < P-P < 200μV)
- 干净信号 (P-P < 100μV)

---

### 实验5: N3召回率分析

**目的**: 分析N3深睡阶段召回率优化

**运行命令**:
```bash
cd 项目根目录
python 代码提交包/02_核心实验代码/实验5_N3召回率分析/n3_recall_analysis.py
```

**分析内容**:
- N3阶段Delta波特征分析
- 噪声对N3识别的影响
- 去噪后N3召回率提升

---

### 实验6: 数据量实验

**目的**: 分析训练数据量对模型性能的影响

**运行命令**:
```bash
cd 项目根目录
python 代码提交包/02_核心实验代码/实验6_数据量实验/train_data_scaling.py
```

**数据比例**: 20%, 40%, 60%, 80%, 100%

---

### 实验7: 对比分析

**目的**: 与现有去噪方法对比

**运行命令**:
```bash
cd 项目根目录
python 代码提交包/02_核心实验代码/实验7_对比分析/comparative_analysis.py
```

**对比方法**:
- 带通滤波
- 小波去噪
- 本文方法

---

### 实验8: GradCAM分析

**目的**: 可视化模型关注的频段区域

**运行命令**:
```bash
cd 项目根目录
python 代码提交包/02_核心实验代码/实验8_GradCAM分析/gradcam_analysis.py
```

**输出**: GradCAM热力图

---

### 运行推理应用

```bash
streamlit run 代码提交包/03_推理应用/推理应用.py
```

功能：
- 上传EEG信号文件
- 实时去噪处理
- 睡眠分期预测
- 结果可视化

---

## 实验结果摘要

| 实验 | 主要指标 | 结果 |
|------|----------|------|
| Sleep-EDF去噪 | NRR | 92.08% (vs ASR 75.67%) |
| 消融实验 | 每模块正向贡献 | N3: 42.1%→79.1%, CC: 0.72→0.96 |
| 睡眠分期对比 | Balanced Accuracy | 77.70%→78.38% (+0.68pp) |
| DREAMS跨数据集 | N3召回率 | 28.0%→70.4% (+42.4pp) |
| ASR对比 | ASR N3误杀 | 95.2%→36.7% (ASR) vs 92.6% (MSR-Denoiser) |
| Delta能量保持 | vs ASR | 92.8% (MSR-Denoiser) vs 78.3% (ASR) |

## 模型架构

**MSR-Denoiser** (Multi-Scale Residual Denoiser, V4_Paper_Denoiser.h5):
- **参数量**: 69,433
- **输入**: 单通道EEG, 3000 samples @ 100Hz (30秒epoch)
- **多尺度并行分支**: k=3 (EMG快变), k=5 (纺锤波), k=7 (Delta/EOG慢波)
- **SE通道注意力**: 96通道→6→96, 内容自适应加权
- **全局跳跃连接**: output = input + R(input), 残差学习
- **损失函数**: L_RMSE + 0.1·L_Delta + 0.05·L_BandWeighted

## 注意事项

1. 实验代码需要在项目根目录下运行
2. 需要预先准备好数据集和模型权重文件
3. 部分实验需要GPU加速（推荐显存≥8GB）
4. 实验结果会保存在 `05_处理结果/` 目录下

## 联系方式

如有问题，请联系: linruzhe@bjut.edu.cn
