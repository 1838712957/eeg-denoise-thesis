# 毕业论文项目 - 核心代码指南

## 项目概述

本项目是一个**EEG脑电信号去噪与睡眠分期分析系统**，核心研究问题是：
> 深度学习去噪模型（V4）在去除噪声的同时，是否会过度平滑导致关键生理特征（如Delta慢波）丢失，进而影响下游睡眠分期任务的准确性？

---

## 目录结构

```
02_核心代码/
├── bootstrap_paths.py          # 路径初始化（所有脚本的入口依赖）
├── path_utils.py               # 路径工具函数
├── 01_数据预处理/              # 数据加载与预处理
├── 02_模型训练/                # 模型训练脚本
├── 03_推理应用/                # 在线推理引擎（Streamlit）
├── 04_分期分析/                # 睡眠分期对比分析
├── 05_批量评估/                # 批量评估与可视化
└── 06_辅助脚本/                # 运行脚本
```

---

## 核心模型架构

### 1. 去噪模型 (V4 - 1D ResNet + SE注意力)

**文件位置**: `04_分期分析/分期分析.py` 中的模型定义

**网络结构**:
```python
# 核心组件
class SEBlock(layers.Layer):
    """Squeeze-and-Excitation注意力模块"""
    # 通过全局平均池化 + 全连接层学习通道权重
    # 实现自适应特征重标定

class Res_BasicBlock(layers.Layer):
    """残差基本块"""
    # Conv1D(32) -> BN -> ReLU -> Conv1D(16) -> BN -> ReLU -> Conv1D(32) -> BN -> ReLU
    # 可选SE注意力
    # 残差连接: output + input

class BasicBlockall(layers.Layer):
    """多尺度残差块"""
    # 并行使用3种卷积核: kernel_size = 3, 5, 7
    # 捕获不同尺度的时序特征
    # 输出拼接: [bblock3, bblock5, bblock7]

# 完整模型
Input(512, 1) -> Conv1D(32, 7) -> BN -> ReLU -> BasicBlockall(use_se=True) -> Conv1D(1, 7) -> Output
```

**输入/输出**:
- 输入: `(batch, 512, 1)` - 2秒EEG片段 @ 256Hz
- 输出: `(batch, 512, 1)` - 去噪后的EEG片段

### 2. 睡眠分期模型 (DeepSleepNet裁判模型)

**文件位置**: `03_训练模型/DeepSleepNet裁判模型.h5`

**功能**: 对EEG信号进行5类睡眠分期（Wake, N1, N2, N3, REM）

**输入/输出**:
- 输入: `(batch, 3000, 1)` - 30秒EEG片段 @ 100Hz
- 输出: `(batch, 5)` - 5类睡眠阶段的概率分布

---

## 关键代码文件说明

### 1. `bootstrap_paths.py` - 路径初始化

**作用**: 定义项目根目录，所有其他脚本依赖此文件定位数据/模型

```python
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 项目根目录
```

### 2. `01_数据预处理/数据预处理.py` - DREAMS数据集处理

**功能**: 将DREAMS数据集的EDF文件转换为训练用的numpy数组

**关键步骤**:
1. 读取EDF文件 -> 重采样到256Hz
2. 按512采样点(2秒)切片
3. Z-score标准化
4. 保存为`.npy`格式

### 3. `02_模型训练/模型训练.py` - 睡眠分期CNN训练

**功能**: 训练一个基础的睡眠分期CNN模型（非去噪模型）

**模型结构**:
```python
class BetterSleepCNN(nn.Module):
    # Conv1D(1->16, k=64, s=2) -> BN -> ReLU -> MaxPool
    # Conv1D(16->32, k=32, s=2) -> BN -> ReLU -> MaxPool
    # Conv1D(32->64, k=16, s=2) -> BN -> ReLU -> MaxPool
    # AdaptiveAvgPool1d(10) -> Flatten -> Linear(640->128) -> Dropout -> Linear(128->5)
```

**处理样本不平衡**: 使用加权CrossEntropyLoss

### 4. `03_推理应用/推理应用.py` - Streamlit在线推理引擎

**功能**: 提供Web界面，上传EDF文件后实时进行去噪和分期分析

**核心流程**:
```python
def process_data_and_infer(edf_file, txt_file):
    # 1. 读取EDF -> 重采样到100Hz
    # 2. 切片为30秒epoch (3000采样点)
    # 3. V4模型去噪
    # 4. DeepSleepNet分别对原始/去噪信号预测
    # 5. 计算PSD功率谱密度对比
    # 6. 返回分期准确率变化
```

**输出指标**:
- 整体分期准确率变化
- N1浅睡期召回率变化
- N3深睡期召回率变化
- PSD频域对比图

### 5. `04_分期分析/分期分析.py` - 批量分期对比分析

**功能**: 批量处理多个受试者，对比去噪前后的睡眠分期准确率

**核心流程**:
```python
for each_subject:
    # 1. 加载EEG数据
    # 2. V4模型去噪
    # 3. YASA进行睡眠分期（原始信号）
    # 4. YASA进行睡眠分期（去噪信号）
    # 5. 与真值对比，计算各阶段准确率
```

**YASA**: Yet Another Sleep Analyzer - 开源睡眠分期工具

### 6. `05_批量评估/数据评估.py` - 去噪效果评估

**功能**: 计算去噪模型的定量指标

**核心指标**:
```python
def calc_rrmse(clean, est):
    """相对均方根误差 - 越小越好"""
    # RRMSE = RMS(error) / RMS(clean)

def calc_cc(clean, est):
    """皮尔逊相关系数 - 越接近1越好"""
    # CC = corrcoef(clean, denoised)
```

---

## 数据流图

```
原始EEG数据 (EDF)
    │
    ▼
┌─────────────────┐
│  数据预处理      │  重采样、切片、标准化
│  (01_数据预处理) │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  V4去噪模型      │  1D ResNet + SE
│  (512采样点输入) │
└─────────────────┘
    │
    ▼
去噪后EEG信号
    │
    ▼
┌─────────────────┐
│  DeepSleepNet    │  睡眠分期
│  (3000采样点输入)│
└─────────────────┘
    │
    ▼
睡眠分期结果 (Wake/N1/N2/N3/REM)
```

---

## 关键发现（论文核心结论）

### 受害样本分析

通过Grad-CAM可视化发现：

1. **原始信号**: DeepSleepNet注意力集中在Delta慢波区域(0.5-4Hz)，正确识别N3期
2. **V4去噪信号**: Delta慢波被过度平滑，注意力焦点分散，导致N3期被误判为Wake

**典型受害样本**:
- Epoch 4: N3 (95.02%) → Wake (93.74%)
- Epoch 57: N3 (91.54%) → Wake (87.43%)

### PSD能量分析

V4去噪后，Delta频段(0.5-4Hz)能量显著下降，证明过度平滑导致生理特征丢失。

---

## 模型文件位置

```
03_训练模型/
├── V4最优去噪模型.h5          # V4去噪模型
├── DeepSleepNet裁判模型.h5    # 睡眠分期模型
├── 去噪模型v1.h5
├── 去噪模型v2分析版.h5
└── 去噪模型v2最终版.h5
```

---

## 依赖库

```
tensorflow >= 2.x
torch
mne              # EEG数据处理
yasa             # 睡眠分期
numpy, pandas
matplotlib
streamlit        # Web界面
scipy            # 信号处理
sklearn          # 评估指标
```

---

## 快速开始

```python
# 1. 加载去噪模型
from tensorflow.keras.models import load_model
denoise_model = load_model("03_训练模型/V4最优去噪模型.h5", compile=False)

# 2. 加载分期模型
staging_model = load_model("03_训练模型/DeepSleepNet裁判模型.h5", compile=False)

# 3. 处理EEG信号
# 输入: (batch, 512, 1) @ 256Hz
denoised = denoise_model.predict(noisy_eeg)

# 4. 睡眠分期
# 输入: (batch, 3000, 1) @ 100Hz
stages = staging_model.predict(eeg_30s_epochs)
```

---

## 注意事项

1. **采样率差异**: 去噪模型使用256Hz，分期模型使用100Hz，需要重采样
2. **信号长度**: 去噪模型输入512点(2秒)，分期模型输入3000点(30秒)
3. **标准化**: 所有信号需要Z-score标准化
4. **模型加载**: 使用`compile=False`加速加载，因为只做推理

---

*此文档由AI自动生成，用于辅助理解项目代码结构*