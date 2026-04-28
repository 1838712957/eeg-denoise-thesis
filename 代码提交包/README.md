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
   │   ├── DeepSleepNet裁判模型.h5
   │   ├── Baseline.h5
   │   ├── V4_wo_SE.h5
   │   ├── V4_Single_Scale.h5
   │   └── V4_Complete.h5
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

**目的**: 验证模型各组件的贡献

**模型变体**:
1. Baseline: 基础1D CNN
2. V4 w/o SE: 移除SE注意力
3. V4 Single Scale: 单一尺度
4. V4 Complete: 完整模型

**运行命令**:
```bash
cd 项目根目录
python 代码提交包/02_核心实验代码/实验2_消融实验/ablation_study.py
```

**输出文件**: `05_处理结果/消融实验/ablation_results.txt`

**实验流程**:
1. 加载DeepSleepNet裁判模型
2. 加载各变体去噪模型权重
3. 对DREAMS数据集进行去噪处理
4. 使用DeepSleepNet进行睡眠分期
5. 计算各指标并保存结果

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
| Sleep-EDF去噪 | RRMSE | 18.5% |
| 消融实验 | SE模块贡献 | +2.1% |
| 睡眠分期对比 | 准确率提升 | +5.2% |
| 鲁棒性分析 | 高噪声下性能 | 稳定 |
| N3召回率 | 召回率提升 | +12.3% |
| 数据量实验 | 最优数据量 | 80%训练集 |

## 模型架构

V4去噪模型采用编码器-解码器结构：
- **编码器**: 多尺度卷积 (kernel=3,5,7) + SE注意力
- **解码器**: 转置卷积 + 跳跃连接
- **损失函数**: MSE + Delta波能量保持

## 注意事项

1. 实验代码需要在项目根目录下运行
2. 需要预先准备好数据集和模型权重文件
3. 部分实验需要GPU加速（推荐显存≥8GB）
4. 实验结果会保存在 `05_处理结果/` 目录下

## 联系方式

如有问题，请联系: linruzhe@bjut.edu.cn
