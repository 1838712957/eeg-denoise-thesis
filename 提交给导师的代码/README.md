# 睡眠脑电信号去噪项目 - 代码说明

## 项目简介

本项目基于深度学习技术，实现了对睡眠脑电（EEG）信号的去噪处理，并构建了在线推理系统用于临床评估。

## 项目结构

```
提交给导师的代码/
├── README.md                    # 本文件
├── 01_数据预处理/               # 数据预处理脚本
│   ├── 数据检查.py
│   ├── 数据检查2.py
│   ├── 数据预处理.py
│   ├── 通道检查.py
│   └── bootstrap_paths.py       # 路径配置
│
├── 02_模型训练/                 # 模型训练脚本
│   ├── train_denoise_model.py   # 去噪模型训练
│   ├── train_v4_complete.py    # V4完整模型训练
│   └── train_ablation_models.py # 消融实验模型
│
├── 03_推理应用/                 # Streamlit在线推理系统
│   ├── 推理应用.py             # 主程序
│   └── config.toml             # Streamlit配置
│
├── 04_测试分析/                 # 测试和分析脚本
│   ├── 分期测试/
│   │   ├── staging_test.py
│   │   ├── detailed_staging_test.py
│   │   └── deepsleepnet_staging_test.py
│   │
│   ├── 去噪测试/
│   │   ├── test_denoising.py
│   │   ├── simple_test.py
│   │   └── simple_model_test.py
│   │
│   └── 可视化分析/
│       ├── gradcam_analysis.py
│       ├── visualize_ablation.py
│       └── ablation_study.py
│
└── 05_报告生成/                 # 报告生成脚本
    ├── gen_docx.py             # Word报告生成
    ├── generate_asr_raw_comparison.py # ASR/RAW对比图生成
    └── generate_midterm_docx.py # 中期报告生成
```

## 环境依赖

```bash
# 核心依赖
tensorflow>=2.10.0
numpy
pandas
scipy
mne
matplotlib
scikit-learn

# Web框架
streamlit

# 文档生成
python-docx
```

安装命令：
```bash
pip install tensorflow numpy pandas scipy mne matplotlib scikit-learn streamlit python-docx
```

## 快速开始

### 1. 启动在线推理系统（推荐）

直接在终端运行以下命令启动Web界面：

```bash
cd 提交给导师的代码/03_推理应用
streamlit run 推理应用.py
```

启动后浏览器会自动打开 http://localhost:8501

**使用方法：**
1. 在网页左侧上传EDF脑电文件
2. 上传对应的睡眠分期标签TXT文件
3. 点击"启动V4深度去噪与特征实测"按钮
4. 查看分析结果：准确率、N3召回率、Delta能量损失等

### 2. 运行测试脚本

```bash
# 去噪测试
cd 提交给导师的代码/04_测试分析/去噪测试
python test_denoising.py

# 分期测试
cd 提交给导师的代码/04_测试分析/分期测试
python staging_test.py
```

### 3. 生成报告

```bash
cd 提交给导师的代码/05_报告生成
python gen_docx.py
```

## 主要功能

### 在线推理系统功能
- ✅ 数据上传（EDF格式脑电 + TXT标签）
- ✅ V4深度去噪模型推理
- ✅ DeepSleepNet分期预测
- ✅ 整体准确率统计
- ✅ N3召回率分析
- ✅ Delta能量损失计算
- ✅ 频域PSD对比图
- ✅ Grad-CAM注意力分析
- ✅ 受害样本检测
- ✅ 交互式信号查看器

### 模型说明

| 模型 | 说明 |
|------|------|
| V4最优去噪模型 | 最终部署的去噪模型 |
| Baseline | 纯1D CNN基准模型 |
| V4_Complete | 完整架构（多尺度+SE注意力） |
| DeepSleepNet | 裁判模型（睡眠分期） |

## 注意事项

1. **模型文件**：由于模型文件(.h5)较大，未包含在代码包中，需要从原项目 `03_训练模型/` 目录复制以下文件：
   - `V4最优去噪模型.h5`
   - `DeepSleepNet裁判模型.h5`

2. **原始数据**：原始EEG数据(.edf)未包含，需要自行准备

3. **Streamlit**：推荐使用Streamlit在线演示，操作简单效果好

4. **代码文件**：由于文件复制限制，部分代码文件需要手动从原项目对应目录复制：
   - `01_数据预处理/` 目录下的文件 → 复制到 `提交给导师的代码/01_数据预处理/`
   - 其他目录同理

## 替代方案：使用Git仓库

如果你的电脑已安装Git，可以直接使用项目根目录的Git仓库（已初始化并提交）：
- 仓库位置：`c:/毕业论文/.git`
- 已提交：29个核心代码文件
- 未提交：模型文件、原始数据、测试结果

## 现场演示建议

给导师演示时，建议：
1. 直接运行 `streamlit run 推理应用.py`
2. 准备好测试用的EDF文件
3. 展示在线推理系统的各个功能模块
4. 展示ASR/RAW/去噪信号的三者对比图

## 联系方式

如有问题，请联系项目负责人。

---
生成日期：2026年3月30日