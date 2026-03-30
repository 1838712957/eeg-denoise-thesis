const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, 
        AlignmentType, BorderStyle, WidthType, ShadingType } = require('docx');
const fs = require('fs');

// 优化后的文本内容
const sections = [
  {
    title: '一、课题简介与研究背景',
    subsections: [
      {
        title: '1.1 为什么做这个研究',
        paragraphs: [
          '脑电图（EEG）是记录大脑神经元电活动的主要手段之一，在睡眠医学和脑机接口研究中应用广泛。但头皮EEG信号有个让人头疼的特点——幅值极低（微伏级）、非平稳性强，采集时很容易被各种噪声"污染"。生理伪迹（眼电EOG、肌电EMG）、工频干扰、电极接触不良等问题，都会让原始信号质量大打折扣。',
          '深度学习在EEG去噪领域这两年进展很快，一维卷积网络（1D CNN）和残差网络（ResNet）的端到端方案，效果已经超越了传统的线性滤波和ICA盲源分离。但在实际应用于整夜睡眠数据时，我们发现了一个有意思的现象：同一个模型，在不同睡眠阶段的表现差异巨大——N1期去得挺干净，N3期却把慢波给"抹"掉了。'
        ]
      },
      {
        title: '1.2 研究目标',
        paragraphs: [
          '这个课题的核心目标很明确：搭建一套基于深度学习的睡眠伪差矫正系统，能够有效去除EEG中的噪声干扰，同时保护好N3期Delta慢波这类有临床价值的特征。验证时不能只看传统的RRMSE、相关系数这些指标，还要看下游任务（睡眠分期）的表现，因为临床上有意义才是最终标准。另外，搞清楚为什么深度学习模型会在不同睡眠阶段表现出这么大的性能分化，是这次研究要回答的关键问题。'
        ]
      }
    ]
  },
  {
    title: '二、目前已完成的工作',
    subsections: [
      {
        title: '2.1 数据和模型的基础搭建',
        paragraphs: [
          '数据方面，基于Sleep-EDF和DREAMS两个公开数据集完成了预处理流水线：从原始EDF文件中选取EEG通道，重采样到100Hz，按30秒epoch分段，并与睡眠标签对齐。这套流程已经封装成标准化的数据加载接口。',
          '裁判模型方面，部署了DeepSleepNet作为下游任务评估器。这是个CNN+BiLSTM的架构，输入单通道EEG（30秒epoch，100Hz采样率），输出5类分期（W、N1、N2、N3、REM）。在原始信号上的验证准确率大约75%，作为评估基准是够用的。'
        ]
      },
      {
        title: '2.2 模型迭代：从Baseline到V4',
        paragraphs: [
          '去噪模型这块做了几轮迭代，从简单到复杂依次是：'
        ],
        insertTable: 'model'
      }
    ]
  },
  {
    title: '三、核心实验发现',
    subsections: [
      {
        title: '3.1 消融实验数据',
        paragraphs: ['实验数据汇总如下：'],
        insertTable: 'ablation'
      },
      {
        title: '3.2 一个意外的发现',
        paragraphs: [
          '在分析实验结果时，有个数据让我困惑了很久：V4_Complete模型的Delta频段能量保持率高达93.5%，但时域相关系数CC却是-0.68——负值！这意味着什么？频域能量保持得好，不代表时域波形也是对的。',
          '进一步排查后发现，模型输出的信号发生了相位反转（相当于波形上下颠倒），同时幅度异常放大（RRMSE超过140%）。更讽刺的是，最简单的Baseline模型反而表现更好：N3召回率79.09%，比V4_Complete的57.15%高出22个百分点。复杂的SE注意力机制和多尺度大卷积核，反而加剧了形态学的畸变。'
        ]
      },
      {
        title: '3.3 RAW/ASR/去噪信号三方对比',
        paragraphs: [
          '为了更全面地评估，还做了与传统ASR算法的对比实验。从NRR（噪声抑制比）来看，我们的方法比ASR高出不少（比如SC4001E0：89.43% vs 63.62%），但从下游分期准确率来看，效果却参差不齐。',
          '这个结果说明一个问题：传统指标和临床效果之间，可能存在脱节。ASR虽然噪声去除不如深度学习彻底，但它保留了信号的"可分性"，让下游分期器能正常工作。'
        ]
      }
    ]
  },
  {
    title: '四、问题出在哪里',
    subsections: [
      {
        title: '4.1 1D CNN的"频谱偏好"',
        paragraphs: [
          '1D CNN在处理时间序列时有个内在特性——对高频成分更敏感，对低频成分容易过度平滑。这跟卷积核的感受野有关：感受野越大，低频信息保留得越好，但同时也可能把低频慢波"平滑"掉。',
          'N1阶段的EEG本身高频噪声多，模型去噪效果就好；N3阶段的Delta慢波频率低、幅度大，模型反而把它当成"噪声"给压制了。'
        ]
      },
      {
        title: '4.2 训练数据的分布偏差',
        paragraphs: [
          '更根本的原因在于训练数据。EEGdenoiseNet是常用的去噪训练数据集，但里面几乎没有N3期深睡阶段的样本。这意味着模型从来没见过"正常的巨幅低频慢波"，遇到N3阶段的Delta波时，只能把它归类为异常噪声。这是典型的分布外（OOD）问题——训练分布和应用分布不匹配。'
        ]
      }
    ]
  },
  {
    title: '五、Grad-CAM能看到什么',
    subsections: [
      {
        title: '',
        paragraphs: [
          '用Grad-CAM对模型做可解释性分析，发现了一些有趣的规律：',
          '首先，模型对高频噪声区域（>20Hz的EMG成分）关注度很高，热力图均值在0.17-0.23之间。这解释了为什么N1去噪效果好——N1本来就高频噪声多，跟训练数据分布匹配。',
          '其次，在N3的Delta慢波区域，Grad-CAM显示模型把这些低频高幅信号判定为"异常"，要去除的伪迹。因为训练集里没见过这种信号，模型只能按已有经验处理。',
          '最后，模型对相位变化很敏感。N3慢波从负相位转到正相位时，Grad-CAM显示模型会把它识别为"非生理信号"并尝试修正，这直接导致了相位反转问题。'
        ]
      }
    ]
  },
  {
    title: '六、在线推理系统',
    subsections: [
      {
        title: '',
        paragraphs: [
          '基于Streamlit框架搭建了一个轻量级Web应用，名叫"脑电去噪端到端在线推理与临床评估引擎"。核心功能包括：上传EDF脑电文件和睡眠标签，系统自动完成预处理、去噪和分期预测。',
          '结果展示包括：原始vs去噪信号的时域对比、功率谱密度对比、Grad-CAM热力图，以及分期准确率和N3召回率的对比报告。这个系统方便直观地观察模型在不同受试者、不同睡眠阶段的表现，也便于后续的算法迭代验证。系统本地运行地址：http://localhost:8501'
        ]
      }
    ]
  },
  {
    title: '七、后续工作',
    subsections: [
      {
        title: '',
        paragraphs: [
          '接下来两个月主要做几件事：',
          '算法层面：修复相位反转和尺度坍塌问题，可能需要在损失函数中加入相位约束；尝试以更简单的Baseline架构为基础进行改进，而不是继续堆复杂模块。数据层面：考虑引入N3阶段样本进行fine-tune，缓解分布偏差问题。系统层面：完善Web界面的交互体验，优化可视化效果。论文层面：精修图表，整理实验数据，准备撰写。'
        ]
      }
    ]
  },
  {
    title: '八、参考文献',
    subsections: [
      {
        title: '',
        paragraphs: [
          '[1] Supratak A, et al. DeepSleepNet: A simple yet effective approach for automatic sleep stage scoring. 2017.',
          '[2] Mullen T, et al. Real-time modeling and 3D visualization of source dynamics and connectivity using wearable EEG. 2012.',
          '[3] Delorme A, Makeig S. EEGLAB: An open source toolbox for analysis of single-trial EEG dynamics. 2004.',
          '[4] Hu J, et al. Squeeze-and-excitation networks. 2018.',
          '[5] He K, et al. Deep residual learning for image recognition. 2016.'
        ]
      }
    ]
  },
  {
    title: '九、致谢',
    subsections: [
      {
        title: '',
        paragraphs: [
          '感谢指导老师的耐心指导，也感谢实验室同学的讨论和帮助。',
          '报告日期：2026年3月29日    完成进度：约80%'
        ]
      }
    ]
  }
];

// 表格数据
const modelTableData = [
  ['模型版本', '架构特点', '设计意图'],
  ['Baseline', '纯1D CNN，无残差、无注意力', '基准对照'],
  ['V4_wo_SE', '多尺度残差 + 移除SE注意力', '验证注意力机制的作用'],
  ['V4_Single_Scale', '单一尺度(kernel=3) + SE注意力', '验证多尺度的必要性'],
  ['V4_Complete', '多尺度残差 + SE注意力', '完整架构'],
  ['V4最优去噪模型', '基于V4_Complete优化', '最终部署版本']
];

const ablationTableData = [
  ['模型', 'RRMSE(%)', 'CC', 'Delta保持(%)', '准确率(%)', 'N3召回(%)'],
  ['Original', '-', '-', '-', '11.79', '19.31'],
  ['Baseline', '150.30', '-0.69', '68.52', '28.13', '79.09'],
  ['V4_wo_SE', '145.51', '-0.62', '64.62', '21.81', '62.62'],
  ['V4_Single_Scale', '142.89', '-0.60', '71.45', '26.64', '72.56'],
  ['V4_Complete', '152.25', '-0.68', '93.50', '22.47', '57.15']
];

// 创建表格
function createTable(data, widths) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "AAAAAA" };
  const borders = { top: border, bottom: border, left: border, right: border };
  
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: widths,
    rows: data.map((row, rowIndex) => {
      return new TableRow({
        children: row.map((cell, cellIndex) => {
          return new TableCell({
            borders,
            width: { size: widths[cellIndex], type: WidthType.DXA },
            shading: rowIndex === 0 ? { fill: "E8E8E8", type: ShadingType.CLEAR } : undefined,
            margins: { top: 80, bottom: 80, left: 120, right: 120 },
            children: [new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [new TextRun({ text: cell, size: 21, bold: rowIndex === 0 })]
            })]
          });
        })
      });
    })
  });
}

// 构建文档内容
const docChildren = [];

// 标题
docChildren.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 200 },
  children: [new TextRun({ text: '毕业设计中期检查报告', size: 44, bold: true, font: '黑体' })]
}));
docChildren.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 400 },
  children: [new TextRun({ text: '基于深度学习的睡眠脑电信号去噪：跨域适应性问题研究与临床数据挽救', size: 32, font: '宋体' })]
}));

// 遍历章节
sections.forEach((section) => {
  // 章节标题
  docChildren.push(new Paragraph({
    spacing: { before: 360, after: 200 },
    children: [new TextRun({ text: section.title, size: 32, bold: true, font: '黑体' })]
  }));
  
  section.subsections.forEach((sub) => {
    // 小节标题（如果有）
    if (sub.title) {
      docChildren.push(new Paragraph({
        spacing: { before: 240, after: 160 },
        children: [new TextRun({ text: sub.title, size: 28, bold: true, font: '黑体' })]
      }));
    }
    
    // 段落
    sub.paragraphs.forEach(para => {
      docChildren.push(new Paragraph({
        spacing: { after: 200, line: 360 },
        indent: { firstLine: 480 },
        children: [new TextRun({ text: para, size: 24, font: '宋体' })]
      }));
    });
    
    // 插入表格
    if (sub.insertTable === 'model') {
      docChildren.push(createTable(modelTableData, [2500, 4000, 2860]));
      docChildren.push(new Paragraph({ spacing: { after: 200 }, children: [] }));
    }
    if (sub.insertTable === 'ablation') {
      docChildren.push(createTable(ablationTableData, [1800, 1500, 1500, 1800, 1500, 1500]));
      docChildren.push(new Paragraph({ spacing: { after: 200 }, children: [] }));
    }
  });
});

// 创建文档
const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: '宋体', size: 24 }
      }
    }
  },
  sections: [{
    properties: {
      page: {
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    children: docChildren
  }]
});

// 保存文档
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync('01_论文文档/中期报告_V4_优化版.docx', buffer);
  console.log('Document created: 01_论文文档/中期报告_V4_优化版.docx');
});
