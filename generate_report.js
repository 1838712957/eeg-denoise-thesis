const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
        Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType, 
        ShadingType, PageNumber, PageBreak, LevelFormat, VerticalAlign } = require('docx');
const fs = require('fs');

// 读取图片
const ablationComparison = fs.readFileSync('C:\\毕业论文\\05_处理结果\\消融实验\\ablation_all_models_comparison.png');
const ablationVisual = fs.readFileSync('C:\\毕业论文\\05_处理结果\\消融实验\\ablation_visual_comparison.png');
const comparison4001 = fs.readFileSync('C:\\毕业论文\\05_处理结果\\对比图片\\SC4001E0_Comparison.png');
const comparison4002 = fs.readFileSync('C:\\毕业论文\\05_处理结果\\对比图片\\SC4002E0_Comparison.png');
const threeWayComparison = fs.readFileSync('C:\\毕业论文\\05_处理结果\\V4输出\\subject17_3Way_Comparison.png');
const psdEvidence = fs.readFileSync('C:\\毕业论文\\05_处理结果\\V4输出\\subject17_PSD_Evidence.png');
const gradCAM4011 = fs.readFileSync('C:\\毕业论文\\05_处理结果\\GradCAM分析\\SC4011E0_GradCAM.png');
const victimHeatmap = fs.readFileSync('C:\\毕业论文\\05_处理结果\\受害样本分析\\Victim_Epoch57_HeatmapComparison.png');

// 表格边框样式
const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const headerShading = { fill: "2E75B6", type: ShadingType.CLEAR };
const altRowShading = { fill: "F2F2F2", type: ShadingType.CLEAR };

// 创建文档
const doc = new Document({
  styles: {
    default: { document: { run: { font: "SimSun", size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "SimHei", color: "2E75B6" },
        paragraph: { spacing: { before: 400, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "SimHei", color: "404040" },
        paragraph: { spacing: { before: 300, after: 150 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "SimHei", color: "595959" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 } },
    ]
  },
  numbering: {
    config: [
      { reference: "numbers",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ]
  },
  sections: [{
    properties: {
      page: { 
        size: { width: 11906, height: 16838 }, // A4
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "毕业设计中期检查报告", font: "SimSun", size: 20, color: "808080" })]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "第 ", font: "SimSun", size: 20 }),
            new TextRun({ children: [PageNumber.CURRENT], font: "SimSun", size: 20 }),
            new TextRun({ text: " 页", font: "SimSun", size: 20 })
          ]
        })]
      })
    },
    children: [
      // 封面标题
      new Paragraph({ spacing: { before: 2000 } }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "毕业设计中期检查报告", font: "SimHei", size: 52, bold: true, color: "1F4E79" })]
      }),
      new Paragraph({ spacing: { before: 600 } }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "基于深度学习的EEG信号去噪与睡眠分期研究", font: "SimHei", size: 32, color: "404040" })]
      }),
      new Paragraph({ spacing: { before: 2000 } }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "报告日期：2026年3月26日", font: "SimSun", size: 24 })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "完成进度：约70%", font: "SimSun", size: 24 })]
      }),
      
      // 分页
      new Paragraph({ children: [new PageBreak()] }),
      
      // 一、课题简介与前期目标回顾
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("一、课题简介与前期目标回顾")] }),
      
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("1.1 研究背景")] }),
      new Paragraph({
        spacing: { after: 200 },
        children: [new TextRun({ text: "脑电图（EEG）信号去噪在睡眠分期中具有至关重要的作用。然而，传统去噪评估方法存在严重局限性：", font: "SimSun", size: 24 })]
      }),
      
      // 问题卡片
      new Table({
        width: { size: 9060, type: WidthType.DXA },
        columnWidths: [9060],
        rows: [
          new TableRow({
            children: [
              new TableCell({
                borders,
                shading: { fill: "FFF2CC", type: ShadingType.CLEAR },
                margins: { top: 120, bottom: 120, left: 200, right: 200 },
                children: [
                  new Paragraph({ children: [new TextRun({ text: "⚠ 传统指标的欺骗性", bold: true, font: "SimHei", size: 24 })] }),
                  new Paragraph({ children: [new TextRun({ text: "仅关注RRMSE、CC等数学指标，无法反映波形的生物学语义", font: "SimSun", size: 22 })] })
                ]
              })
            ]
          }),
          new TableRow({
            children: [
              new TableCell({
                borders,
                shading: { fill: "FCE4D6", type: ShadingType.CLEAR },
                margins: { top: 120, bottom: 120, left: 200, right: 200 },
                children: [
                  new Paragraph({ children: [new TextRun({ text: "⚠ 临床特征的丢失", bold: true, font: "SimHei", size: 24 })] }),
                  new Paragraph({ children: [new TextRun({ text: "去噪后的信号可能「数学上干净」但「临床上无用」", font: "SimSun", size: 22 })] })
                ]
              })
            ]
          }),
          new TableRow({
            children: [
              new TableCell({
                borders,
                shading: { fill: "E2EFDA", type: ShadingType.CLEAR },
                margins: { top: 120, bottom: 120, left: 200, right: 200 },
                children: [
                  new Paragraph({ children: [new TextRun({ text: "⚠ 下游任务性能下降", bold: true, font: "SimHei", size: 24 })] }),
                  new Paragraph({ children: [new TextRun({ text: "过度平滑导致关键生理特征（如N3期Delta慢波）被误判为噪声", font: "SimSun", size: 22 })] })
                ]
              })
            ]
          })
        ]
      }),
      
      new Paragraph({ spacing: { before: 300 } }),
      
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("1.2 原定目标")] }),
      new Paragraph({
        spacing: { after: 100 },
        children: [new TextRun({ text: "设计并实现一个基于深度学习的睡眠伪差矫正系统，要求：", font: "SimSun", size: 24 })]
      }),
      new Paragraph({ numbering: { reference: "numbers", level: 0 }, children: [new TextRun({ text: "有效去除EEG信号中的噪声干扰", font: "SimSun", size: 24 })] }),
      new Paragraph({ numbering: { reference: "numbers", level: 0 }, children: [new TextRun({ text: "尽可能保护下游临床分期（如N3期Delta慢波）的形态特征", font: "SimSun", size: 24 })] }),
      new Paragraph({ numbering: { reference: "numbers", level: 0 }, children: [new TextRun({ text: "建立多维评估体系，验证去噪效果的临床有效性", font: "SimSun", size: 24 })] }),
      
      // 分页
      new Paragraph({ children: [new PageBreak()] }),
      
      // 二、目前已完成的工作
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("二、目前已完成的工作")] }),
      
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("2.1 数据流水线与基座模型搭建")] }),
      new Paragraph({
        spacing: { after: 200 },
        children: [new TextRun({ text: "数据处理流程：", font: "SimHei", size: 24, bold: true })]
      }),
      
      // 流程图表格
      new Table({
        width: { size: 9060, type: WidthType.DXA },
        columnWidths: [1812, 1812, 1812, 1812, 1812],
        rows: [
          new TableRow({
            children: [
              new TableCell({ borders, shading: { fill: "DEEAF6", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER,
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "原始EDF文件", font: "SimSun", size: 20 })] })] }),
              new TableCell({ borders, shading: { fill: "E2EFDA", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER,
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "通道选择(EEG)", font: "SimSun", size: 20 })] })] }),
              new TableCell({ borders, shading: { fill: "FFF2CC", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER,
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "重采样(100Hz)", font: "SimSun", size: 20 })] })] }),
              new TableCell({ borders, shading: { fill: "FCE4D6", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER,
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "分段(30s epoch)", font: "SimSun", size: 20 })] })] }),
              new TableCell({ borders, shading: { fill: "EDEDED", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER,
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "标签对齐", font: "SimSun", size: 20 })] })] }),
            ]
          })
        ]
      }),
      
      new Paragraph({ spacing: { before: 200 } }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "完成了基于Sleep-EDF和DREAMS数据集的数据预处理、重采样与对齐", font: "SimSun", size: 24 })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "实现了自动化的标签解析和epoch分割", font: "SimSun", size: 24 })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "建立了标准化的数据加载接口", font: "SimSun", size: 24 })] }),
      
      new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun("裁判模型部署")] }),
      new Paragraph({
        spacing: { after: 200 },
        children: [new TextRun({ text: "成功部署了DeepSleepNet作为下游任务的【裁判模型】，用于评估去噪信号的临床有效性。", font: "SimSun", size: 24 })]
      }),
      
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("2.2 多版本去噪模型的迭代与消融实验")] }),
      new Paragraph({
        spacing: { after: 200 },
        children: [new TextRun({ text: "模型架构演进：从Baseline逐步引入SE注意力、多尺度卷积等模块，并进行了系统的消融实验。", font: "SimSun", size: 24 })]
      }),
      
      // 消融实验图
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new ImageRun({ type: "png", data: ablationComparison, transformation: { width: 550, height: 350 },
          altText: { title: "消融实验对比", description: "所有模型变体的性能对比", name: "ablation_comparison" } })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 300 },
        children: [new TextRun({ text: "图1：消融实验结果对比——所有模型变体的综合性能评估", font: "SimSun", size: 20, italics: true, color: "595959" })]
      }),
      
      // 分页
      new Paragraph({ children: [new PageBreak()] }),
      
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("2.3 核心实验发现（高光部分）")] }),
      
      // 发现A
      new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun("发现A：频域保真与时域失真的冲突")] }),
      new Table({
        width: { size: 9060, type: WidthType.DXA },
        columnWidths: [4530, 4530],
        rows: [
          new TableRow({
            children: [
              new TableCell({ borders, shading: { fill: "E2EFDA", type: ShadingType.CLEAR }, margins: { top: 100, bottom: 100, left: 150, right: 150 },
                children: [new Paragraph({ children: [new TextRun({ text: "Delta能量93.5%保持", font: "SimHei", size: 22, bold: true, color: "2E75B6" })] }),
                          new Paragraph({ children: [new TextRun({ text: "频域「看起来很好」", font: "SimSun", size: 20 })] })] }),
              new TableCell({ borders, shading: { fill: "FCE4D6", type: ShadingType.CLEAR }, margins: { top: 100, bottom: 100, left: 150, right: 150 },
                children: [new Paragraph({ children: [new TextRun({ text: "CC = -0.68", font: "SimHei", size: 22, bold: true, color: "C00000" })] }),
                          new Paragraph({ children: [new TextRun({ text: "时域「完全反转」", font: "SimSun", size: 20 })] })] }),
            ]
          })
        ]
      }),
      new Paragraph({
        spacing: { before: 100, after: 200 },
        children: [new TextRun({ text: "关键发现：V4_Complete模型保留了高达93.5%的Delta频段能量，但时域相关系数（CC）为-0.68（负值！）", font: "SimSun", size: 24 })]
      }),
      
      // 发现B
      new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun("发现B：信号相位反转问题")] }),
      
      // 三对比图
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new ImageRun({ type: "png", data: threeWayComparison, transformation: { width: 550, height: 280 },
          altText: { title: "三对比图", description: "时域波形三对比分析", name: "three_way" } })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 200 },
        children: [new TextRun({ text: "图2：时域波形三对比分析——验证相位反转假设", font: "SimSun", size: 20, italics: true, color: "595959" })]
      }),
      
      new Paragraph({ children: [new TextRun({ text: "从时域波形可视化图中可以清晰看到：", font: "SimSun", size: 24 })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "左列：原始信号（蓝）与V4_Complete输出（红）对比，CC≈-0.58", font: "SimSun", size: 24 })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "中列：将去噪信号反转后，与原始信号高度吻合（验证了相位反转假设）", font: "SimSun", size: 24 })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "右列：功率谱密度对比，Delta频段能量确实得到保留", font: "SimSun", size: 24 })] }),
      
      // PSD证据图
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new ImageRun({ type: "png", data: psdEvidence, transformation: { width: 500, height: 300 },
          altText: { title: "PSD证据", description: "功率谱密度分析证据", name: "psd_evidence" } })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 300 },
        children: [new TextRun({ text: "图3：功率谱密度分析——频域能量保留的证据", font: "SimSun", size: 20, italics: true, color: "595959" })]
      }),
      
      // 分页
      new Paragraph({ children: [new PageBreak()] }),
      
      // 发现C
      new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun("发现C：注意力的副作用")] }),
      new Paragraph({
        spacing: { after: 200 },
        children: [new TextRun({ text: "Baseline模型反而表现更好：复杂的SE注意力机制和多尺度大卷积核，反而加剧了形态学的畸变。", font: "SimSun", size: 24 })]
      }),
      
      // 消融可视化
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new ImageRun({ type: "png", data: ablationVisual, transformation: { width: 550, height: 350 },
          altText: { title: "消融可视化", description: "消融实验可视化对比", name: "ablation_visual" } })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 300 },
        children: [new TextRun({ text: "图4：消融实验可视化——模型复杂度与性能的关系", font: "SimSun", size: 20, italics: true, color: "595959" })]
      }),
      
      // 时域波形对比
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("2.4 时域波形可视化分析")] }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new ImageRun({ type: "png", data: comparison4001, transformation: { width: 550, height: 280 },
          altText: { title: "SC4001对比", description: "SC4001时域对比", name: "comparison_4001" } })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 200 },
        children: [new TextRun({ text: "图5：SC4001受试者EEG信号时域对比分析", font: "SimSun", size: 20, italics: true, color: "595959" })]
      }),
      
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new ImageRun({ type: "png", data: comparison4002, transformation: { width: 550, height: 280 },
          altText: { title: "SC4002对比", description: "SC4002时域对比", name: "comparison_4002" } })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 300 },
        children: [new TextRun({ text: "图6：SC4002受试者EEG信号时域对比分析", font: "SimSun", size: 20, italics: true, color: "595959" })]
      }),
      
      // 分页
      new Paragraph({ children: [new PageBreak()] }),
      
      // 三、存在的主要问题与难点
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("三、存在的主要问题与难点")] }),
      
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("3.1 指标欺骗性问题")] }),
      new Paragraph({
        spacing: { after: 200 },
        children: [new TextRun({ text: "问题描述：传统的去噪评价指标（MSE、RRMSE）在面对医疗生理信号时，无法有效反映波形的生物学语义。", font: "SimSun", size: 24 })]
      }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "RRMSE显示去噪效果「良好」，但下游任务准确率下降", font: "SimSun", size: 24 })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "CC为负值，说明信号被反转，但频域能量指标无法捕捉这一致命问题", font: "SimSun", size: 24 })] }),
      
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("3.2 复杂网络的副作用")] }),
      new Paragraph({
        spacing: { after: 200 },
        children: [new TextRun({ text: "问题描述：引入SE注意力机制和多尺度大卷积核，反而加剧了形态学的畸变。", font: "SimSun", size: 24 })]
      }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "Baseline（最简单架构）在N3召回率上表现最好", font: "SimSun", size: 24 })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "V4_Complete（最复杂架构）在准确率上表现最差", font: "SimSun", size: 24 })] }),
      
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("3.3 归一化与反归一化缺陷")] }),
      new Paragraph({
        spacing: { after: 200 },
        children: [new TextRun({ text: "问题描述：模型训练时的归一化策略可能导致推理时的尺度坍塌和相位反转。", font: "SimSun", size: 24 })]
      }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "输出信号幅度异常放大（RRMSE > 140%）", font: "SimSun", size: 24 })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "相位完全反转（CC < 0）", font: "SimSun", size: 24 })] }),
      
      // GradCAM分析
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("3.4 GradCAM可解释性分析")] }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new ImageRun({ type: "png", data: gradCAM4011, transformation: { width: 500, height: 320 },
          altText: { title: "GradCAM", description: "GradCAM热力图分析", name: "gradcam" } })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 300 },
        children: [new TextRun({ text: "图7：GradCAM热力图——模型关注区域的可视化分析", font: "SimSun", size: 20, italics: true, color: "595959" })]
      }),
      
      // 分页
      new Paragraph({ children: [new PageBreak()] }),
      
      // 四、下一步工作计划
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("四、下一步工作计划")] }),
      
      // 时间线表格
      new Table({
        width: { size: 9060, type: WidthType.DXA },
        columnWidths: [2000, 7060],
        rows: [
          new TableRow({
            children: [
              new TableCell({ borders, shading: headerShading, margins: { top: 100, bottom: 100, left: 150, right: 150 },
                children: [new Paragraph({ children: [new TextRun({ text: "时间阶段", font: "SimHei", size: 22, bold: true, color: "FFFFFF" })] })] }),
              new TableCell({ borders, shading: headerShading, margins: { top: 100, bottom: 100, left: 150, right: 150 },
                children: [new Paragraph({ children: [new TextRun({ text: "工作内容", font: "SimHei", size: 22, bold: true, color: "FFFFFF" })] })] }),
            ]
          }),
          new TableRow({
            children: [
              new TableCell({ borders, shading: altRowShading, margins: { top: 100, bottom: 100, left: 150, right: 150 },
                children: [new Paragraph({ children: [new TextRun({ text: "第1-2周", font: "SimSun", size: 22, bold: true })] })] }),
              new TableCell({ borders, shading: altRowShading, margins: { top: 100, bottom: 100, left: 150, right: 150 },
                children: [new Paragraph({ children: [new TextRun({ text: "修复与完善算法层面：归一化策略优化、损失函数改进", font: "SimSun", size: 22 })] })] }),
            ]
          }),
          new TableRow({
            children: [
              new TableCell({ borders, margins: { top: 100, bottom: 100, left: 150, right: 150 },
                children: [new Paragraph({ children: [new TextRun({ text: "第3-4周", font: "SimSun", size: 22, bold: true })] })] }),
              new TableCell({ borders, margins: { top: 100, bottom: 100, left: 150, right: 150 },
                children: [new Paragraph({ children: [new TextRun({ text: "系统设计与实现：基于Streamlit框架开发可视化交互网页", font: "SimSun", size: 22 })] })] }),
            ]
          }),
          new TableRow({
            children: [
              new TableCell({ borders, shading: altRowShading, margins: { top: 100, bottom: 100, left: 150, right: 150 },
                children: [new Paragraph({ children: [new TextRun({ text: "第5-6周", font: "SimSun", size: 22, bold: true })] })] }),
              new TableCell({ borders, shading: altRowShading, margins: { top: 100, bottom: 100, left: 150, right: 150 },
                children: [new Paragraph({ children: [new TextRun({ text: "论文撰写与图表精修", font: "SimSun", size: 22 })] })] }),
            ]
          }),
        ]
      }),
      
      new Paragraph({ spacing: { before: 300 } }),
      new Paragraph({ children: [new TextRun({ text: "系统功能规划：", font: "SimHei", size: 24, bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "EEG数据上传（支持EDF格式）", font: "SimSun", size: 24 })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "去噪波形对比展示（时域+频域）", font: "SimSun", size: 24 })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "在线睡眠分期预测", font: "SimSun", size: 24 })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "Grad-CAM可解释性分析", font: "SimSun", size: 24 })] }),
      
      // 分页
      new Paragraph({ children: [new PageBreak()] }),
      
      // 五、阶段性成果清单
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("五、阶段性成果清单")] }),
      
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("5.1 代码成果")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "完整的数据预处理流水线（Sleep-EDF + DREAMS）", font: "SimSun", size: 24 })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "4个版本的去噪模型实现（Baseline, V2, V3, V4_Complete）", font: "SimSun", size: 24 })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "消融实验框架与评估脚本", font: "SimSun", size: 24 })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "DeepSleepNet裁判模型集成", font: "SimSun", size: 24 })] }),
      
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("5.2 模型成果")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "训练好的多版本去噪模型权重", font: "SimSun", size: 24 })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "预训练的DeepSleepNet分期模型", font: "SimSun", size: 24 })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: "完整的消融实验数据集", font: "SimSun", size: 24 })] }),
      
      // 六、参考文献
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("六、参考文献")] }),
      new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "[1] Supratak A, Dong H, Wu C, et al. DeepSleepNet: A model for automatic sleep stage scoring based on raw single-channel EEG. 2017.", font: "SimSun", size: 22 })] }),
      new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "[2] Mullen T, Kothe C, Chi Y M, et al. Real-time modeling and 3D visualization of source dynamics. 2012.", font: "SimSun", size: 22 })] }),
      new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "[3] Delorme A, Makeig S. EEGLAB: an open source toolbox for analysis of single-trial EEG dynamics. 2004.", font: "SimSun", size: 22 })] }),
      new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "[4] Hu J, Shen L, Sun G. Squeeze-and-excitation networks. 2018.", font: "SimSun", size: 22 })] }),
      new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "[5] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition. 2016.", font: "SimSun", size: 22 })] }),
      
      // 七、致谢
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("七、致谢")] }),
      new Paragraph({
        spacing: { after: 300 },
        children: [new TextRun({ text: "感谢指导教师的悉心指导，感谢实验室同学的帮助与支持。特别感谢开源社区提供的EEGLAB、MNE-Python等工具包，为本研究的顺利开展提供了重要支持。", font: "SimSun", size: 24 })]
      }),
      
      // 受害样本分析
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("附录：受害样本分析")] }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new ImageRun({ type: "png", data: victimHeatmap, transformation: { width: 550, height: 350 },
          altText: { title: "受害样本", description: "受害样本热力图对比", name: "victim_heatmap" } })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 200 },
        children: [new TextRun({ text: "图8：受害样本热力图对比——分析模型在困难样本上的表现", font: "SimSun", size: 20, italics: true, color: "595959" })]
      }),
    ]
  }]
});

// 生成文档
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync('C:\\毕业论文\\中期报告_美化版.docx', buffer);
  console.log('文档生成成功：中期报告_美化版.docx');
});
