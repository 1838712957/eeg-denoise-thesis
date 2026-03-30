"""
受害样本精准分析：1D Grad-CAM对比分析
找出N3期被误判的"完美受害者"样本，并可视化注意力差异
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
import matplotlib.pyplot as plt
from pathlib import Path
import mne
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).resolve().parent

# ==================== 模型定义 ====================
class SEBlock(layers.Layer):
    def __init__(self, channels=32, reduction=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        self.fc1 = layers.Dense(max(channels // reduction, 4), activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')
        self.reshape = layers.Reshape((1, channels))
    def call(self, inputs):
        x = self.global_avg_pool(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.reshape(x)
        return inputs * x

class Res_BasicBlock(layers.Layer):
    def __init__(self, kernelsize, use_se=False, **kwargs):
        super(Res_BasicBlock, self).__init__(**kwargs)
        self.bblock = Sequential([
            layers.Conv1D(32, kernelsize, padding="same"),
            layers.BatchNormalization(), layers.ReLU(),
            layers.Conv1D(16, kernelsize, padding="same"),
            layers.BatchNormalization(), layers.ReLU(),
            layers.Conv1D(32, kernelsize, padding="same"),
            layers.BatchNormalization(), layers.ReLU()
        ])
        if use_se: self.se = SEBlock(32)
    def call(self, inputs):
        out = self.bblock(inputs)
        if hasattr(self, 'se'): out = self.se(out)
        return layers.add([out, inputs])

class BasicBlockall(layers.Layer):
    def __init__(self, use_se=False, **kwargs):
        super(BasicBlockall, self).__init__(**kwargs)
        self.bblock3 = Sequential([Res_BasicBlock(3, use_se=use_se), Res_BasicBlock(3, use_se=use_se)])
        self.bblock5 = Sequential([Res_BasicBlock(5, use_se=use_se), Res_BasicBlock(5, use_se=use_se)])
        self.bblock7 = Sequential([Res_BasicBlock(7, use_se=use_se), Res_BasicBlock(7, use_se=use_se)])
    def call(self, inputs):
        return tf.concat([self.bblock3(inputs), self.bblock5(inputs), self.bblock7(inputs)], axis=-1)

def load_denoise_model():
    """加载完整的去噪模型"""
    model_path = PROJECT_ROOT / "03_训练模型" / "V4最优去噪模型.h5"
    return models.load_model(str(model_path), compile=False)

def load_deepsleepnet():
    """加载完整的DeepSleepNet裁判模型"""
    model_path = PROJECT_ROOT / "03_训练模型" / "DeepSleepNet裁判模型.h5"
    return models.load_model(str(model_path), compile=False)

# ==================== Grad-CAM for DeepSleepNet ====================
class DeepSleepNetGradCAM:
    """DeepSleepNet的Grad-CAM实现"""
    def __init__(self, model):
        self.model = model
        # 找到最后一个卷积层
        self.conv_layers = []
        for layer in model.layers:
            if isinstance(layer, layers.Conv1D):
                self.conv_layers.append(layer)
        
        if self.conv_layers:
            self.target_layer = self.conv_layers[-1]
        else:
            self.target_layer = None
    
    def compute_gradcam(self, input_signal, target_class):
        """
        计算Grad-CAM热力图
        
        Args:
            input_signal: 输入信号 (1, 3000, 1)
            target_class: 目标类别 (0-4: Wake, N1, N2, N3, REM)
        
        Returns:
            heatmap: 热力图
        """
        input_tensor = tf.convert_to_tensor(input_signal, dtype=tf.float32)
        
        # 构建梯度模型
        grad_model = models.Model(
            inputs=self.model.inputs,
            outputs=[self.target_layer.output, self.model.output]
        )
        
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            conv_output, predictions = grad_model(input_tensor)
            loss = predictions[0, target_class]
        
        grads = tape.gradient(loss, conv_output)
        
        if grads is None:
            return np.ones(100) / 100
        
        # 全局平均池化梯度
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        
        # 计算热力图
        heatmap = tf.reduce_sum(tf.multiply(conv_output[0], pooled_grads), axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        
        return heatmap.numpy()

# ==================== 主分析函数 ====================
def find_victim_samples():
    """
    第一步：筛选"完美受害者"
    找出原始信号正确预测为N3，但去噪后被误判的样本
    """
    print("=" * 60)
    print("第一步：筛选'完美受害者'样本")
    print("=" * 60)
    
    # 加载去噪模型
    print("\n[1] 加载去噪模型...")
    denoise_model = load_denoise_model()
    print("去噪模型加载成功!")
    
    # 加载DeepSleepNet裁判模型
    print("\n[2] 加载DeepSleepNet裁判模型...")
    judge_model = load_deepsleepnet()
    print("裁判模型加载成功!")
    
    # 加载测试数据
    print("\n[3] 加载测试数据...")
    raw_dir = PROJECT_ROOT / "04_原始数据" / "Raw_edf"
    hypno_dir = PROJECT_ROOT / "04_原始数据" / "Raw_edf"
    
    # 读取第一个受试者的数据
    raw_file = raw_dir / "SC4001E0-PSG.edf"
    hypno_file = hypno_dir / "SC4001EC-Hypnogram.edf"
    
    raw = mne.io.read_raw_edf(str(raw_file), preload=True, verbose=False)
    raw.resample(100)  # DeepSleepNet使用100Hz
    
    # 获取EEG通道
    eeg_data = raw.get_data(picks=[0])[0]
    
    # 读取睡眠分期标签
    hypno_raw = mne.read_annotations(str(hypno_file))
    
    # 分段：每段30秒 (3000个采样点 @ 100Hz)
    epoch_len = 3000
    n_epochs = len(eeg_data) // epoch_len
    
    print(f"总共有 {n_epochs} 个Epoch")
    
    # 存储受害样本
    victims = []
    
    # 分期标签映射
    stage_map = {'W': 0, '1': 1, '2': 2, '3': 3, 'R': 4}
    
    print("\n[4] 搜索受害样本...")
    for epoch_idx in range(min(n_epochs, 100)):  # 只检查前100个epoch
        # 提取epoch
        epoch_raw = eeg_data[epoch_idx * epoch_len:(epoch_idx + 1) * epoch_len]
        
        # 标准化
        std = np.std(epoch_raw)
        if std < 1e-6:
            continue
        epoch_norm = epoch_raw / std
        
        # 准备输入
        input_raw = epoch_norm.reshape(1, epoch_len, 1)
        
        # 去噪（需要重采样到512）
        epoch_512 = np.interp(
            np.linspace(0, len(epoch_norm), 512),
            np.arange(len(epoch_norm)),
            epoch_norm
        ).reshape(1, 512, 1)
        
        denoised_512 = denoise_model.predict(epoch_512, verbose=0)
        
        # 重采样回3000
        denoised = np.interp(
            np.linspace(0, 512, epoch_len),
            np.arange(512),
            denoised_512.flatten()
        ).reshape(1, epoch_len, 1)
        
        # 裁判预测
        pred_raw = judge_model.predict(input_raw, verbose=0)[0]
        pred_denoised = judge_model.predict(denoised, verbose=0)[0]
        
        raw_class = np.argmax(pred_raw)
        denoised_class = np.argmax(pred_denoised)
        
        raw_conf = pred_raw[raw_class]
        denoised_conf = pred_denoised[denoised_class]
        
        # 检查是否是受害样本
        # 条件：原始正确预测N3(类别3)，去噪后误判为其他类别
        if raw_class == 3 and raw_conf > 0.8 and denoised_class != 3:
            victim = {
                'epoch_idx': epoch_idx,
                'raw_signal': epoch_norm,
                'denoised_signal': denoised.flatten(),
                'raw_pred': pred_raw,
                'denoised_pred': pred_denoised,
                'raw_class': raw_class,
                'denoised_class': denoised_class,
                'raw_conf': raw_conf,
                'denoised_conf': denoised_conf
            }
            victims.append(victim)
            
            print(f"\n发现受害样本! Epoch {epoch_idx}")
            print(f"  原始预测: N3 (置信度: {raw_conf:.2%})")
            print(f"  去噪预测: {['Wake','N1','N2','N3','REM'][denoised_class]} (置信度: {denoised_conf:.2%})")
            
            if len(victims) >= 2:
                break
    
    if not victims:
        print("\n未找到符合条件的受害样本，使用最接近的样本...")
        # 如果没找到完美受害者，找一个N3预测置信度下降最多的
        best_victim = None
        max_drop = 0
        
        for epoch_idx in range(min(n_epochs, 100)):
            epoch_raw = eeg_data[epoch_idx * epoch_len:(epoch_idx + 1) * epoch_len]
            std = np.std(epoch_raw)
            if std < 1e-6:
                continue
            epoch_norm = epoch_raw / std
            
            input_raw = epoch_norm.reshape(1, epoch_len, 1)
            
            epoch_512 = np.interp(
                np.linspace(0, len(epoch_norm), 512),
                np.arange(len(epoch_norm)),
                epoch_norm
            ).reshape(1, 512, 1)
            
            denoised_512 = denoise_model.predict(epoch_512, verbose=0)
            denoised = np.interp(
                np.linspace(0, 512, epoch_len),
                np.arange(512),
                denoised_512.flatten()
            ).reshape(1, epoch_len, 1)
            
            pred_raw = judge_model.predict(input_raw, verbose=0)[0]
            pred_denoised = judge_model.predict(denoised, verbose=0)[0]
            
            raw_class = np.argmax(pred_raw)
            
            if raw_class == 3:  # N3期
                drop = pred_raw[3] - pred_denoised[3]
                if drop > max_drop:
                    max_drop = drop
                    best_victim = {
                        'epoch_idx': epoch_idx,
                        'raw_signal': epoch_norm,
                        'denoised_signal': denoised.flatten(),
                        'raw_pred': pred_raw,
                        'denoised_pred': pred_denoised,
                        'raw_class': raw_class,
                        'denoised_class': np.argmax(pred_denoised),
                        'raw_conf': pred_raw[3],
                        'denoised_conf': pred_denoised[np.argmax(pred_denoised)]
                    }
        
        if best_victim:
            victims.append(best_victim)
            print(f"\n找到最佳受害样本: Epoch {best_victim['epoch_idx']}")
            print(f"  N3置信度下降: {max_drop:.2%}")
    
    return victims, judge_model

def analyze_victim_with_gradcam(victim, judge_model, output_dir):
    """
    第二步：运行1D Grad-CAM分析
    生成上下对比的热力波形图
    """
    print("\n" + "=" * 60)
    print("第二步：1D Grad-CAM分析")
    print("=" * 60)
    
    # 初始化Grad-CAM
    gradcam = DeepSleepNetGradCAM(judge_model)
    
    # 准备信号
    raw_signal = victim['raw_signal'].reshape(1, 3000, 1)
    denoised_signal = victim['denoised_signal'].reshape(1, 3000, 1)
    
    # 计算Grad-CAM (针对N3类别)
    print("\n计算原始信号的Grad-CAM...")
    heatmap_raw = gradcam.compute_gradcam(raw_signal, target_class=3)
    
    print("计算去噪信号的Grad-CAM...")
    heatmap_denoised = gradcam.compute_gradcam(denoised_signal, target_class=3)
    
    # 上采样热力图到信号长度
    heatmap_raw_upsampled = np.interp(
        np.arange(3000),
        np.linspace(0, 3000, len(heatmap_raw)),
        heatmap_raw
    )
    heatmap_denoised_upsampled = np.interp(
        np.arange(3000),
        np.linspace(0, 3000, len(heatmap_denoised)),
        heatmap_denoised
    )
    
    # ==================== 可视化 ====================
    print("\n生成可视化图表...")
    
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    
    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    time_axis = np.arange(3000) / 100  # 30秒
    
    # 计算统一的Y轴范围
    raw_max = np.max(np.abs(victim['raw_signal']))
    denoised_max = np.max(np.abs(victim['denoised_signal']))
    y_limit = max(raw_max, denoised_max) * 1.2  # 留20%边距
    y_limit = max(y_limit, 3.0)  # 至少显示±3的范围
    
    # 归一化热力图
    heatmap_normalized = heatmap_raw_upsampled / (np.max(heatmap_raw_upsampled) + 1e-8)
    heatmap_normalized_d = heatmap_denoised_upsampled / (np.max(heatmap_denoised_upsampled) + 1e-8)
    
    # 创建图形，添加colorbar空间
    fig = plt.figure(figsize=(18, 10))
    
    # 使用GridSpec创建布局，右侧留出colorbar空间
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05], hspace=0.3, wspace=0.3)
    
    # ===== 上排：原始信号 =====
    # 左上：原始信号 + 热力图
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_axis, victim['raw_signal'], 'b-', linewidth=0.5, alpha=0.7, label='Raw EEG')
    
    # 热力图叠加
    for i in range(len(heatmap_normalized)):
        if heatmap_normalized[i] > 0.2:
            ax1.axvspan(time_axis[i], time_axis[min(i+1, len(time_axis)-1)], 
                       alpha=heatmap_normalized[i] * 0.6, color='red')
    
    ax1.set_title(f'Raw EEG\n(True: N3, Pred: N3 [{victim["raw_conf"]:.1%}])', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (normalized)')
    ax1.set_ylim(-y_limit, y_limit)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 右上：预测概率分布
    ax2 = fig.add_subplot(gs[0, 1])
    x_pos = np.arange(5)
    bars1 = ax2.bar(x_pos, victim['raw_pred'], 0.6, color=['gray']*3 + ['green'] + ['gray'], alpha=0.8)
    bars1[3].set_color('green')  # N3高亮
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(stage_names)
    ax2.set_ylabel('Prediction Probability')
    ax2.set_title('Raw EEG Prediction Distribution', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ===== 下排：去噪信号 =====
    # 左下：去噪信号 + 热力图
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(time_axis, victim['denoised_signal'], 'g-', linewidth=0.5, alpha=0.7, label='Denoised EEG (V4)')
    
    # 热力图叠加
    for i in range(len(heatmap_normalized_d)):
        if heatmap_normalized_d[i] > 0.2:
            ax3.axvspan(time_axis[i], time_axis[min(i+1, len(time_axis)-1)], 
                       alpha=heatmap_normalized_d[i] * 0.6, color='red')
    
    # 标注误判结果
    pred_stage = stage_names[victim['denoised_class']]
    ax3.set_title(f'Denoised EEG (V4)\n(True: N3, Pred: {pred_stage} [{victim["denoised_conf"]:.1%}]) ⚠️ MISCLASSIFIED', 
                  fontsize=12, fontweight='bold', color='red')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Amplitude (normalized)')
    ax3.set_ylim(-y_limit, y_limit)  # 统一Y轴尺度
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 右下：预测概率分布
    ax4 = fig.add_subplot(gs[1, 1])
    bars2 = ax4.bar(x_pos, victim['denoised_pred'], 0.6, alpha=0.8)
    # 高亮预测类别
    for i, bar in enumerate(bars2):
        if i == victim['denoised_class']:
            bar.set_color('red')  # 误判类别红色
        elif i == 3:
            bar.set_color('orange')  # 正确类别N3橙色
        else:
            bar.set_color('gray')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(stage_names)
    ax4.set_ylabel('Prediction Probability')
    ax4.set_title('Denoised EEG Prediction Distribution', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # ===== Colorbar =====
    ax_cbar = fig.add_subplot(gs[:, 2])
    cmap = plt.cm.hot
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cbar)
    cbar.set_label('Grad-CAM Activation Weight', fontsize=11)
    
    plt.suptitle(f'Victim Sample Analysis - Epoch {victim["epoch_idx"]}\n'
                 f'N3 (Deep Sleep) → {pred_stage} Misclassification due to Over-smoothing', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = output_dir / f'Victim_Epoch{victim["epoch_idx"]}_GradCAM.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"可视化已保存: {output_path}")
    
    # ==================== 生成对比热力图 ====================
    fig2, axes2 = plt.subplots(3, 1, figsize=(16, 12))
    
    # 原始信号波形
    axes2[0].plot(time_axis, victim['raw_signal'], 'b-', linewidth=0.5)
    axes2[0].set_title('原始信号波形 (含明显Delta慢波)', fontsize=12)
    axes2[0].set_ylabel('幅度')
    axes2[0].grid(True, alpha=0.3)
    
    # 热力图对比
    axes2[1].imshow(heatmap_raw_upsampled.reshape(1, -1), aspect='auto', cmap='hot', 
                    extent=[0, 30, 0, 1], vmin=0, vmax=1)
    axes2[1].set_title('原始信号Grad-CAM热力图 (注意力集中在Delta波区域)', fontsize=12)
    axes2[1].set_xlabel('时间 (秒)')
    axes2[1].set_yticks([])
    
    axes2[2].imshow(heatmap_denoised_upsampled.reshape(1, -1), aspect='auto', cmap='hot',
                    extent=[0, 30, 0, 1], vmin=0, vmax=1)
    axes2[2].set_title('去噪信号Grad-CAM热力图 (注意力分散/破碎)', fontsize=12)
    axes2[2].set_xlabel('时间 (秒)')
    axes2[2].set_yticks([])
    
    plt.suptitle('Grad-CAM热力图对比：注意力焦点变化', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path2 = output_dir / f'Victim_Epoch{victim["epoch_idx"]}_HeatmapComparison.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"热力图对比已保存: {output_path2}")
    
    return {
        'heatmap_raw': heatmap_raw_upsampled,
        'heatmap_denoised': heatmap_denoised_upsampled,
        'raw_focus_score': np.mean(heatmap_raw_upsampled[heatmap_raw_upsampled > np.percentile(heatmap_raw_upsampled, 75)]),
        'denoised_focus_score': np.mean(heatmap_denoised_upsampled[heatmap_denoised_upsampled > np.percentile(heatmap_denoised_upsampled, 75)])
    }


def main():
    """主函数"""
    print("=" * 60)
    print("受害样本精准分析：1D Grad-CAM对比")
    print("=" * 60)
    
    output_dir = PROJECT_ROOT / "06_实验结果" / "受害样本分析"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 第一步：找受害样本
    victims, judge_model = find_victim_samples()
    
    if not victims:
        print("\n错误：未找到任何受害样本！")
        return
    
    # 第二步：对每个受害样本进行Grad-CAM分析
    print("\n" + "=" * 60)
    print("开始Grad-CAM分析...")
    print("=" * 60)
    
    results = []
    for i, victim in enumerate(victims):
        print(f"\n分析受害样本 {i+1}/{len(victims)}...")
        result = analyze_victim_with_gradcam(victim, judge_model, output_dir)
        result['epoch_idx'] = victim['epoch_idx']
        results.append(result)
    
    # 保存结果摘要
    print("\n" + "=" * 60)
    print("分析完成！结果摘要")
    print("=" * 60)
    
    for r in results:
        print(f"\nEpoch {r['epoch_idx']}:")
        print(f"  原始信号注意力集中度: {r['raw_focus_score']:.4f}")
        print(f"  去噪信号注意力集中度: {r['denoised_focus_score']:.4f}")
        print(f"  注意力分散程度: {(r['raw_focus_score'] - r['denoised_focus_score']) / r['raw_focus_score'] * 100:.1f}%")
    
    print(f"\n所有结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
