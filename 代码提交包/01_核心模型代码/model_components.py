"""
模型核心组件定义
包含: SEBlock, Res_BasicBlock, BasicBlockall
"""
import tensorflow as tf
from tensorflow.keras import layers, Sequential


class SEBlock(layers.Layer):
    """
    Squeeze-and-Excitation注意力模块
    
    通过全局平均池化 + 全连接层学习通道权重，实现自适应特征重标定
    
    参数:
        channels: 输入通道数
        reduction: 压缩比例，用于减少参数量
    """
    def __init__(self, channels=32, reduction=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        self.fc1 = layers.Dense(max(channels // reduction, 4), activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')
        self.reshape = layers.Reshape((1, channels))
    
    def call(self, inputs):
        # Squeeze: 全局平均池化
        x = self.global_avg_pool(inputs)
        # Excitation: 两个全连接层学习通道权重
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.reshape(x)
        # Scale: 重新加权
        return inputs * x
    
    def get_config(self):
        config = super(SEBlock, self).get_config()
        config.update({
            "channels": self.channels, 
            "reduction": self.reduction
        })
        return config


class Res_BasicBlock(layers.Layer):
    """
    残差基础块
    
    结构: Conv1D(32) -> BN -> ReLU -> Conv1D(16) -> BN -> ReLU -> Conv1D(32) -> BN -> ReLU
    可选SE注意力
    残差连接: output + input
    
    参数:
        kernelsize: 卷积核大小
        stride: 步长
        use_se: 是否使用SE注意力
        se_reduction: SE模块压缩比例
    """
    def __init__(self, kernelsize, stride=1, use_se=False, se_reduction=16, **kwargs):
        super(Res_BasicBlock, self).__init__(**kwargs)
        self.kernelsize = kernelsize
        self.stride = stride
        self.use_se = use_se
        self.se_reduction = se_reduction
        
        self.bblock = Sequential([
            layers.Conv1D(32, kernelsize, strides=stride, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv1D(16, kernelsize, strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv1D(32, kernelsize, strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        
        if use_se:
            self.se = SEBlock(32, se_reduction)
    
    def call(self, inputs):
        out = self.bblock(inputs)
        if self.use_se:
            out = self.se(out)
        return layers.add([out, inputs])
    
    def get_config(self):
        config = super(Res_BasicBlock, self).get_config()
        config.update({
            "kernelsize": self.kernelsize, 
            "stride": self.stride, 
            "use_se": self.use_se, 
            "se_reduction": self.se_reduction
        })
        return config


class BasicBlockall(layers.Layer):
    """
    多尺度并行卷积块
    
    并行使用3种卷积核: kernel_size = 3, 5, 7
    捕获不同尺度的时序特征:
    - kernel_size=3: 捕捉高频肌电细节、尖峰电位
    - kernel_size=5: 捕捉睡眠纺锤波、Beta波波动
    - kernel_size=7: 捕捉Delta波、K-复合波轮廓、眼动伪迹
    
    输出拼接: [bblock3, bblock5, bblock7]
    
    参数:
        stride: 步长
        use_se: 是否使用SE注意力
        se_reduction: SE模块压缩比例
    """
    def __init__(self, stride=1, use_se=False, se_reduction=16, **kwargs):
        super(BasicBlockall, self).__init__(**kwargs)
        self.stride = stride
        self.use_se = use_se
        self.se_reduction = se_reduction
        
        # 三个并行的残差分支
        self.bblock3 = Sequential([
            Res_BasicBlock(3, use_se=use_se), 
            Res_BasicBlock(3, use_se=use_se)
        ])
        self.bblock5 = Sequential([
            Res_BasicBlock(5, use_se=use_se), 
            Res_BasicBlock(5, use_se=use_se)
        ])
        self.bblock7 = Sequential([
            Res_BasicBlock(7, use_se=use_se), 
            Res_BasicBlock(7, use_se=use_se)
        ])
    
    def call(self, inputs):
        out3 = self.bblock3(inputs)
        out5 = self.bblock5(inputs)
        out7 = self.bblock7(inputs)
        return tf.concat([out3, out5, out7], axis=-1)
    
    def get_config(self):
        config = super(BasicBlockall, self).get_config()
        config.update({
            "stride": self.stride, 
            "use_se": self.use_se, 
            "se_reduction": self.se_reduction
        })
        return config