import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import sys
from pathlib import Path

# ================= 配置 =================
CORE_CODE_DIR = Path(__file__).resolve().parents[2] / "02_核心代码"
if str(CORE_CODE_DIR) not in sys.path:
    sys.path.append(str(CORE_CODE_DIR))
from bootstrap_paths import PROJECT_ROOT
data_path = str(PROJECT_ROOT / "08_第三方工具" / "eeglab2025.1.0" / "sleep_dataset.npz")
BATCH_SIZE = 64      #稍微调大一点Batch Size
EPOCHS = 20          # 增加训练轮数
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用设备: {device}")

# ================= 1. 加载与预处理 =================
print("1. 正在加载数据...")
try:
    data = np.load(data_path)
    X = data['x_data'] 
    y = data['y_labels']
except Exception as e:
    print(f"错误: 无法加载数据，请检查路径。{e}")
    exit()

# Z-score 标准化
X_mean = np.mean(X)
X_std = np.std(X)
X = (X - X_mean) / (X_std + 1e-6) # 加个极小值防止除以0

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# 拆分数据
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor
)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(f"训练集: {len(X_train)} | 测试集: {len(X_test)}")

# ================= 2. 定义改进的 CNN 模型 =================
class BetterSleepCNN(nn.Module):
    def __init__(self):
        super(BetterSleepCNN, self).__init__()
        # 结构：Conv -> BatchNorm -> ReLU -> MaxPool
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=2, padding=32),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=32, stride=2, padding=16),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # 将时间维度压缩到固定长度 10，保留一定的时序信息
        self.adaptive_pool = nn.AdaptiveAvgPool1d(10) 
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 10, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # 防止过拟合
            nn.Linear(128, 5)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.adaptive_pool(x)
        x = self.fc(x)
        return x

model = BetterSleepCNN().to(device)

# ================= 3. 处理样本不平衡 =================
class_counts = torch.bincount(y_train)
total_samples = len(y_train)
# 计算权重，加 1e-6 防止除以 0
class_weights = total_samples / (len(class_counts) * (class_counts.float() + 1e-6))
class_weights = class_weights.to(device)

print(f"各类别样本数: {class_counts.cpu().numpy()}")
print(f"使用类别权重: {class_weights.cpu().numpy()}")

# 带权重的损失函数
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

# ================= 4. 训练循环 =================
print("\n3. 开始训练模型...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # 测试部分
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
    train_acc = 100 * correct / total
    test_acc = 100 * test_correct / test_total
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

# ================= 5. 最终详细评估 =================
print("\n================= 最终报告 =================")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds, digits=4))
print("混淆矩阵:\n", confusion_matrix(all_labels, all_preds))