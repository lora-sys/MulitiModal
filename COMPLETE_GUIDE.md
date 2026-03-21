# 多模态机器学习完整实践指南

## 目录
1. [项目概述](#项目概述)
2. [数据集制作](#数据集制作)
3. [模型架构设计](#模型架构设计)
4. [训练流程](#训练流程)
5. [评估指标](#评估指标)
6. [实验设计](#实验设计)
7. [实战案例](#实战案例)
8. [最佳实践](#最佳实践)

---

## 项目概述

### 项目背景
**目标**：基于多模态数据（动态波形 + 静态特征）预测按摩椅舒适度

**任务类型**：
- 分类任务（3类：一般、正常、良好）
- 回归任务（中医诊断分数30-100分）

**核心技术栈**：
- 深度学习框架：PyTorch
- 数据处理：NumPy, Pandas
- 可视化：Matplotlib
- 实验管理：自定义实验日志

### 核心挑战
1. **多模态融合**：如何有效融合时序数据和结构化数据
2. **数据质量**：动态波形是仿真数据，静态数据是真实数据
3. **泛化能力**：避免模型"偷懒"过度依赖仿真数据
4. **实时性**：低延迟推理需求（<100ms）

---

## 数据集制作

### 1. 数据源收集

#### 原始数据结构
```python
# 静态数据（真实用户）
static_data = {
    '年龄': [25, 30, 35, ...],           # 数值型
    'BMI': [22.5, 24.0, 26.3, ...],      # 数值型
    '心率': [72, 68, 75, ...],            # 数值型
    '血氧': [98, 97, 99, ...],            # 数值型
    '舌面诊评分1': [8, 7, 9, ...],        # 数值型
    '舌面诊评分2': [6, 8, 7, ...],        # 数值型
    '体质': ['平和质', '气虚质', ...],    # 分类（39种）
    '中医诊断分数': [75, 80, 70, ...]     # 标签（30-100分）
}

# 动态波形（仿真生成）
waveforms = {
    '压力波形': [array_1, array_2, ...],  # 2通道 × 1000点
    '噪声类型': ['gaussian', 'drift', ...] # 9种噪声类型
}
```

#### 数据来源
- **静态数据**：9406个真实用户健康档案
- **压力波形库**：10,000条仿真波形（含9种噪声类型）
- **标签**：中医诊断分数（30-100分，15个离散值）

### 2. 数据预处理

#### 预处理原则
**核心原则**：先做场景专属的极致干净预处理，再做贴合真实量产场景的可控噪声注入

#### 完整预处理流程

**步骤1：硬过滤 - 剔除无效样本**
```python
def hard_filter(raw_signal, fs=50):
    """
    使用滑动窗口3σ法检测异常值
    """
    s = pd.Series(raw_signal)
    rolling_mean = s.rolling(window=15, center=True).mean()
    rolling_std = s.rolling(window=15, center=True).std()
    rolling_std = rolling_std.bfill().ffill()
    
    # 检测异常值
    is_anomaly = (s > rolling_mean + 3 * rolling_std) | \
                 (s < rolling_mean - 3 * rolling_std)
    
    # 用相邻有效值插值替换
    s_clean = s.copy()
    s_clean[is_anomaly] = np.nan
    s_clean = s_clean.interpolate(method="cubic")
    s_clean = s_clean.bfill().ffill()
    
    return s_clean.values
```

**步骤2：NaN处理**
```python
def handle_nan_values(signal):
    """使用线性插值修复丢失的采样点"""
    s = pd.Series(signal)
    s_clean = s.interpolate(method='linear')
    s_clean = s_clean.bfill().ffill()
    return s_clean.values
```

**步骤3：场景化软降噪**
```python
def scene_denoise(signal, fs=50, highcut=10):
    """
    - 低通滤波去除高频机械振动
    - 基线校正去除零点漂移
    """
    # 低通巴特沃斯滤波
    s_filtered = nk.signal_filter(
        signal,
        sampling_rate=fs,
        highcut=highcut,
        method="butterworth",
        order=4,
    )
    
    # 基线校正
    baseline = np.median(s_filtered)
    s_corrected = s_filtered - baseline
    
    return s_corrected
```

**步骤4：个体归一化**
```python
def individual_normalize(signal):
    """转换为相对基线的变化率"""
    s_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
    return s_norm.astype(np.float32)
```

**完整流程**
```python
def clean_preprocess(signal, fs=50):
    signal = handle_nan_values(signal)
    signal = hard_filter(signal, fs)
    signal = scene_denoise(signal, fs, highcut=10)
    signal = individual_normalize(signal)
    return signal
```

### 3. 数据集划分

#### 划分策略
```python
from sklearn.model_selection import train_test_split

# 方法1：按比例划分（7:2:1）
train_val, test = train_test_split(
    data, test_size=0.1, random_state=42, stratify=labels
)
train, val = train_test_split(
    train_val, test_size=0.22, random_state=42, stratify=labels
)

# 方法2：交叉验证（Stratified K-Fold）
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, y):
    # 训练和验证
    pass
```

#### 避免数据泄露
```python
# ✅ 正确：使用训练集的统计量标准化
train_mean = train_data.mean()
train_std = train_data.std()
train_normalized = (train_data - train_mean) / train_std
test_normalized = (test_data - train_mean) / train_std

# ❌ 错误：使用全部数据的统计量
all_mean = all_data.mean()
all_std = all_data.std()
train_normalized = (train_data - all_mean) / all_std  # 数据泄露！
```

### 4. 数据集保存

#### NPZ格式保存
```python
import numpy as np

np.savez(
    'unified_dataset.npz',
    dynamic=dynamic_data,          # (N, 2, 1000)
    static_basic=static_basic,     # (N, 4)
    static_scores=static_scores,   # (N, 2)
    constitution=constitution,     # (N,)
    label=labels                   # (N,)
)

# 加载数据
data = np.load('unified_dataset.npz')
dynamic = data['dynamic']
label = data['label']
```

### 5. 数据增强（可选）

#### 噪声注入
```python
def inject_controlled_noise(signal, noise_type="baseline_shift"):
    """
    训练时动态注入噪声，提升鲁棒性
    """
    if noise_type == "baseline_shift":
        # 基线偏移 ±5%
        shift = np.random.uniform(-0.05, 0.05) * (signal.max() - signal.min())
        noisy_signal = signal + shift
    
    elif noise_type == "amplitude_scaling":
        # 幅度缩放 ±10-15%
        scale = np.random.uniform(0.9, 1.1)
        noisy_signal = signal * scale
    
    elif noise_type == "gaussian_noise":
        # 高斯噪声 SNR=30-40dB
        noise_level = 0.01 * np.std(signal)
        noise = np.random.normal(0, noise_level, len(signal))
        noisy_signal = signal + noise
    
    return noisy_signal
```

#### 训练时动态增强
```python
for epoch in range(num_epochs):
    for batch in train_loader:
        # 50%概率注入噪声
        if np.random.random() < 0.5:
            noise_type = np.random.choice([
                "baseline_shift", "amplitude_scaling", "gaussian_noise"
            ])
            dynamic = batch['dynamic']
            for i in range(len(dynamic)):
                for channel in range(2):
                    dynamic[i, channel] = inject_controlled_noise(
                        dynamic[i, channel], noise_type
                    )
        # 正常训练
        pass
```

---

## 模型架构设计

### 1. 模型设计原则

#### 设计考量
1. **多模态融合策略**：如何融合时序数据和结构化数据
2. **计算效率**：实时性要求（<100ms延迟）
3. **可扩展性**：支持后期添加新模态
4. **可解释性**：理解各模态的贡献度

### 2. 基础架构组件

#### 时序编码器（1D CNN + LSTM）
```python
class TemporalEncoder(nn.Module):
    """处理动态波形数据"""
    def __init__(self, input_channels=2, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 256)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # x: (batch, channels, seq_len)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # LSTM
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
        lstm_out, _ = self.lstm(x)
        
        # 全局平均池化
        x = lstm_out.mean(dim=1)
        x = self.fc(x)
        x = self.dropout(x)
        return x
```

#### 静态编码器（MLP）
```python
class StaticEncoder(nn.Module):
    """处理静态特征"""
    def __init__(self, input_dim=4, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
```

### 3. 融合策略

#### 策略1：简单拼接（Simple Concat）
```python
class SimpleConcatModel(nn.Module):
    """最简单的融合方式：直接拼接特征"""
    def __init__(self, num_classes=3):
        super().__init__()
        self.temporal_encoder = TemporalEncoder()
        self.static_encoder = StaticEncoder()
        
        # 直接拼接
        total_dim = 256 + 128  # temporal + static
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, dynamic, static):
        temporal_feat = self.temporal_encoder(dynamic)
        static_feat = self.static_encoder(static)
        
        # 拼接
        combined = torch.cat([temporal_feat, static_feat], dim=1)
        output = self.classifier(combined)
        return output
```

#### 策略2：晚融合Transformer（Late Fusion）
```python
class LateFusionModel(nn.Module):
    """独立编码后用Transformer融合"""
    def __init__(self, num_classes=3, num_heads=4):
        super().__init__()
        self.temporal_encoder = TemporalEncoder()
        self.static_encoder = StaticEncoder()
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, dynamic, static):
        temporal_feat = self.temporal_encoder(dynamic)  # (batch, 256)
        static_feat = self.static_encoder(static)      # (batch, 128)
        
        # 拼接并扩展维度
        combined = torch.cat([temporal_feat, static_feat], dim=1)  # (batch, 384)
        combined = combined.unsqueeze(0)  # (1, batch, 384)
        
        # Transformer融合
        fused = self.transformer(combined)
        fused = fused.squeeze(0)  # (batch, 384)
        
        output = self.classifier(fused[:, :256])
        return output
```

#### 策略3：交叉注意力（Cross-Attention）
```python
class CrossAttentionModel(nn.Module):
    """使用交叉注意力机制融合多模态"""
    def __init__(self, num_classes=3, num_heads=4):
        super().__init__()
        self.temporal_encoder = TemporalEncoder()
        self.static_encoder = StaticEncoder()
        
        # 多头交叉注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=num_heads,
            dropout=0.1
        )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, dynamic, static):
        temporal_feat = self.temporal_encoder(dynamic)  # (batch, 256)
        static_feat = self.static_encoder(static)      # (batch, 128)
        
        # 将静态特征投影到256维
        static_proj = F.linear(static_feat, torch.randn(256, 128))
        
        # 交叉注意力：用静态特征查询时序特征
        attn_output, _ = self.cross_attention(
            query=static_proj.unsqueeze(0),
            key=temporal_feat.unsqueeze(0),
            value=temporal_feat.unsqueeze(0)
        )
        attn_output = attn_output.squeeze(0)
        
        # 门控融合
        gate_weight = self.gate(torch.cat([temporal_feat, static_feat], dim=1))
        fused = gate_weight * attn_output + (1 - gate_weight) * temporal_feat
        
        output = self.classifier(fused)
        return output
```

### 4. 回归模型

#### 修改输出层
```python
# 将分类任务的输出层改为回归
class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # ... 编码器部分 ...
        
        # 分类输出：num_classes
        # 回归输出：1
        self.regressor = nn.Linear(256, 1)
    
    def forward(self, dynamic, static):
        fused = self.encode(dynamic, static)
        output = self.regressor(fused)
        return output.squeeze()
```

---

## 训练流程

### 1. 数据加载

#### 自定义Dataset
```python
class MultiModalDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        self.dynamic = torch.from_numpy(data['dynamic']).float()
        self.static_basic = torch.from_numpy(data['static_basic']).float()
        self.static_scores = torch.from_numpy(data['static_scores']).float()
        self.constitution = torch.from_numpy(data['constitution']).long()
        self.label = torch.from_numpy(data['label']).float()
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return {
            'dynamic': self.dynamic[idx],
            'static_basic': self.static_basic[idx],
            'static_scores': self.static_scores[idx],
            'constitution': self.constitution[idx],
            'label': self.label[idx]
        }
```

#### DataLoader
```python
train_dataset = MultiModalDataset('train_data.npz')
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

### 2. 训练配置

#### 超参数选择
```python
config = {
    # 数据配置
    'batch_size': 32,
    'num_workers': 4,
    
    # 优化器配置
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'optimizer': 'AdamW',
    
    # 训练配置
    'num_epochs': 50,
    'warmup_epochs': 5,
    'early_stopping_patience': 5,
    
    # 学习率调度
    'scheduler': 'CosineAnnealingWarmupRestarts',
    'min_lr': 1e-6,
    'T_0': 10,
    
    # 模型配置
    'hidden_dim': 128,
    'num_heads': 4,
    'dropout': 0.2,
    
    # 其他
    'random_seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
```

### 3. 训练循环

#### 标准训练循环
```python
def train(model, train_loader, val_loader, config):
    # 设置随机种子
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmupRestarts(
        optimizer,
        T_0=config['T_0'],
        T_mult=1,
        eta_min=config['min_lr']
    )
    
    # 损失函数
    criterion = nn.MSELoss()  # 回归任务
    # criterion = nn.CrossEntropyLoss()  # 分类任务
    
    # 早停
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    for epoch in range(config['num_epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(
                batch['dynamic'].to(config['device']),
                batch['static_basic'].to(config['device']),
                batch['static_scores'].to(config['device']),
                batch['constitution'].to(config['device'])
            )
            
            # 计算损失
            loss = criterion(outputs, batch['label'].to(config['device']))
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证阶段
        val_loss = validate(model, val_loader, criterion, config)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # 打印进度
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")
    
    return history
```

#### 验证函数
```python
def validate(model, val_loader, criterion, config):
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                batch['dynamic'].to(config['device']),
                batch['static_basic'].to(config['device']),
                batch['static_scores'].to(config['device']),
                batch['constitution'].to(config['device'])
            )
            
            loss = criterion(outputs, batch['label'].to(config['device']))
            val_loss += loss.item()
    
    return val_loss / len(val_loader)
```

### 4. 模型保存与加载

#### 保存模型
```python
# 保存整个模型
torch.save(model, 'model.pth')

# 只保存状态字典（推荐）
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_loss': best_val_loss,
    'config': config
}, 'checkpoint.pth')
```

#### 加载模型
```python
# 加载状态字典
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
epoch = checkpoint['epoch']
best_val_loss = checkpoint['best_val_loss']
```

---

## 评估指标

### 1. 分类任务指标

#### 准确率（Accuracy）
```python
def accuracy_score(y_true, y_pred):
    """准确率"""
    return np.mean(y_true == y_pred)
```

#### 精确率（Precision）
```python
def precision_score(y_true, y_pred, average='macro'):
    """精确率"""
    from sklearn.metrics import precision_score as ps
    return ps(y_true, y_pred, average=average)
```

#### 召回率（Recall）
```python
def recall_score(y_true, y_pred, average='macro'):
    """召回率"""
    from sklearn.metrics import recall_score as rs
    return rs(y_true, y_pred, average=average)
```

#### F1分数（F1-Score）
```python
def f1_score(y_true, y_pred, average='macro'):
    """F1分数"""
    from sklearn.metrics import f1_score as f1
    return f1(y_true, y_pred, average=average)
```

#### 混淆矩阵
```python
def confusion_matrix(y_true, y_pred):
    """混淆矩阵"""
    from sklearn.metrics import confusion_matrix as cm
    return cm(y_true, y_pred)
```

### 2. 回归任务指标

#### 平均绝对误差（MAE）
```python
def mean_absolute_error(y_true, y_pred):
    """平均绝对误差"""
    return np.mean(np.abs(y_true - y_pred))
```

#### 均方根误差（RMSE）
```python
def root_mean_squared_error(y_true, y_pred):
    """均方根误差"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
```

#### 决定系数（R²）
```python
def r2_score(y_true, y_pred):
    """决定系数"""
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)
```

#### Pearson相关系数
```python
def pearson_correlation(y_true, y_pred):
    """Pearson相关系数"""
    return np.corrcoef(y_true, y_pred)[0, 1]
```

### 3. 综合评估函数

```python
def evaluate_model(model, data_loader, config, task='classification'):
    """综合评估模型"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(
                batch['dynamic'].to(config['device']),
                batch['static_basic'].to(config['device']),
                batch['static_scores'].to(config['device']),
                batch['constitution'].to(config['device'])
            )
            
            if task == 'classification':
                preds = outputs.argmax(dim=1).cpu().numpy()
            else:
                preds = outputs.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(batch['label'].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算指标
    if task == 'classification':
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds),
            'recall': recall_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds),
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }
    else:
        metrics = {
            'mae': mean_absolute_error(all_labels, all_preds),
            'rmse': root_mean_squared_error(all_labels, all_preds),
            'r2': r2_score(all_labels, all_preds),
            'pearson': pearson_correlation(all_labels, all_preds)
        }
    
    return metrics
```

---

## 实验设计

### 1. 对比实验

#### 目的
比较不同模型架构的性能，选择最优模型

#### 设计方法
```python
# 定义模型列表
models = {
    'Simple Concat': SimpleConcatModel,
    'Late Fusion': LateFusionModel,
    'Cross-Attention': CrossAttentionModel
}

# 训练所有模型
results = {}
for name, model_class in models.items():
    print(f"\nTraining {name}...")
    
    model = model_class(num_classes=3).to(device)
    history = train(model, train_loader, val_loader, config)
    
    # 评估
    test_metrics = evaluate_model(model, test_loader, config)
    results[name] = {
        'history': history,
        'test_metrics': test_metrics
    }

# 对比结果
print("\n=== Model Comparison ===")
for name, result in results.items():
    metrics = result['test_metrics']
    print(f"{name}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
```

### 2. 消融实验

#### 目的
分析各模态对模型性能的贡献度

#### 设计方法
```python
def ablation_study(model, test_loader, config, modalities):
    """消融实验"""
    baseline_metrics = evaluate_model(model, test_loader, config)
    
    results = {}
    results['baseline'] = baseline_metrics
    
    # 测试去掉每个模态
    for modality in modalities:
        print(f"\nTesting without {modality}...")
        
        # 修改数据加载器，去掉对应模态
        modified_loader = remove_modality(test_loader, modality)
        
        metrics = evaluate_model(model, modified_loader, config)
        results[f'no_{modality}'] = metrics
        
        # 计算性能下降
        if 'accuracy' in baseline_metrics:
            performance_drop = baseline_metrics['accuracy'] - metrics['accuracy']
            print(f"  Performance drop: {performance_drop:.4f}")
    
    return results

# 执行消融实验
modalities = ['dynamic', 'static_basic', 'static_scores', 'constitution']
ablation_results = ablation_study(model, test_loader, config, modalities)
```

### 3. 5折交叉验证

#### 目的
验证模型在不同数据划分下的稳定性

#### 设计方法
```python
from sklearn.model_selection import StratifiedKFold

def cross_validation(model_class, dataset, config, k=5):
    """5折交叉验证"""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    all_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, dataset.labels)):
        print(f"\n=== Fold {fold + 1}/{k} ===")
        
        # 划分数据
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # 训练模型
        model = model_class(num_classes=3).to(config['device'])
        history = train(model, train_loader, val_loader, config)
        
        # 评估
        val_metrics = evaluate_model(model, val_loader, config)
        
        all_results.append({
            'fold': fold + 1,
            'history': history,
            'metrics': val_metrics
        })
    
    # 统计结果
    accuracies = [r['metrics']['accuracy'] for r in all_results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print(f"\n=== Cross-Validation Results ===")
    print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    
    return all_results
```

### 4. 实时测试

#### 目的
验证模型在实时场景下的表现

#### 设计方法
```python
def realtime_test(model, test_data, config, window_size=1000, step_size=100):
    """实时测试（滑动窗口）"""
    model.eval()
    
    predictions = []
    targets = []
    latencies = []
    
    with torch.no_grad():
        for i in range(0, len(test_data) - window_size, step_size):
            start_time = time.time()
            
            # 提取窗口数据
            window = test_data[i:i+window_size]
            
            # 预处理
            processed_window = preprocess_window(window)
            
            # 预测
            output = model(processed_window.to(config['device']))
            
            # 记录延迟
            latency = time.time() - start_time
            latencies.append(latency)
            
            # 记录结果
            predictions.append(output.cpu().numpy())
            targets.append(test_data['label'][i + window_size])
    
    # 计算指标
    metrics = {
        'predictions': np.array(predictions),
        'targets': np.array(targets),
        'mean_latency': np.mean(latencies),
        'max_latency': np.max(latencies)
    }
    
    # 计算准确率
    if 'accuracy' in metrics:
        metrics['accuracy'] = accuracy_score(
            metrics['targets'], 
            metrics['predictions']
        )
    
    return metrics
```

### 5. 鲁棒性测试

#### 目的
测试模型对噪声的抗干扰能力

#### 设计方法
```python
def robustness_test(model, test_loader, config, noise_types):
    """鲁棒性测试"""
    baseline_metrics = evaluate_model(model, test_loader, config)
    
    results = {'baseline': baseline_metrics}
    
    for noise_type in noise_types:
        print(f"\nTesting with {noise_type} noise...")
        
        # 添加噪声
        noisy_loader = add_noise(test_loader, noise_type)
        
        # 评估
        metrics = evaluate_model(model, noisy_loader, config)
        results[noise_type] = metrics
        
        # 计算性能下降
        if 'accuracy' in baseline_metrics:
            performance_drop = baseline_metrics['accuracy'] - metrics['accuracy']
            print(f"  Performance drop: {performance_drop:.4f}")
    
    return results

# 噪声类型
noise_types = ['gaussian', 'drift', 'dropout', 'baseline_shift', 'amplitude_scaling']
robustness_results = robustness_test(model, test_loader, config, noise_types)
```

---

## 实战案例

### 案例1：分类任务完整流程

#### 数据准备
```python
# 1. 加载数据
dataset = MultiModalDataset('unified_dataset.npz')

# 2. 划分数据集
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)

# 3. 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

#### 模型训练
```python
# 1. 定义模型
model = CrossAttentionModel(num_classes=3).to('cuda')

# 2. 定义配置
config = {
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'num_epochs': 50,
    'warmup_epochs': 5,
    'early_stopping_patience': 5,
    'device': 'cuda'
}

# 3. 训练
history = train(model, train_loader, val_loader, config)
```

#### 模型评估
```python
# 1. 测试集评估
test_metrics = evaluate_model(model, test_loader, config, task='classification')

print("=== Test Set Results ===")
print(f"Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Precision: {test_metrics['precision']:.4f}")
print(f"Recall: {test_metrics['recall']:.4f}")
print(f"F1 Score: {test_metrics['f1']:.4f}")

# 2. 混淆矩阵
import matplotlib.pyplot as plt
import seaborn as sns

cm = test_metrics['confusion_matrix']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
```

#### 实验验证
```python
# 1. 对比实验
models = {
    'Simple Concat': SimpleConcatModel,
    'Late Fusion': LateFusionModel,
    'Cross-Attention': CrossAttentionModel
}

comparison_results = {}
for name, model_class in models.items():
    model = model_class(num_classes=3).to('cuda')
    train(model, train_loader, val_loader, config)
    metrics = evaluate_model(model, test_loader, config)
    comparison_results[name] = metrics

# 2. 消融实验
ablation_results = ablation_study(model, test_loader, config, 
                                   ['dynamic', 'static_basic', 'static_scores'])

# 3. 5折交叉验证
cv_results = cross_validation(CrossAttentionModel, dataset, config, k=5)
```

### 案例2：回归任务完整流程

#### 修改输出层
```python
class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_encoder = TemporalEncoder()
        self.static_encoder = StaticEncoder()
        self.cross_attention = nn.MultiheadAttention(embed_dim=256, num_heads=4)
        self.regressor = nn.Linear(256, 1)
    
    def forward(self, dynamic, static_basic, static_scores, constitution):
        # 编码
        temporal_feat = self.temporal_encoder(dynamic)
        static_feat = self.static_encoder(static_basic)
        
        # 融合
        static_proj = F.linear(static_feat, torch.randn(256, 128))
        attn_output, _ = self.cross_attention(
            query=static_proj.unsqueeze(0),
            key=temporal_feat.unsqueeze(0),
            value=temporal_feat.unsqueeze(0)
        )
        fused = attn_output.squeeze(0)
        
        # 回归
        output = self.regressor(fused)
        return output.squeeze()
```

#### 修改损失函数
```python
# 分类任务使用CrossEntropyLoss
criterion = nn.CrossEntropyLoss()

# 回归任务使用MSE
criterion = nn.MSELoss()
```

#### 评估指标
```python
# 回归任务评估
test_metrics = evaluate_model(model, test_loader, config, task='regression')

print("=== Test Set Results ===")
print(f"MAE: {test_metrics['mae']:.4f}")
print(f"RMSE: {test_metrics['rmse']:.4f}")
print(f"R²: {test_metrics['r2']:.4f}")
print(f"Pearson: {test_metrics['pearson']:.4f}")
```

---

## 最佳实践

### 1. 数据处理

#### ✅ 推荐做法
- 先做极致干净的预处理，再做可控噪声注入
- 使用训练集的统计量标准化，避免数据泄露
- 采用分层采样保持类别平衡
- 使用数据增强提升鲁棒性

#### ❌ 避免错误
- 不能先加噪再预处理（等于白加噪声）
- 不能把故障异常当噪声注入
- 不能在验证/测试集上加噪
- 不能过度加噪改变样本标签

### 2. 模型设计

#### ✅ 推荐做法
- 从简单模型开始，逐步增加复杂度
- 使用交叉注意力机制融合多模态
- 添加dropout防止过拟合
- 使用预训练模型加速收敛

#### ❌ 避免错误
- 不要一开始就用复杂模型
- 不要忽略模型可解释性
- 不要过度依赖单一模态

### 3. 训练策略

#### ✅ 推荐做法
- 使用学习率预热（Warmup）
- 采用余弦退火学习率调度
- 设置合理的早停策略
- 使用梯度裁剪防止梯度爆炸

#### ❌ 避免错误
- 不要使用固定的学习率
- 不要忽略验证集的监控
- 不要训练太久导致过拟合

### 4. 实验设计

#### ✅ 推荐做法
- 进行对比实验选择最优模型
- 执行消融实验分析模态贡献
- 使用交叉验证验证稳定性
- 测试模型鲁棒性

#### ❌ 避免错误
- 不要只看单一指标
- 不要忽略模型泛化能力
- 不要在训练集上评估性能

### 5. 评估与部署

#### ✅ 推荐做法
- 使用多个指标综合评估
- 在独立测试集上验证
- 监控推理延迟和资源消耗
- 记录详细的实验日志

#### ❌ 避免错误
- 不要只报告最好的结果
- 不要在训练数据上评估
- 不要忽略模型限制

---

## 总结

### 关键要点

1. **数据质量最重要**
   - 极致干净的预处理是基础
   - 可控的噪声注入提升鲁棒性
   - 严格避免数据泄露

2. **模型从简单到复杂**
   - 先用Simple Concat建立基线
   - 再用Transformer提升性能
   - 最后用Cross-Attention优化融合

3. **实验验证必不可少**
   - 对比实验选择最优模型
   - 消融实验理解模态贡献
   - 交叉验证验证稳定性
   - 鲁棒性测试确保可靠性

4. **持续迭代优化**
   - 基于实验结果调整策略
   - 记录详细的实验日志
   - 保持代码可复现性

### 技能树

#### 基础技能
- [ ] 数据预处理（清洗、标准化、增强）
- [ ] 模型架构设计（编码器、融合策略）
- [ ] 训练流程（优化器、学习率调度、早停）
- [ ] 评估指标（分类、回归）

#### 进阶技能
- [ ] 实验设计（对比、消融、交叉验证）
- [ ] 可视化分析（学习曲线、混淆矩阵）
- [ ] 鲁棒性测试（噪声、异常检测）
- [ ] 模型部署（延迟优化、资源管理）

#### 高级技能
- [ ] 迁移学习（预训练模型、微调）
- [ ] 自动化机器学习（超参数优化、架构搜索）
- [ ] 分布式训练（多GPU、混合精度）
- [ ] 持续学习（在线学习、灾难性遗忘）

### 延伸学习

#### 推荐资源
- **书籍**：《深度学习》、《动手学深度学习》
- **论文**：多模态学习相关综述
- **课程**：Coursera深度学习专项课程
- **实践**：Kaggle竞赛、GitHub开源项目

#### 下一步方向
1. 探索更复杂的融合策略（图神经网络、注意力机制）
2. 研究自监督学习在多模态中的应用
3. 学习模型压缩和量化技术
4. 尝试联邦学习和隐私保护

---

## 附录

### A. 完整代码示例

详见项目目录：`/home/lora/repos/MulitiModal/experiment/`

### B. 实验日志

详见：`/home/lora/repos/MulitiModal/experiment/EXPERIMENT_LOG.md`

### C. 配置文件

详见：`/home/lora/repos/MulitiModal/experiment/model/config_*.py`

---

**文档版本**：1.0  
**最后更新**：2026-03-21  
**作者**：多模态机器学习团队