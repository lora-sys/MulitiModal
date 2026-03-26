"""
改进的训练脚本示例

展示如何使用统一的日志系统和配置管理系统
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiment.utils.logger import get_logger, setup_logging
from experiment.config.base_config import ExperimentConfig, load_config_from_yaml


def main():
    """主训练函数"""

    # ==================== 1. 初始化日志系统 ====================
    setup_logging(
        log_dir='experiment/logs',
        level='INFO',
        console=True,
        file=True
    )

    # 获取日志记录器
    logger = get_logger(__name__)
    logger.info("=" * 60)
    logger.info("开始训练 - 使用改进的日志和配置系统")
    logger.info("=" * 60)

    # ==================== 2. 加载配置 ====================
    try:
        # 方式1：使用默认配置
        config = ExperimentConfig()

        # 方式2：从YAML文件加载配置（如果存在）
        config_path = 'experiment/config/experiment_config.yaml'
        if os.path.exists(config_path):
            logger.info(f"从YAML文件加载配置: {config_path}")
            config = load_config_from_yaml(config_path)

        logger.info(f"模型类型: {config.model.model_type}")
        logger.info(f"任务类型: {config.model.task_type}")
        logger.info(f"批次大小: {config.data.batch_size}")
        logger.info(f"学习率: {config.training.learning_rate}")
        logger.info(f"训练轮数: {config.training.num_epochs}")

    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        return

    # ==================== 3. 设置设备 ====================
    try:
        device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {device}")
    except Exception as e:
        logger.error(f"设置设备失败: {e}")
        device = torch.device('cpu')

    # ==================== 4. 加载数据 ====================
    try:
        from experiment.dataset.unified_source import UnifiedNPZDataSource
        from experiment.dataset.unified_dataset import UnifiedMultimodalDataset

        # 根据任务类型选择数据集
        if config.model.task_type == 'classification':
            dataset_path = config.data.classification_dataset
        else:
            dataset_path = config.data.regression_dataset

        logger.info(f"加载数据集: {dataset_path}")

        source = UnifiedNPZDataSource(dataset_path)
        if not source.initialize():
            logger.error("数据源初始化失败")
            return

        dataset = UnifiedMultimodalDataset(source)

        # 划分数据集
        train_size = int(config.data.train_ratio * len(dataset))
        val_size = int(config.data.val_ratio * len(dataset))
        test_size = len(dataset) - train_size - val_size

        logger.info(f"数据集划分 - 训练集: {train_size}, 验证集: {val_size}, 测试集: {test_size}")

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers
        )

    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # ==================== 5. 创建模型 ====================
    try:
        from experiment.model.model import get_model

        logger.info(f"创建模型: {config.model.model_type}")

        model = get_model(
            model_type=config.model.model_type,
            num_classes=config.model.num_classes,
            num_constitutions=config.data.num_constitutions,
            shared_dim=config.model.shared_dim,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout
        ).to(device)

        logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

    except Exception as e:
        logger.error(f"创建模型失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # ==================== 6. 定义优化器和损失函数 ====================
    try:
        # 优化器
        if config.training.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay
            )
        elif config.training.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay
            )
        elif config.training.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.training.learning_rate,
                momentum=0.9,
                weight_decay=config.training.weight_decay
            )
        else:
            logger.warning(f"未知的优化器: {config.training.optimizer}, 使用默认Adam")
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay
            )

        # 损失函数
        if config.model.task_type == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        logger.info(f"优化器: {config.training.optimizer}")
        logger.info(f"损失函数: {type(criterion).__name__}")

    except Exception as e:
        logger.error(f"定义优化器和损失函数失败: {e}")
        return

    # ==================== 7. 训练循环 ====================
    try:
        logger.info("开始训练...")

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(config.training.num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                # 获取数据
                dynamic = batch['dynamic'].to(device)
                static_basic = batch['static_basic'].to(device)
                static_scores = batch['static_scores'].to(device)
                constitution = batch['constitution'].to(device)

                if config.model.task_type == 'classification':
                    labels = batch['label'].to(device)

                    # 前向传播
                    outputs = model(dynamic, static_basic, static_scores, constitution)
                    loss = criterion(outputs, labels)

                    # 计算准确率
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()
                else:
                    scores = batch['scores'].to(device)

                    # 前向传播
                    outputs = model(dynamic, static_basic, static_scores, constitution)
                    loss = criterion(outputs.squeeze(), scores)

                # 反向传播
                loss.backward()

                # 梯度裁剪
                if config.training.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)

                optimizer.step()

                train_loss += loss.item()

                # 每100个batch打印一次
                if (batch_idx + 1) % 100 == 0:
                    logger.debug(f"Epoch [{epoch+1}/{config.training.num_epochs}], "
                                f"Batch [{batch_idx+1}/{len(train_loader)}], "
                                f"Loss: {loss.item():.4f}")

            # 计算训练指标
            avg_train_loss = train_loss / len(train_loader)
            if config.model.task_type == 'classification':
                train_acc = 100. * train_correct / train_total
                logger.info(f"Epoch [{epoch+1}/{config.training.num_epochs}] "
                           f"Train Loss: {avg_train_loss:.4f}, "
                           f"Train Acc: {train_acc:.2f}%")
            else:
                logger.info(f"Epoch [{epoch+1}/{config.training.num_epochs}] "
                           f"Train Loss: {avg_train_loss:.4f}")

            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    dynamic = batch['dynamic'].to(device)
                    static_basic = batch['static_basic'].to(device)
                    static_scores = batch['static_scores'].to(device)
                    constitution = batch['constitution'].to(device)

                    if config.model.task_type == 'classification':
                        labels = batch['label'].to(device)
                        outputs = model(dynamic, static_basic, static_scores, constitution)
                        loss = criterion(outputs, labels)

                        _, predicted = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()
                    else:
                        scores = batch['scores'].to(device)
                        outputs = model(dynamic, static_basic, static_scores, constitution)
                        loss = criterion(outputs.squeeze(), scores)

                    val_loss += loss.item()

            # 计算验证指标
            avg_val_loss = val_loss / len(val_loader)
            if config.model.task_type == 'classification':
                val_acc = 100. * val_correct / val_total
                logger.info(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            else:
                logger.info(f"Val Loss: {avg_val_loss:.4f}")

            # 早停检查
            if config.training.early_stopping:
                if avg_val_loss < best_val_loss - config.training.min_delta:
                    best_val_loss = avg_val_loss
                    patience_counter = 0

                    # 保存最佳模型
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': avg_val_loss,
                    }, 'experiment/checkpoints/best_model.pth')
                    logger.info(f"保存最佳模型 (Val Loss: {avg_val_loss:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= config.training.patience:
                        logger.info(f"早停触发: {config.training.patience} 个epoch没有改进")
                        break

    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # ==================== 8. 训练完成 ====================
    logger.info("=" * 60)
    logger.info("训练完成!")
    logger.info(f"最佳验证损失: {best_val_loss:.4f}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()