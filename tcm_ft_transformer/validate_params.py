"""
参数验证工具
防止模型参数和训练配置混淆
"""

# 模型参数白名单
MODEL_PARAMS = {
    'n_features',  # 输入特征维度
    'n_classes',   # 输出类别数
    'd_token',     # Token 嵌入维度
    'n_heads',     # 注意力头数
    'n_layers',    # Transformer 层数
    'dropout',     # Dropout 比例
}

# 训练配置白名单
TRAIN_PARAMS = {
    'batch_size',          # 批次大小
    'num_epochs',          # 训练轮数
    'learning_rate',       # 学习率
    'weight_decay',        # 权重衰减
    'warmup_ratio',        # Warmup 比例
    'grad_clip_max_norm',  # 梯度裁剪阈值
    'patience',            # 早停耐心值
    'device',              # 设备
    'checkpoint_dir',      # 检查点目录
}


def validate_model_params(params):
    """
    验证模型参数
    
    Args:
        params: 参数字典
        
    Returns:
        is_valid: 是否有效
        invalid_params: 无效参数列表
    """
    invalid_params = [k for k in params.keys() if k not in MODEL_PARAMS]
    
    if invalid_params:
        print(f"❌ 发现无效的模型参数: {invalid_params}")
        print(f"   模型参数只接受: {MODEL_PARAMS}")
        return False
    
    return True


def validate_train_params(params):
    """
    验证训练配置
    
    Args:
        params: 参数字典
        
    Returns:
        is_valid: 是否有效
        invalid_params: 无效参数列表
    """
    invalid_params = [k for k in params.keys() if k not in TRAIN_PARAMS]
    
    if invalid_params:
        print(f"❌ 发现无效的训练配置: {invalid_params}")
        print(f"   训练配置只接受: {TRAIN_PARAMS}")
        return False
    
    return True


def split_params(all_params):
    """
    自动分离模型参数和训练配置
    
    Args:
        all_params: 所有参数字典
        
    Returns:
        model_params: 模型参数
        train_params: 训练配置
        unknown_params: 未知参数
    """
    model_params = {k: v for k, v in all_params.items() if k in MODEL_PARAMS}
    train_params = {k: v for k, v in all_params.items() if k in TRAIN_PARAMS}
    unknown_params = {k: v for k, v in all_params.items() 
                      if k not in MODEL_PARAMS and k not in TRAIN_PARAMS}
    
    if unknown_params:
        print(f"⚠️  发现未知参数: {list(unknown_params.keys())}")
    
    return model_params, train_params, unknown_params


if __name__ == "__main__":
    # 测试
    print("=" * 60)
    print("测试参数验证")
    print("=" * 60)
    
    # 测试 1: 正确的模型参数
    params1 = {
        'n_features': 8,
        'n_classes': 9,
        'd_token': 64,
        'n_heads': 4,
        'n_layers': 3,
        'dropout': 0.3
    }
    print(f"\n测试 1: 正确的模型参数")
    print(f"  参数: {params1}")
    print(f"  验证结果: {'✅ 有效' if validate_model_params(params1) else '❌ 无效'}")
    
    # 测试 2: 包含训练配置的模型参数（错误）
    params2 = {
        'n_features': 8,
        'n_classes': 9,
        'learning_rate': 0.001,  # ❌ 这是训练配置
        'dropout': 0.3
    }
    print(f"\n测试 2: 包含训练配置的模型参数（错误）")
    print(f"  参数: {params2}")
    print(f"  验证结果: {'✅ 有效' if validate_model_params(params2) else '❌ 无效'}")
    
    # 测试 3: 自动分离参数
    params3 = {
        'n_features': 8,
        'n_classes': 9,
        'd_token': 64,
        'learning_rate': 0.001,
        'batch_size': 256,
        'dropout': 0.3
    }
    print(f"\n测试 3: 自动分离参数")
    print(f"  所有参数: {params3}")
    model_params, train_params, unknown_params = split_params(params3)
    print(f"  模型参数: {model_params}")
    print(f"  训练配置: {train_params}")
    print(f"  未知参数: {unknown_params}")