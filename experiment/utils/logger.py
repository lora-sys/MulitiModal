"""
统一日志系统

提供统一的日志管理功能，支持控制台和文件输出
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


# 日志级别映射
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""

    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
        'RESET': '\033[0m',       # 重置
    }

    def format(self, record):
        # 保存原始levelname
        original_levelname = record.levelname

        # 添加颜色（仅用于控制台输出）
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"

        # 格式化记录
        formatted = super().format(record)

        # 恢复原始levelname（供其他handlers使用）
        record.levelname = original_levelname

        return formatted


def setup_logging(
    log_dir: str = 'logs',
    level: str = 'INFO',
    console: bool = True,
    file: bool = True,
    filename: Optional[str] = None
) -> None:
    """设置全局日志配置

    Args:
        log_dir: 日志文件目录
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: 是否输出到控制台
        file: 是否输出到文件
        filename: 日志文件名（默认为当前时间戳）
    """
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))

    # 清除现有的处理器
    root_logger.handlers.clear()

    # 日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))

        # 使用带颜色的格式化器
        console_formatter = ColoredFormatter(log_format, datefmt=date_format)
        console_handler.setFormatter(console_formatter)

        root_logger.addHandler(console_handler)

    # 文件处理器
    if file:
        if filename is None:
            from datetime import datetime
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        file_path = log_path / filename
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别

        # 文件使用标准格式化器（不带颜色）
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)

        root_logger.addHandler(file_handler)


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """获取日志记录器

    Args:
        name: 日志记录器名称（通常使用 __name__）
        level: 日志级别（可选，如果不设置则使用全局配置）

    Returns:
        logging.Logger: 日志记录器实例

    Examples:
        >>> from experiment.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("开始训练...")
        >>> logger.debug(f"批次大小: {batch_size}")
        >>> logger.error("模型加载失败")
    """
    logger = logging.getLogger(name)

    # 如果指定了级别，则设置
    if level is not None:
        logger.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))

    return logger


# 注意：不再自动初始化日志系统
# 使用者需要显式调用 setup_logging() 来配置日志系统
# 例如：
#   from experiment.utils.logger import setup_logging
#   setup_logging(log_dir='experiment/logs', level='INFO')