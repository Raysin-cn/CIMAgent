"""
CIM (Content Influence Modeling) 模块

该模块提供了社交媒体内容影响建模的核心功能，包括：
- 立场检测和分析
- 帖子生成
- 数据准备和处理
- 可视化分析
"""

from .core.stance_detector import StanceDetector
from .core.post_generator import PostGenerator, GeneratedPost
from .core.data_injector import OasisPostInjector
from .utils.data_manager import DataManager
from .utils.visualizer import StanceVisualizer
from .config import Config

__version__ = "1.0.0"
__author__ = "CIMAgent Team"

__all__ = [
    "StanceDetector",
    "PostGenerator", 
    "GeneratedPost",
    "OasisPostInjector",
    "DataManager",
    "StanceVisualizer",
    "Config"
]
