"""
核心功能模块

包含CIM模块的核心功能类：
- StanceDetector: 立场检测器
- PostGenerator: 帖子生成器  
- OasisPostInjector: 数据注入器
"""

from .stance_detector import StanceDetector
from .post_generator import PostGenerator, GeneratedPost
from .data_injector import OasisPostInjector

__all__ = ["StanceDetector", "PostGenerator", "GeneratedPost", "OasisPostInjector"] 