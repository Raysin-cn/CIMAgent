"""
配置管理模块

集中管理CIM模块的所有配置参数，包括模型配置、数据库配置、路径配置等
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from camel.types import ModelPlatformType, ModelType


@dataclass
class ModelConfig:
    """模型配置"""
    platform: ModelPlatformType = ModelPlatformType.VLLM
    model_type: str = "Qwen3-30B-A3B-GPTQ-Int4"
    url: str = "http://localhost:12345/v1"
    api_key: str = "NONE"
    max_tokens: int = 10000
    temperature: float = 0
    timeout: int = 30

    # platform: ModelPlatformType = ModelPlatformType.OPENAI
    # model_type: ModelType = ModelType.GPT_4O_MINI
    # url: str = "https://api.gptgod.online/v1/"
    # api_key: str = "sk-0F8p7ljy9VVJa0555Y8te4XSoINHh0t72WDooFhOHkxL0kTP"
    # # max_tokens: int = 10000
    # temperature: float = 1.0
    # timeout: int = 30
    


@dataclass
class PathConfig:
    """路径配置"""
    data_dir: str = "./data"
    db_path: str = "./data/twitter_simulation.db"
    users_file: str = "./data/users_info_new.csv"
    posts_file: str = "./data/generated_posts.csv"



@dataclass
class StanceConfig:
    """立场检测配置"""
    max_retries: int = 5
    batch_size: int = 50
    max_concurrent: int = 3
    post_limit: int = 3
    confidence_threshold: float = 0.7


@dataclass
class PostGenerationConfig:
    """帖子生成配置"""
    min_length: int = 50
    max_length: int = 280
    include_hashtags: bool = True
    include_emojis: bool = True


@dataclass
class VisualizationConfig:
    """可视化配置"""
    figure_size: tuple = (12, 8)
    dpi: int = 300
    style: str = "seaborn"
    color_palette: str = "Set2"
    save_format: str = "png"


@dataclass
class Config:
    """主配置类"""
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    stance: StanceConfig = field(default_factory=StanceConfig)
    post_generation: PostGenerationConfig = field(default_factory=PostGenerationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # 环境变量配置
    debug: bool = False
    log_level: str = "INFO"
    
    def __post_init__(self):
        """初始化后处理"""
        # 从环境变量加载配置
        self._load_from_env()
        
        # 确保所有路径都是绝对路径
        self._normalize_paths()
    
    def _load_from_env(self):
        """从环境变量加载配置"""
        # 模型配置
        if os.getenv("CIM_MODEL_PLATFORM"):
            self.model.platform = os.getenv("CIM_MODEL_PLATFORM")
        if os.getenv("CIM_MODEL_TYPE"):
            self.model.model_type = os.getenv("CIM_MODEL_TYPE")
        if os.getenv("CIM_MODEL_URL"):
            self.model.url = os.getenv("CIM_MODEL_URL")
        
        # 数据库配置
        if os.getenv("CIM_DB_PATH"):
            self.paths.db_path = os.getenv("CIM_DB_PATH")
        
        # 调试模式
        if os.getenv("CIM_DEBUG"):
            self.debug = os.getenv("CIM_DEBUG").lower() == "true"
            self.log_level = "DEBUG" if self.debug else "INFO"
    
    def _normalize_paths(self):
        """标准化路径为绝对路径"""
        base_path = Path.cwd()
        self.paths.data_dir = str(Path(self.paths.data_dir).resolve())
        self.paths.db_path = str(Path(self.paths.db_path).resolve())
        self.paths.users_file = str(Path(self.paths.users_file).resolve())
        self.paths.posts_file = str(Path(self.paths.posts_file).resolve())
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {
            "model": self.model.__dict__,
            "paths": self.paths.__dict__,
            "stance": self.stance.__dict__,
            "post_generation": self.post_generation.__dict__,
            "visualization": self.visualization.__dict__,
            "debug": self.debug,
            "log_level": self.log_level
        }


# 全局配置实例
config = Config() 