"""
配置管理模块

集中管理CIM模块的所有配置参数，包括模型配置、数据库配置、路径配置等
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """模型配置"""
    platform: str = "VLLM"
    # model_type: str = "/data/model/Qwen3-14B"
    model_type: str = "/home/models/Qwen3-8B"
    url: str = "http://localhost:12345/v1"
    max_tokens: int = 10000
    temperature: float = 1.0
    timeout: int = 1000


@dataclass
class DatabaseConfig:
    """数据库配置"""
    path: str = "./data/twitter_simulation.db"
    backup_path: str = "./data/backup/"
    max_connections: int = 10


@dataclass
class PathConfig:
    """路径配置"""
    data_dir: str = "./data"
    output_dir: str = "./data/output"
    figs_dir: str = "./data/figs"
    processed_dir: str = "./data/processed"
    
    def __post_init__(self):
        """确保必要的目录存在"""
        for path in [self.data_dir, self.output_dir, self.figs_dir, self.processed_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)


@dataclass
class StanceConfig:
    """立场检测配置"""
    default_topic: str = "中美贸易关税"
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
    topics_file: str = "./data/raw/topics.json"
    users_file: str = "./data/raw/users_info.csv"


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
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
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
            self.database.path = os.getenv("CIM_DB_PATH")
        
        # 调试模式
        if os.getenv("CIM_DEBUG"):
            self.debug = os.getenv("CIM_DEBUG").lower() == "true"
            self.log_level = "DEBUG" if self.debug else "INFO"
    
    def _normalize_paths(self):
        """标准化路径为绝对路径"""
        base_path = Path.cwd()
        
        self.database.path = str(Path(self.database.path).resolve())
        self.database.backup_path = str(Path(self.database.backup_path).resolve())
        
        self.paths.data_dir = str(Path(self.paths.data_dir).resolve())
        self.paths.output_dir = str(Path(self.paths.output_dir).resolve())
        self.paths.figs_dir = str(Path(self.paths.figs_dir).resolve())
        
        self.post_generation.topics_file = str(Path(self.post_generation.topics_file).resolve())
        self.post_generation.users_file = str(Path(self.post_generation.users_file).resolve())
    
    def get_file_path(self, file_type: str, filename: str) -> str:
        """获取指定类型文件的完整路径"""
        path_map = {
            "output": self.paths.output_dir,
            "figs": self.paths.figs_dir,
            "data": self.paths.data_dir,
            "processed": self.paths.processed_dir
        }
        
        base_path = path_map.get(file_type, self.paths.data_dir)
        return str(Path(base_path) / filename)
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {
            "model": self.model.__dict__,
            "database": self.database.__dict__,
            "paths": self.paths.__dict__,
            "stance": self.stance.__dict__,
            "post_generation": self.post_generation.__dict__,
            "visualization": self.visualization.__dict__,
            "debug": self.debug,
            "log_level": self.log_level
        }


# 全局配置实例
config = Config() 