"""
数据管理器模块

提供数据管理功能，包括：
- 数据备份和恢复
- 数据清理和整理
- 数据格式转换
- 数据验证
"""

import os
import shutil
import json
import pandas as pd
import sqlite3
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from ..config import config


# 配置日志
logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)


class DataManager:
    """数据管理器"""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化数据管理器
        
        Args:
            data_dir: 数据目录路径，如果为None则使用配置中的默认路径
        """
        self.data_dir = Path(data_dir or config.paths.data_dir)
        self.backup_dir = Path(config.database.backup_path)
        
        # 确保目录存在
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"初始化数据管理器，数据目录: {self.data_dir}")
    
    def backup_database(self, db_path: Optional[str] = None, backup_name: Optional[str] = None) -> str:
        """
        备份数据库文件
        
        Args:
            db_path: 数据库文件路径，如果为None则使用配置中的默认路径
            backup_name: 备份文件名，如果为None则自动生成
            
        Returns:
            备份文件路径
        """
        db_path = db_path or config.database.path
        
        if not os.path.exists(db_path):
            logger.error(f"数据库文件不存在: {db_path}")
            raise FileNotFoundError(f"数据库文件不存在: {db_path}")
        
        # 生成备份文件名
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"twitter_simulation_backup_{timestamp}.db"
        
        backup_path = self.backup_dir / backup_name
        
        try:
            # 复制数据库文件
            shutil.copy2(db_path, backup_path)
            logger.info(f"✓ 数据库备份成功: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"❌ 数据库备份失败: {e}")
            raise
    
    def restore_database(self, backup_path: str, target_path: Optional[str] = None) -> str:
        """
        恢复数据库文件
        
        Args:
            backup_path: 备份文件路径
            target_path: 目标文件路径，如果为None则使用配置中的默认路径
            
        Returns:
            恢复的文件路径
        """
        target_path = target_path or config.database.path
        
        if not os.path.exists(backup_path):
            logger.error(f"备份文件不存在: {backup_path}")
            raise FileNotFoundError(f"备份文件不存在: {backup_path}")
        
        try:
            # 确保目标目录存在
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # 复制备份文件到目标位置
            shutil.copy2(backup_path, target_path)
            logger.info(f"✓ 数据库恢复成功: {target_path}")
            return target_path
            
        except Exception as e:
            logger.error(f"❌ 数据库恢复失败: {e}")
            raise
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        列出所有备份文件
        
        Returns:
            备份文件信息列表
        """
        backups = []
        
        try:
            for backup_file in self.backup_dir.glob("*.db"):
                stat = backup_file.stat()
                backups.append({
                    "name": backup_file.name,
                    "path": str(backup_file),
                    "size": stat.st_size,
                    "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            
            # 按修改时间排序
            backups.sort(key=lambda x: x["modified_time"], reverse=True)
            
            logger.info(f"找到 {len(backups)} 个备份文件")
            return backups
            
        except Exception as e:
            logger.error(f"❌ 列出备份文件失败: {e}")
            return []
    
    def clean_old_backups(self, keep_count: int = 10) -> int:
        """
        清理旧的备份文件，保留指定数量的最新备份
        
        Args:
            keep_count: 保留的备份文件数量
            
        Returns:
            删除的备份文件数量
        """
        backups = self.list_backups()
        
        if len(backups) <= keep_count:
            logger.info("备份文件数量未超过限制，无需清理")
            return 0
        
        # 删除多余的备份文件
        deleted_count = 0
        for backup in backups[keep_count:]:
            try:
                os.remove(backup["path"])
                deleted_count += 1
                logger.debug(f"删除备份文件: {backup['name']}")
            except Exception as e:
                logger.error(f"删除备份文件失败 {backup['name']}: {e}")
        
        logger.info(f"✓ 清理了 {deleted_count} 个旧备份文件")
        return deleted_count
    
    def organize_data_files(self) -> Dict[str, int]:
        """
        整理数据文件，按类型分类存储
        
        Returns:
            整理结果统计
        """
        # 创建子目录
        subdirs = {
            "raw": self.data_dir / "raw",
            "processed": self.data_dir / "processed", 
            "output": self.data_dir / "output",
            "temp": self.data_dir / "temp"
        }
        
        for subdir in subdirs.values():
            subdir.mkdir(exist_ok=True)
        
        # 移动文件
        moved_files = {}
        
        try:
            for file_path in self.data_dir.glob("*"):
                if file_path.is_file():
                    # 根据文件扩展名分类
                    if file_path.suffix in ['.csv', '.json', '.db']:
                        if 'raw' in file_path.name.lower() or 'original' in file_path.name.lower():
                            target_dir = subdirs["raw"]
                        elif 'processed' in file_path.name.lower() or 'result' in file_path.name.lower():
                            target_dir = subdirs["processed"]
                        elif 'output' in file_path.name.lower() or 'final' in file_path.name.lower():
                            target_dir = subdirs["output"]
                        else:
                            target_dir = subdirs["processed"]
                        
                        # 移动文件
                        target_path = target_dir / file_path.name
                        if target_path != file_path:
                            shutil.move(str(file_path), str(target_path))
                            moved_files[target_dir.name] = moved_files.get(target_dir.name, 0) + 1
            
            logger.info(f"✓ 数据文件整理完成: {moved_files}")
            return moved_files
            
        except Exception as e:
            logger.error(f"❌ 数据文件整理失败: {e}")
            return {}
    
    def validate_database(self, db_path: Optional[str] = None) -> Dict[str, Any]:
        """
        验证数据库完整性
        
        Args:
            db_path: 数据库文件路径，如果为None则使用配置中的默认路径
            
        Returns:
            验证结果
        """
        db_path = db_path or config.database.path
        
        validation_result = {
            "database_exists": False,
            "tables": [],
            "table_counts": {},
            "integrity_check": False,
            "errors": []
        }
        
        try:
            if not os.path.exists(db_path):
                validation_result["errors"].append("数据库文件不存在")
                return validation_result
            
            validation_result["database_exists"] = True
            
            # 连接数据库
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 获取所有表
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            validation_result["tables"] = tables
            
            # 检查每个表的记录数
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    validation_result["table_counts"][table] = count
                except Exception as e:
                    validation_result["errors"].append(f"检查表 {table} 失败: {e}")
            
            # 完整性检查
            try:
                cursor.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]
                validation_result["integrity_check"] = integrity_result == "ok"
                if integrity_result != "ok":
                    validation_result["errors"].append(f"数据库完整性检查失败: {integrity_result}")
            except Exception as e:
                validation_result["errors"].append(f"完整性检查失败: {e}")
            
            conn.close()
            
            logger.info(f"✓ 数据库验证完成: {len(validation_result['tables'])} 个表")
            return validation_result
            
        except Exception as e:
            validation_result["errors"].append(f"数据库验证失败: {e}")
            logger.error(f"❌ 数据库验证失败: {e}")
            return validation_result
    
    def export_data_summary(self, output_path: Optional[str] = None) -> str:
        """
        导出数据摘要报告
        
        Args:
            output_path: 输出文件路径，如果为None则自动生成
            
        Returns:
            输出文件路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = config.get_file_path("output", f"data_summary_{timestamp}.json")
        
        try:
            summary = {
                "data_directory": str(self.data_dir),
                "backup_directory": str(self.backup_dir),
                "backups": self.list_backups(),
                "database_validation": self.validate_database(),
                "file_organization": self.organize_data_files(),
                "generated_time": datetime.now().isoformat()
            }
            
            # 保存摘要
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✓ 数据摘要已导出: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ 导出数据摘要失败: {e}")
            raise
    
    def cleanup_temp_files(self) -> int:
        """
        清理临时文件
        
        Returns:
            删除的文件数量
        """
        temp_dir = self.data_dir / "temp"
        deleted_count = 0
        
        try:
            if temp_dir.exists():
                for file_path in temp_dir.glob("*"):
                    if file_path.is_file():
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"删除临时文件: {file_path.name}")
            
            logger.info(f"✓ 清理了 {deleted_count} 个临时文件")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ 清理临时文件失败: {e}")
            return 0 