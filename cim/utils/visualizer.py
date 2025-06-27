"""
可视化工具模块

提供数据可视化功能，包括：
- 立场分布可视化
- 时间演化分析图表
- 用户网络图
- 统计图表生成
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

from ..config import config


# 配置日志
logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)


class StanceVisualizer:
    """立场可视化器"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录，如果为None则使用配置中的默认路径
        """
        self.output_dir = Path(output_dir or config.paths.figs_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置matplotlib样式
        plt.style.use(config.visualization.style)
        sns.set_palette(config.visualization.color_palette)
        
        logger.info(f"初始化可视化器，输出目录: {self.output_dir}")
    
    def plot_stance_distribution(self, 
                               stance_data: List[Dict], 
                               title: str = "立场分布",
                               save_name: Optional[str] = None) -> str:
        """
        绘制立场分布图
        
        Args:
            stance_data: 立场数据列表
            title: 图表标题
            save_name: 保存文件名，如果为None则自动生成
            
        Returns:
            保存的文件路径
        """
        try:
            # 提取立场信息
            stances = []
            for item in stance_data:
                if "error" not in item and "main_stance" in item:
                    stances.append(item["main_stance"])
            
            if not stances:
                logger.warning("没有有效的立场数据")
                return ""
            
            # 统计立场分布
            stance_counts = pd.Series(stances).value_counts()
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.visualization.figure_size)
            
            # 饼图
            colors = plt.cm.Set3(np.linspace(0, 1, len(stance_counts)))
            wedges, texts, autotexts = ax1.pie(
                stance_counts.values, 
                labels=stance_counts.index,
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            ax1.set_title(f"{title} - 饼图")
            
            # 柱状图
            bars = ax2.bar(stance_counts.index, stance_counts.values, color=colors)
            ax2.set_title(f"{title} - 柱状图")
            ax2.set_ylabel("用户数量")
            
            # 添加数值标签
            for bar, count in zip(bars, stance_counts.values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{count}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # 保存图表
            if save_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_name = f"stance_distribution_{timestamp}.{config.visualization.save_format}"
            
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=config.visualization.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✓ 立场分布图已保存: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"❌ 绘制立场分布图失败: {e}")
            return ""
    
    def plot_stance_evolution(self, 
                            evolution_data: Dict, 
                            title: str = "立场演化分析",
                            save_name: Optional[str] = None) -> str:
        """
        绘制立场演化图
        
        Args:
            evolution_data: 演化数据
            title: 图表标题
            save_name: 保存文件名，如果为None则自动生成
            
        Returns:
            保存的文件路径
        """
        try:
            if "user_evolution" not in evolution_data:
                logger.warning("演化数据格式不正确")
                return ""
            
            # 提取时间步数据
            timesteps = []
            stance_distributions = {}
            
            for user_id, user_data in evolution_data["user_evolution"].items():
                for stance_info in user_data.get("timestep_stances", []):
                    timestep = stance_info["timestep"]
                    stance = stance_info["stance"]
                    
                    if timestep not in timesteps:
                        timesteps.append(timestep)
                    
                    if timestep not in stance_distributions:
                        stance_distributions[timestep] = {}
                    
                    stance_distributions[timestep][stance] = stance_distributions[timestep].get(stance, 0) + 1
            
            timesteps.sort()
            
            # 创建数据框
            evolution_df = pd.DataFrame()
            for timestep in timesteps:
                for stance, count in stance_distributions[timestep].items():
                    evolution_df = evolution_df.append({
                        'timestep': timestep,
                        'stance': stance,
                        'count': count
                    }, ignore_index=True)
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # 堆叠面积图
            pivot_df = evolution_df.pivot(index='timestep', columns='stance', values='count').fillna(0)
            pivot_df.plot(kind='area', stacked=True, ax=ax1)
            ax1.set_title(f"{title} - 立场演化趋势")
            ax1.set_xlabel("时间步")
            ax1.set_ylabel("用户数量")
            ax1.legend(title="立场")
            
            # 热力图
            heatmap_data = evolution_df.pivot_table(
                index='stance', 
                columns='timestep', 
                values='count', 
                fill_value=0
            )
            sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax2)
            ax2.set_title(f"{title} - 立场分布热力图")
            ax2.set_xlabel("时间步")
            ax2.set_ylabel("立场")
            
            plt.tight_layout()
            
            # 保存图表
            if save_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_name = f"stance_evolution_{timestamp}.{config.visualization.save_format}"
            
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=config.visualization.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✓ 立场演化图已保存: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"❌ 绘制立场演化图失败: {e}")
            return ""
    
    def plot_individual_evolution(self, 
                                evolution_data: Dict, 
                                user_ids: Optional[List[str]] = None,
                                title: str = "个体立场演化",
                                save_name: Optional[str] = None) -> str:
        """
        绘制个体立场演化图
        
        Args:
            evolution_data: 演化数据
            user_ids: 要显示的用户ID列表，如果为None则显示所有用户
            title: 图表标题
            save_name: 保存文件名，如果为None则自动生成
            
        Returns:
            保存的文件路径
        """
        try:
            if "user_evolution" not in evolution_data:
                logger.warning("演化数据格式不正确")
                return ""
            
            # 选择用户
            available_users = list(evolution_data["user_evolution"].keys())
            if user_ids is None:
                user_ids = available_users[:10]  # 默认显示前10个用户
            else:
                user_ids = [uid for uid in user_ids if uid in available_users]
            
            if not user_ids:
                logger.warning("没有有效的用户数据")
                return ""
            
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for i, user_id in enumerate(user_ids[:4]):  # 最多显示4个用户
                if i >= len(axes):
                    break
                
                user_data = evolution_data["user_evolution"][user_id]
                timesteps = []
                stances = []
                confidences = []
                
                for stance_info in user_data.get("timestep_stances", []):
                    timesteps.append(stance_info["timestep"])
                    stances.append(stance_info["stance"])
                    confidences.append(stance_info["confidence"])
                
                if timesteps:
                    # 立场变化线图
                    ax = axes[i]
                    ax.plot(timesteps, stances, 'o-', linewidth=2, markersize=6)
                    ax.set_title(f"用户 {user_id} ({user_data.get('user_name', '')})")
                    ax.set_xlabel("时间步")
                    ax.set_ylabel("立场")
                    ax.grid(True, alpha=0.3)
                    
                    # 设置y轴标签
                    unique_stances = list(set(stances))
                    ax.set_yticks(range(len(unique_stances)))
                    ax.set_yticklabels(unique_stances)
            
            plt.tight_layout()
            
            # 保存图表
            if save_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_name = f"individual_evolution_{timestamp}.{config.visualization.save_format}"
            
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=config.visualization.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✓ 个体演化图已保存: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"❌ 绘制个体演化图失败: {e}")
            return ""
    
    def plot_confidence_analysis(self, 
                               stance_data: List[Dict], 
                               title: str = "置信度分析",
                               save_name: Optional[str] = None) -> str:
        """
        绘制置信度分析图
        
        Args:
            stance_data: 立场数据列表
            title: 图表标题
            save_name: 保存文件名，如果为None则自动生成
            
        Returns:
            保存的文件路径
        """
        try:
            # 提取置信度数据
            confidences = []
            stances = []
            
            for item in stance_data:
                if "error" not in item and "average_confidence" in item:
                    confidences.append(item["average_confidence"])
                    stances.append(item["main_stance"])
            
            if not confidences:
                logger.warning("没有有效的置信度数据")
                return ""
            
            # 创建图表
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 置信度分布直方图
            ax1.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title("置信度分布")
            ax1.set_xlabel("置信度")
            ax1.set_ylabel("频次")
            ax1.grid(True, alpha=0.3)
            
            # 置信度箱线图
            confidence_df = pd.DataFrame({'stance': stances, 'confidence': confidences})
            confidence_df.boxplot(column='confidence', by='stance', ax=ax2)
            ax2.set_title("各立场置信度分布")
            ax2.set_xlabel("立场")
            ax2.set_ylabel("置信度")
            
            # 置信度vs立场散点图
            for stance in set(stances):
                stance_confidences = [c for c, s in zip(confidences, stances) if s == stance]
                ax3.scatter([stance] * len(stance_confidences), stance_confidences, alpha=0.6)
            ax3.set_title("置信度vs立场")
            ax3.set_xlabel("立场")
            ax3.set_ylabel("置信度")
            ax3.tick_params(axis='x', rotation=45)
            
            # 置信度统计
            stats_df = confidence_df.groupby('stance')['confidence'].agg(['mean', 'std', 'count'])
            stats_df.plot(kind='bar', ax=ax4)
            ax4.set_title("各立场置信度统计")
            ax4.set_xlabel("立场")
            ax4.set_ylabel("置信度")
            ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # 保存图表
            if save_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_name = f"confidence_analysis_{timestamp}.{config.visualization.save_format}"
            
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=config.visualization.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✓ 置信度分析图已保存: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"❌ 绘制置信度分析图失败: {e}")
            return ""
    
    def create_comprehensive_report(self, 
                                  stance_data: List[Dict], 
                                  evolution_data: Optional[Dict] = None,
                                  title: str = "立场检测综合分析报告") -> str:
        """
        创建综合分析报告
        
        Args:
            stance_data: 立场数据列表
            evolution_data: 演化数据，可选
            title: 报告标题
            
        Returns:
            保存的文件路径
        """
        try:
            # 创建大图表
            fig = plt.figure(figsize=(20, 16))
            
            # 1. 立场分布饼图
            ax1 = plt.subplot(2, 3, 1)
            stances = [item["main_stance"] for item in stance_data if "error" not in item and "main_stance" in item]
            stance_counts = pd.Series(stances).value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(stance_counts)))
            ax1.pie(stance_counts.values, labels=stance_counts.index, autopct='%1.1f%%', colors=colors)
            ax1.set_title("立场分布")
            
            # 2. 置信度分布
            ax2 = plt.subplot(2, 3, 2)
            confidences = [item["average_confidence"] for item in stance_data if "error" not in item and "average_confidence" in item]
            ax2.hist(confidences, bins=15, alpha=0.7, color='lightcoral')
            ax2.set_title("置信度分布")
            ax2.set_xlabel("置信度")
            ax2.set_ylabel("频次")
            
            # 3. 用户活跃度分析
            ax3 = plt.subplot(2, 3, 3)
            post_counts = [item["posts_analyzed"] for item in stance_data if "error" not in item and "posts_analyzed" in item]
            ax3.hist(post_counts, bins=10, alpha=0.7, color='lightgreen')
            ax3.set_title("用户活跃度分析")
            ax3.set_xlabel("分析帖子数")
            ax3.set_ylabel("用户数")
            
            # 4. 立场vs置信度散点图
            ax4 = plt.subplot(2, 3, 4)
            stance_conf_df = pd.DataFrame([
                (item["main_stance"], item["average_confidence"]) 
                for item in stance_data 
                if "error" not in item and "main_stance" in item and "average_confidence" in item
            ], columns=['stance', 'confidence'])
            
            for stance in stance_conf_df['stance'].unique():
                subset = stance_conf_df[stance_conf_df['stance'] == stance]
                ax4.scatter(subset['stance'], subset['confidence'], alpha=0.6, label=stance)
            ax4.set_title("立场vs置信度")
            ax4.set_xlabel("立场")
            ax4.set_ylabel("置信度")
            ax4.legend()
            
            # 5. 如果有演化数据，显示演化趋势
            if evolution_data and "user_evolution" in evolution_data:
                ax5 = plt.subplot(2, 3, 5)
                # 简化的演化趋势
                timesteps = []
                stance_counts_evo = {}
                
                for user_data in evolution_data["user_evolution"].values():
                    for stance_info in user_data.get("timestep_stances", []):
                        timestep = stance_info["timestep"]
                        stance = stance_info["stance"]
                        
                        if timestep not in timesteps:
                            timesteps.append(timestep)
                        
                        if timestep not in stance_counts_evo:
                            stance_counts_evo[timestep] = {}
                        
                        stance_counts_evo[timestep][stance] = stance_counts_evo[timestep].get(stance, 0) + 1
                
                timesteps.sort()
                for stance in set().union(*[set(counts.keys()) for counts in stance_counts_evo.values()]):
                    counts = [stance_counts_evo.get(ts, {}).get(stance, 0) for ts in timesteps]
                    ax5.plot(timesteps, counts, 'o-', label=stance)
                
                ax5.set_title("立场演化趋势")
                ax5.set_xlabel("时间步")
                ax5.set_ylabel("用户数")
                ax5.legend()
            
            # 6. 统计摘要
            ax6 = plt.subplot(2, 3, 6)
            ax6.axis('off')
            
            # 计算统计信息
            valid_data = [item for item in stance_data if "error" not in item]
            total_users = len(valid_data)
            avg_confidence = np.mean([item["average_confidence"] for item in valid_data if "average_confidence" in item])
            most_common_stance = pd.Series([item["main_stance"] for item in valid_data if "main_stance" in item]).mode()[0]
            
            summary_text = f"""
            统计摘要:
            
            总用户数: {total_users}
            平均置信度: {avg_confidence:.2f}
            最常见立场: {most_common_stance}
            
            立场分布:
            {stance_counts.to_string()}
            
            生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # 保存报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"comprehensive_report_{timestamp}.{config.visualization.save_format}"
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=config.visualization.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✓ 综合分析报告已保存: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"❌ 创建综合分析报告失败: {e}")
            return "" 