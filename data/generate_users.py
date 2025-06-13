import pandas as pd
import json
from typing import List, Dict, Tuple
import asyncio
from datetime import datetime
import os
from collections import defaultdict
from camel.models import ModelFactory
from camel.types import ModelPlatformType
import random
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
# 设置matplotlib支持中文字体显示
import matplotlib.font_manager as fm

# 直接设置文泉驿字体（系统中已安装）
try:
    # 方法1：直接设置字体文件路径
    font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        print(f"成功设置字体: {font_prop.get_name()}")
    else:
        # 方法2：使用字体名称
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
        print("使用字体名称设置")
    
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 测试中文字体
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.text(0.5, 0.5, '测试', fontsize=12)
    plt.close(fig)
    print("中文字体测试成功")
    
except Exception as e:
    print(f"中文字体设置失败: {e}")
    print("将使用英文标签")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

def visualize_adj_matrix(adj_matrix=None, csv_path=None, title="Social Network Visualization", 
                        save_path='./data/network_visualization.png', 
                        use_directed_graph=True, node_size_factor=100, 
                        edge_width_factor=0.5, figsize=(12, 10)):
    """
    可视化社交网络关系图
    
    Args:
        adj_matrix: 邻接矩阵（可选）
        csv_path: CSV文件路径（可选，如果提供则直接从CSV提取关系）
        title: 图表标题
        save_path: 保存路径
        use_directed_graph: 是否使用有向图
        node_size_factor: 节点大小因子
        edge_width_factor: 边宽度因子
        figsize: 图形大小
    """
    # 科研绘图配色方案
    scientific_colors = {
        'background': '#f8f9fa',
        'nodes': '#2c3e50',
        'edges': '#34495e',
        'highlight': '#e74c3c',
        'secondary': '#3498db',
        'tertiary': '#27ae60'
    }
    
    # 创建自定义颜色映射
    node_cmap = LinearSegmentedColormap.from_list('scientific', 
                                                 ['#ecf0f1', '#2c3e50'], N=256)
    edge_cmap = LinearSegmentedColormap.from_list('edge_scientific', 
                                                 ['#bdc3c7', '#34495e'], N=256)
    
    if csv_path is not None:
        # 从CSV文件直接提取关注关系
        df = pd.read_csv(csv_path)
        G = nx.DiGraph() if use_directed_graph else nx.Graph()
        
        # 添加节点
        for _, row in df.iterrows():
            user_id = str(row['user_id'])
            G.add_node(user_id, 
                      name=row.get('name', f'user_{user_id}'),
                      followers_count=row.get('followers_count', 0),
                      following_count=row.get('following_count', 0),
                      description=row.get('description', ''))
        
        # 添加边（关注关系）
        for _, row in df.iterrows():
            user_id = str(row['user_id'])
            following_list = row.get('following_list', [])
            
            # 处理following_list（可能是字符串形式的列表）
            if isinstance(following_list, str):
                try:
                    following_list = eval(following_list)
                except:
                    following_list = []
            
            # 添加关注关系边
            for following_id in following_list:
                if str(following_id) in G.nodes():
                    G.add_edge(user_id, str(following_id))
        
        print(f"从CSV构建了包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边的网络")
        
    elif adj_matrix is not None:
        # 使用提供的邻接矩阵
        adj_matrix = np.array(adj_matrix, dtype=float)
        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph() if use_directed_graph else nx.Graph())
        print(f"从邻接矩阵构建了包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边的网络")
    else:
        raise ValueError("必须提供adj_matrix或csv_path参数之一")
    
    # 计算网络指标
    if G.number_of_nodes() > 0:
        # 度中心性
        degree_centrality = nx.degree_centrality(G)
        # 介数中心性
        betweenness_centrality = nx.betweenness_centrality(G)
        # 接近中心性
        closeness_centrality = nx.closeness_centrality(G)
        
        # 计算节点大小（基于度中心性）
        node_sizes = [degree_centrality[node] * node_size_factor + 50 for node in G.nodes()]
        
        # 计算边权重（基于目标节点的度中心性）
        edge_weights = [degree_centrality[edge[1]] * edge_width_factor + 0.1 for edge in G.edges()]
        
        # 计算节点颜色（基于介数中心性）
        node_colors = [betweenness_centrality[node] for node in G.nodes()]
        
        # 计算边颜色（基于目标节点的接近中心性）
        edge_colors = [closeness_centrality[edge[1]] for edge in G.edges()]
        
        # 创建图形
        plt.figure(figsize=figsize, facecolor=scientific_colors['background'])
        
        # 设置布局
        if G.number_of_nodes() <= 50:
            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        else:
            pos = nx.kamada_kawai_layout(G)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, 
                              edge_color=edge_colors,
                              edge_cmap=edge_cmap,
                              width=edge_weights,
                              alpha=0.6,
                              arrows=use_directed_graph,
                              arrowsize=10,
                              arrowstyle='->',
                              connectionstyle='arc3,rad=0.1')
        
        # 绘制节点
        nodes = nx.draw_networkx_nodes(G, pos,
                                      node_color=node_colors,
                                      node_size=node_sizes,
                                      cmap=node_cmap,
                                      alpha=0.8,
                                      edgecolors=scientific_colors['nodes'],
                                      linewidths=1)
        
        # 添加节点标签（只显示重要节点）
        important_nodes = [node for node in G.nodes() 
                          if degree_centrality[node] > np.percentile(list(degree_centrality.values()), 75)]
        
        if len(important_nodes) <= 20:  # 只标注重要节点，避免标签重叠
            labels = {node: G.nodes[node].get('name', node) for node in important_nodes}
            nx.draw_networkx_labels(G, pos, labels, 
                                  font_size=8, 
                                  font_color=scientific_colors['nodes'],
                                  font_weight='bold')
        
        # 设置图形属性
        plt.title(title, fontsize=16, fontweight='bold', color=scientific_colors['nodes'])
        plt.axis('off')
        
        # 添加颜色条
        if G.number_of_nodes() > 0:
            sm = plt.cm.ScalarMappable(cmap=node_cmap, norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
            cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8, aspect=20)
            cbar.set_label('Betweenness Centrality', fontsize=10, color=scientific_colors['nodes'])
            cbar.ax.tick_params(colors=scientific_colors['nodes'])
        
        # 添加网络统计信息
        stats_text = f"""
网络统计:
• 节点数: {G.number_of_nodes()}
• 边数: {G.number_of_edges()}
• 平均度: {np.mean([d for n, d in G.degree()]):.2f}
• 网络密度: {nx.density(G):.4f}
• 连通分量数: {nx.number_strongly_connected_components(G) if use_directed_graph else nx.number_connected_components(G)}
        """
        
        plt.figtext(0.02, 0.02, stats_text, fontsize=9, 
                   color=scientific_colors['nodes'], 
                   bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor=scientific_colors['background'], 
                           edgecolor=scientific_colors['secondary'], 
                           alpha=0.8))
        
        # 保存图形
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor=scientific_colors['background'])
        plt.close()
        
        print(f"网络可视化已保存到: {save_path}")
        
        # 返回网络对象和统计信息
        return G, {
            'degree_centrality': degree_centrality,
            'betweenness_centrality': betweenness_centrality,
            'closeness_centrality': closeness_centrality
        }
    else:
        print("网络为空，无法可视化")
        return None, None

class UserDataProcessor:
    def __init__(self, csv_path: str):
        """
        初始化用户数据处理器
        
        Args:
            csv_path: CSV文件路径
        """
        self.csv_path = csv_path
        self.model = ModelFactory.create(
            model_platform=ModelPlatformType.VLLM,
            model_type="/data/model/Qwen3-14B",
            url="http://localhost:21474/v1",
            model_config_dict={"max_tokens": 32000}
        )
        self.df = None
        self.user_descriptions = {}
        
    def load_data(self):
        """加载CSV数据"""
        self.df = pd.read_csv(self.csv_path)
        # 创建用户描述字典
        for _, row in self.df.iterrows():
            self.user_descriptions[str(row["user_id"])] = {
                "description": row["description"],
                "user_char": row["user_char"]
            }
        
    async def enhance_user_description(self, description: str) -> str:
        """
        使用LLM增强用户描述
        
        Args:
            description: 原始用户描述
            
        Returns:
            增强后的用户描述
        """
        if not description or pd.isna(description):
            return "No description available"
            
        try:
            messages = [
                {"role": "system", "content": "你是一个专业的社交媒体用户分析专家。请根据用户的原始描述，生成一个更详细、更专业的用户画像描述。保持原始信息的同时，添加更多专业见解。"},
                {"role": "user", "content": f"请分析并增强以下用户描述：{description}"}
            ]
            response = await self.model.arun(messages)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error enhancing description: {e}")
            return description

    async def analyze_potential_followers(self, user_id: str, candidate_users: List[Dict]) -> List[Tuple[str, float, float]]:
        """
        分析用户可能关注和被关注的用户
        
        Args:
            user_id: 当前用户ID
            candidate_users: 候选用户列表，每个元素包含id和描述信息
            
        Returns:
            列表，每个元素为(目标用户ID, 关注可能性分数, 被关注可能性分数)
        """
        user_desc = self.user_descriptions.get(user_id, {})
        if not user_desc:
            return []
            
        try:
            # 构建候选用户描述
            candidates_desc = "\n".join([
                f"用户ID: {c['id']}\n描述: {c['description']}\n特征: {c['user_char']}"
                for c in candidate_users
            ])
            
            messages = [
                {"role": "system", "content": """你是一个社交媒体关系分析专家。请分析当前用户与候选用户之间的双向关系。
                对于每个候选用户，请分别判断：
                1. 当前用户是否可能关注该候选用户
                2. 该候选用户是否可能关注当前用户
                
                请考虑以下因素：
                1. 共同的兴趣爱好
                2. 专业领域
                3. 生活方式
                4. 价值观
                5. 社交圈子
                6. 影响力差异
                7. 内容相关性
                
                请严格按照以下格式回复，每个候选用户一行：
                用户ID: xxx, 关注可能性: 0.8, 被关注可能性: 0.3"""},
                {"role": "user", "content": f"""当前用户描述：
                {user_desc['description']}
                {user_desc['user_char']}
                
                候选用户列表：
                {candidates_desc}
                
                请分析当前用户与每个候选用户之间的双向关注关系，并给出可能性分数。"""}
            ]
            
            response = await self.model.arun(messages)
            
            # 解析响应
            results = []
            content = response.choices[0].message.content
            print(content)
            for line in content.strip().split('\n'):
                try:
                    # 解析每行的结果
                    parts = line.split(',')
                    target_id = parts[0].split(':')[1].strip()
                    follow_score = float(parts[1].split(':')[1].strip())
                    followed_score = float(parts[2].split(':')[1].strip())
                    results.append((target_id, follow_score, followed_score))
                except:
                    continue
                    
            return results
                
        except Exception as e:
            print(f"Error analyzing potential followers: {e}")
            return []
            
    async def generate_relationships(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        生成用户之间的关注关系
        
        Returns:
            (关注关系字典, 被关注关系字典)
        """
        following_relationships = defaultdict(list)
        followers_relationships = defaultdict(list)
        threshold = 0.6  # 关注可能性阈值
        
        # 为每个用户选择约10%的其他用户作为候选
        sample_size = max(1, int(len(self.df) * 0.1))
        relation_tasks = []
        
        for _, row in self.df.iterrows():
            user_id = str(row["user_id"])
            
            # 随机选择候选用户
            other_users = self.df[self.df['user_id'] != user_id].sample(n=sample_size)
            
            # 构建候选用户列表
            candidate_users = []
            for _, target_row in other_users.iterrows():
                target_id = str(target_row["user_id"])
                candidate_users.append({
                    "id": target_id,
                    "description": self.user_descriptions[target_id]["description"],
                    "user_char": self.user_descriptions[target_id]["user_char"]
                })
            
            # 分析潜在关注关系
            relation_tasks.append(self.analyze_potential_followers(user_id, candidate_users))
            
        print("开始分析潜在关注关系")
        results = await asyncio.gather(*relation_tasks)
        print("分析潜在关注关系完成")
        
        # 处理结果
        for user_id, relations in zip(self.df["user_id"], results):
            for target_id, follow_score, followed_score in relations:
                # 处理关注关系
                if follow_score >= threshold:
                    following_relationships[str(user_id)].append(target_id)
                
                # 处理被关注关系
                if followed_score >= threshold:
                    followers_relationships[target_id].append(str(user_id))
                        
        return dict(following_relationships), dict(followers_relationships)
            
    async def process_user_data(self) -> List[Dict]:
        """
        处理用户数据并返回增强后的用户列表
        
        Returns:
            处理后的用户数据列表
        """
        if self.df is None:
            self.load_data()
            
        # 生成关注关系
        following_relationships, followers_relationships = await self.generate_relationships()
        
        # 创建增强后的用户数据
        enhance_user_data = self.df.copy()
        
        # 初始化关注列表和粉丝列表列
        enhance_user_data['following_list'] = [[] for _ in range(len(enhance_user_data))]
        enhance_user_data['followers_list'] = [[] for _ in range(len(enhance_user_data))]
        
        # 更新关注关系
        for user_id, following_list in following_relationships.items():
            mask = enhance_user_data['user_id'] == int(user_id)
            if mask.any():
                idx = enhance_user_data[mask].index[0]
                enhance_user_data.at[idx, 'following_list'] = following_list
                enhance_user_data.at[idx, 'following_agentid_list'] = [enhance_user_data[enhance_user_data['user_id'] == int(following_id)].index[0] for following_id in following_list]
        
        # 更新粉丝关系
        for user_id, followers_list in followers_relationships.items():
            mask = enhance_user_data['user_id'] == int(user_id)
            if mask.any():
                idx = enhance_user_data[mask].index[0]
                enhance_user_data.at[idx, 'followers_list'] = followers_list
        
        # 生成邻接矩阵并可视化
        n_users = len(enhance_user_data)
        adj_matrix = np.zeros((n_users, n_users))
        for i, row in enhance_user_data.iterrows():
            for following_id in row['following_list']:
                j = enhance_user_data[enhance_user_data['user_id'] == int(following_id)].index
                if len(j) > 0:
                    adj_matrix[i, j[0]] = 1
        
        # 使用新的可视化方法
        visualize_adj_matrix(adj_matrix=adj_matrix, 
                           title="User Following Network Visualization",
                           save_path='./data/adj_matrix_network.png')
        
        return enhance_user_data
    
    async def save_processed_data(self, output_path: str):
        """
        保存处理后的数据
        
        Args:
            output_path: 输出文件路径
        """
        enhance_user_data = await self.process_user_data()
        # 将follower_list和following_list从字符串列表转换为整数列表
        enhance_user_data['followers_list'] = enhance_user_data['followers_list'].apply(
            lambda x: [int(i) for i in x]
        )
        enhance_user_data['following_list'] = enhance_user_data['following_list'].apply(
            lambda x: [int(i) for i in x]
        )
        enhance_user_data.to_csv(output_path, index=False)

async def main():
    # 初始化处理器
    processor = UserDataProcessor(
        csv_path="data/twitter_dataset/user_all_id_time.csv"
    )
    
    # 处理数据并保存
    output_path = "data/CIM_experiments/users_info.csv"
    await processor.save_processed_data(output_path)
    print(f"处理完成，数据已保存到: {output_path}")

def following_list_to_following_agentid_list(csv_path: str):
    user_data = pd.read_csv(csv_path)
    for index, row in user_data.iterrows():
        if row['following_list'] is not None and isinstance(row['following_list'], str):
            # 将字符串形式的列表转换为实际的列表
            user_id = row['user_id']
            following_list = eval(row['following_list'])
            following_agentid_list = []
            for following_id in following_list:
                if following_id == user_id:
                    continue
                # 检查是否存在匹配的用户ID
                matched_users = user_data[user_data['user_id'] == int(following_id)]
                if not matched_users.empty:
                    following_agentid_list.append(int(matched_users.index[0]))
            user_data.at[index, 'following_agentid_list'] = following_agentid_list
    user_data.to_csv(csv_path, index=False)

if __name__ == "__main__":
    # asyncio.run(main())
    # following_list_to_following_agentid_list("data/twitter_dataset_CIM/processed_users.csv")
    visualize_adj_matrix(csv_path="./data/CIM_experiments/users_info.csv")