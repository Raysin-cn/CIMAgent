import sqlite3
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
from collections import Counter
import re
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
import sys
import networkx as nx
from matplotlib.font_manager import FontProperties

# 设置matplotlib支持中文显示
try:
    # 尝试使用系统中可能存在的中文字体
    chinese_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi']
    font_found = False
    for font in chinese_fonts:
        try:
            FontProperties(fname=font)
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            font_found = True
            break
        except:
            continue
    
    if not font_found:
        # 如果没有找到中文字体，使用系统默认字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
except:
    # 如果出现任何错误，使用系统默认字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def setup_nltk():
    """设置NLTK环境"""
    try:
        # 检查是否已安装NLTK
        import nltk
    except ImportError:
        print("正在安装NLTK...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
        import nltk

    # 下载必要的NLTK数据
    required_packages = ['punkt', 'stopwords']
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
        except LookupError:
            print(f"正在下载NLTK {package} 数据...")
            nltk.download(package, quiet=True)
            
    # 确保下载 punkt_tab
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("正在下载 punkt_tab 数据...")
        nltk.download('punkt_tab', quiet=True)

# 初始化NLTK
setup_nltk()

class AnalysisType(Enum):
    """分析类型枚举"""
    DB_STRUCTURE = auto()  # 数据库结构分析
    POST_CONTENT = auto()  # 帖子内容分析
    TRACE_ANALYSIS = auto()  # 用户行为分析
    DIRECTED_GRAPH = auto()  # 有向图分析

@dataclass
class AnalysisConfig:
    """分析配置类"""
    analyze_db_structure: bool = True
    analyze_posts: bool = False
    analyze_traces: bool = False
    analyze_directed_graph: bool = False  # 新增有向图分析配置
    output_dir: Optional[str] = None

class DBAnalyzer:
    def __init__(self, db_path: str):
        """
        初始化数据库分析器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.output_dir = None
        # 初始化英文停用词
        self.stop_words = set(stopwords.words('english'))
        # 添加一些额外的常见停用词
        self.stop_words.update({
            'http', 'https', 'www', 'com', 'org', 'net', 'html', 'htm',
            'rt', 'retweet', 'like', 'follow', 'following', 'follower',
            'tweet', 'twitter', 'status', 'update', 'post', 'comment',
            'reply', 'replies', 'mention', 'mentions', 'hashtag', 'hashtags',
            'amp', 'via', 'from', 'just', 'now', 'today', 'yesterday',
            'tomorrow', 'day', 'days', 'week', 'weeks', 'month', 'months',
            'year', 'years', 'time', 'times', 'good', 'great', 'nice',
            'best', 'better', 'well', 'much', 'many', 'lot', 'lots',
            'really', 'very', 'so', 'too', 'also', 'even', 'still',
            'back', 'new', 'old', 'first', 'last', 'next', 'previous',
            'one', 'two', 'three', 'four', 'five', 'six', 'seven',
            'eight', 'nine', 'ten', 'first', 'second', 'third', 'fourth',
            'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth'
        })
        
    def connect(self) -> bool:
        """连接到数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            return True
        except Exception as e:
            print(f"连接数据库失败: {str(e)}")
            return False
            
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None

    def get_tables(self) -> List[str]:
        """获取所有表名"""
        if not self.cursor:
            return []
            
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [table[0] for table in self.cursor.fetchall()]
        
    def get_table_info(self, table_name: str) -> Dict:
        """获取表的结构信息"""
        if not self.cursor:
            return {}
            
        self.cursor.execute(f"PRAGMA table_info({table_name});")
        columns = self.cursor.fetchall()
        
        return {
            'columns': [{'name': col[1], 'type': col[2], 'notnull': col[3], 'default': col[4]} 
                       for col in columns]
        }
        
    def get_table_stats(self, table_name: str) -> Dict:
        """获取表的统计信息"""
        if not self.cursor:
            return {}
            
        # 获取行数
        self.cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        row_count = self.cursor.fetchone()[0]
        
        # 获取表大小
        self.cursor.execute(f"SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size();")
        size = self.cursor.fetchone()[0]
        
        return {
            'row_count': row_count,
            'size_bytes': size
        }

    def analyze_db_structure(self) -> Dict:
        """分析数据库结构"""
        if not self.connect():
            return {}
            
        tables = self.get_tables()
        analysis = {
            'tables': {},
            'total_tables': len(tables),
            'total_size': 0
        }
        
        for table in tables:
            table_info = self.get_table_info(table)
            table_stats = self.get_table_stats(table)
            
            analysis['tables'][table] = {
                'structure': table_info,
                'statistics': table_stats
            }
            analysis['total_size'] += table_stats.get('size_bytes', 0)
            
        return analysis

    def _process_text(self, text: str) -> List[str]:
        """
        处理文本，包括分词和过滤
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 处理后的词列表
        """
        try:
            if not isinstance(text, str):
                return []
            
            # 转换为小写
            text = text.lower()
            # 移除URL
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            # 移除特殊字符和数字
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\d+', '', text)
            # 分词
            words = word_tokenize(text)
            # 过滤停用词和短词
            words = [word for word in words if word not in self.stop_words and len(word) > 2]
            return words
        except Exception as e:
            print(f"文本处理错误: {str(e)}")
            return []

    def analyze_posts(self) -> Dict:
        """分析帖子内容"""
        if not self.connect():
            return {}
            
        # 获取所有帖子
        self.cursor.execute("""
            SELECT p.post_id, p.content, p.created_at, p.num_likes, p.num_dislikes, p.num_shares,
                   u.user_name, u.name
            FROM post p
            LEFT JOIN user u ON p.user_id = u.user_id
        """)
        posts = self.cursor.fetchall()
        
        df = pd.DataFrame(posts, columns=['post_id', 'content', 'created_at', 'likes', 'dislikes', 'shares', 
                                        'user_name', 'name'])
        
        stats = {
            'total_posts': len(df),
            'avg_likes': df['likes'].mean(),
            'avg_dislikes': df['dislikes'].mean(),
            'avg_shares': df['shares'].mean(),
            'total_likes': df['likes'].sum(),
            'total_dislikes': df['dislikes'].sum(),
            'total_shares': df['shares'].sum(),
        }
        
        if not df.empty:
            # 内容分析
            all_content = ' '.join(df['content'].dropna().astype(str))
            words = self._process_text(all_content)
            word_freq = Counter(words)
            
            if word_freq:  # 只有在有词频数据时才生成词云
                stats['word_freq'] = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:50])
                
                # 生成词云
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    max_words=100,
                    contour_width=3,
                    contour_color='steelblue'
                ).generate_from_frequencies(word_freq)
                
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'post_wordcloud.png'))
                plt.close()
            else:
                print("警告：没有足够的词频数据来生成词云")
                stats['word_freq'] = {}
            
            # 时间分布
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['hour'] = df['created_at'].dt.hour
            hourly_posts = df.groupby('hour').size()
            stats['hourly_distribution'] = hourly_posts.to_dict()
            
        return stats

    def analyze_traces(self) -> Dict:
        """分析用户行为轨迹"""
        if not self.connect():
            return {}
            
        # 获取所有行为记录
        self.cursor.execute("""
            SELECT t.user_id, t.created_at, t.action, t.info, u.user_name
            FROM trace t
            LEFT JOIN user u ON t.user_id = u.user_id
            ORDER BY t.created_at
        """)
        traces = self.cursor.fetchall()
        
        df = pd.DataFrame(traces, columns=['user_id', 'created_at', 'action', 'info', 'user_name'])
        
        stats = {
            'total_traces': len(df),
            'unique_users': df['user_id'].nunique(),
            'action_types': df['action'].value_counts().to_dict()
        }
        
        if not df.empty:
            # 时间分析
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['hour'] = df['created_at'].dt.hour
            hourly_traces = df.groupby('hour').size()
            stats['hourly_distribution'] = hourly_traces.to_dict()
            
            # 用户活跃度分析
            user_activity = df.groupby('user_id').size()
            stats['user_activity'] = {
                'min_actions': user_activity.min(),
                'max_actions': user_activity.max(),
                'avg_actions': user_activity.mean(),
                'most_active_users': user_activity.nlargest(10).to_dict()
            }
            
            # 行为序列分析
            action_sequences = df.groupby('user_id')['action'].agg(list)
            stats['common_sequences'] = self._analyze_action_sequences(action_sequences)
            
        return stats

    def _analyze_action_sequences(self, action_sequences: pd.Series) -> Dict:
        """分析用户行为序列"""
        sequences = {}
        for actions in action_sequences:
            for i in range(len(actions)-1):
                seq = f"{actions[i]}->{actions[i+1]}"
                sequences[seq] = sequences.get(seq, 0) + 1
        return dict(sorted(sequences.items(), key=lambda x: x[1], reverse=True)[:20])

    def analyze_directed_graph(self) -> Dict:
        """分析用户互动关系的有向图"""
        if not self.connect():
            return {}
            
        # 获取关注关系
        self.cursor.execute("""
            SELECT follower_id, followee_id
            FROM follow
        """)
        follows = self.cursor.fetchall()
        
        # 获取点赞关系
        self.cursor.execute("""
            SELECT user_id, post_id
            FROM like
        """)
        likes = self.cursor.fetchall()
        
        # 获取点踩关系
        self.cursor.execute("""
            SELECT user_id, post_id
            FROM dislike
        """)
        dislikes = self.cursor.fetchall()
        
        # 获取帖子作者信息
        self.cursor.execute("""
            SELECT post_id, user_id, num_likes, num_dislikes, num_shares
            FROM post
        """)
        posts = self.cursor.fetchall()
        post_authors = {post_id: user_id for post_id, user_id, _, _, _ in posts}
        
        # 获取评论信息
        self.cursor.execute("""
            SELECT comment_id, post_id, user_id, num_likes, num_dislikes
            FROM comment
        """)
        comments = self.cursor.fetchall()
        
        # 获取评论点赞信息
        self.cursor.execute("""
            SELECT user_id, comment_id
            FROM comment_like
        """)
        comment_likes = self.cursor.fetchall()
        
        # 获取评论点踩信息
        self.cursor.execute("""
            SELECT user_id, comment_id
            FROM comment_dislike
        """)
        comment_dislikes = self.cursor.fetchall()
        
        # 构建互动关系图
        interaction_graph = {
            'follows': [],
            'likes': [],
            'dislikes': []
        }
        
        # 处理关注关系
        for follower_id, followee_id in follows:
            interaction_graph['follows'].append({
                'source': follower_id,
                'target': followee_id,
                'type': 'follow'
            })
            
        # 处理点赞关系
        for user_id, post_id in likes:
            if post_id in post_authors:
                interaction_graph['likes'].append({
                    'source': user_id,
                    'target': post_authors[post_id],
                    'type': 'like'
                })
                
        # 处理点踩关系
        for user_id, post_id in dislikes:
            if post_id in post_authors:
                interaction_graph['dislikes'].append({
                    'source': user_id,
                    'target': post_authors[post_id],
                    'type': 'dislike'
                })
        
        # 计算用户指标
        user_metrics = {}
        
        # 初始化所有用户的指标
        all_user_ids = set()
        for post_id, user_id, _, _, _ in posts:
            all_user_ids.add(user_id)
        for comment_id, post_id, user_id, _, _ in comments:
            all_user_ids.add(user_id)
        for follower_id, followee_id in follows:
            all_user_ids.add(follower_id)
            all_user_ids.add(followee_id)
            
        for user_id in all_user_ids:
            user_metrics[user_id] = {
                'post_count': 0,
                'post_likes_received': 0,
                'post_dislikes_received': 0,
                'post_shares_received': 0,
                'comment_count': 0,
                'comment_likes_received': 0,
                'comment_dislikes_received': 0,
                'followers_count': 0,
                'following_count': 0,
                'total_likes_given': 0,
                'total_dislikes_given': 0,
                'total_comment_likes_given': 0,
                'total_comment_dislikes_given': 0
            }
        
        # 统计发帖相关指标
        for post_id, user_id, num_likes, num_dislikes, num_shares in posts:
            user_metrics[user_id]['post_count'] += 1
            user_metrics[user_id]['post_likes_received'] += num_likes
            user_metrics[user_id]['post_dislikes_received'] += num_dislikes
            user_metrics[user_id]['post_shares_received'] += num_shares
        
        # 统计评论相关指标
        for comment_id, post_id, user_id, num_likes, num_dislikes in comments:
            user_metrics[user_id]['comment_count'] += 1
            user_metrics[user_id]['comment_likes_received'] += num_likes
            user_metrics[user_id]['comment_dislikes_received'] += num_dislikes
        
        # 统计关注关系
        for follower_id, followee_id in follows:
            user_metrics[follower_id]['following_count'] += 1
            user_metrics[followee_id]['followers_count'] += 1
        
        # 统计点赞和点踩行为
        for user_id, post_id in likes:
            user_metrics[user_id]['total_likes_given'] += 1
        
        for user_id, post_id in dislikes:
            user_metrics[user_id]['total_dislikes_given'] += 1
        
        for user_id, comment_id in comment_likes:
            user_metrics[user_id]['total_comment_likes_given'] += 1
        
        for user_id, comment_id in comment_dislikes:
            user_metrics[user_id]['total_comment_dislikes_given'] += 1
        
        # 计算用户影响力得分
        for user_id in user_metrics:
            metrics = user_metrics[user_id]
            # 影响力得分 = 关注者数 * 0.3 + 获得的点赞数 * 0.3 + 获得的评论数 * 0.2 + 获得的分享数 * 0.2  #TODO 影响力得分的计算
            metrics['influence_score'] = (
                metrics['followers_count'] * 0.3 +
                metrics['post_likes_received'] * 0.3 +
                metrics['comment_count'] * 0.2 +
                metrics['post_shares_received'] * 0.2
            )
        
        # 获取用户信息
        self.cursor.execute("""
            SELECT user_id, user_name, name
            FROM user
        """)
        user_info = {user_id: {'user_name': user_name, 'name': name} 
                    for user_id, user_name, name in self.cursor.fetchall()}
        
        # 将用户信息添加到指标中
        for user_id in user_metrics:
            if user_id in user_info:
                user_metrics[user_id].update(user_info[user_id])
        
        # 按影响力得分排序
        sorted_users = sorted(
            user_metrics.items(),
            key=lambda x: x[1]['influence_score'],
            reverse=True
        )
        
        # 统计信息
        stats = {
            'total_follows': len(interaction_graph['follows']),
            'total_likes': len(interaction_graph['likes']),
            'total_dislikes': len(interaction_graph['dislikes']),
            'unique_users': len(all_user_ids),
            'top_users': [
                {
                    'user_id': user_id,
                    'user_name': metrics.get('user_name', 'Unknown'),
                    'name': metrics.get('name', 'Unknown'),
                    'influence_score': metrics['influence_score'],
                    'followers_count': metrics['followers_count'],
                    'post_count': metrics['post_count'],
                    'post_likes_received': metrics['post_likes_received']
                }
                for user_id, metrics in sorted_users[:10]  # 只显示前10名用户
            ]
        }
        
        # 生成可视化
        if self.output_dir:
            self._generate_directed_graph_visualization(interaction_graph)
            self._generate_user_metrics_visualization(user_metrics)
            
        return {
            'interaction_graph': interaction_graph,
            'user_metrics': user_metrics,
            'statistics': stats
        }
        
    def _generate_user_metrics_visualization(self, user_metrics: Dict):
        """Generate user metrics visualization"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Prepare data
        metrics_data = []
        for user_id, metrics in user_metrics.items():
            if 'user_name' in metrics:  # Only process users with usernames
                metrics_data.append({
                    'user_name': metrics['user_name'],
                    'influence_score': metrics['influence_score'],
                    'followers_count': metrics['followers_count'],
                    'post_count': metrics['post_count'],
                    'post_likes_received': metrics['post_likes_received']
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(metrics_data)
        
        # Sort by influence score
        df = df.sort_values('influence_score', ascending=False).head(20)
        
        # Create plots
        plt.figure(figsize=(15, 10))
        
        # Plot influence score
        plt.subplot(2, 2, 1)
        sns.barplot(data=df, x='influence_score', y='user_name')
        plt.title('User Influence Score Ranking')
        plt.xlabel('Influence Score')
        plt.ylabel('Username')
        
        # Plot followers count
        plt.subplot(2, 2, 2)
        sns.barplot(data=df, x='followers_count', y='user_name')
        plt.title('User Followers Count Ranking')
        plt.xlabel('Followers Count')
        plt.ylabel('Username')
        
        # Plot post count
        plt.subplot(2, 2, 3)
        sns.barplot(data=df, x='post_count', y='user_name')
        plt.title('User Post Count Ranking')
        plt.xlabel('Post Count')
        plt.ylabel('Username')
        
        # Plot likes received
        plt.subplot(2, 2, 4)
        sns.barplot(data=df, x='post_likes_received', y='user_name')
        plt.title('User Likes Received Ranking')
        plt.xlabel('Likes Received')
        plt.ylabel('Username')
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, 'user_metrics.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    def _generate_directed_graph_visualization(self, interaction_graph: Dict):
        """生成有向图可视化"""
        import networkx as nx
        import matplotlib.pyplot as plt
        
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加边
        for edge_type, edges in interaction_graph.items():
            for edge in edges:
                G.add_edge(
                    edge['source'],
                    edge['target'],
                    type=edge['type']
                )
                
        # 设置布局
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # 绘制图形
        plt.figure(figsize=(12, 8))
        
        # 绘制不同类型的边
        edge_colors = {
            'follow': 'blue',
            'like': 'green',
            'dislike': 'red'
        }
        
        for edge_type in ['follow', 'like', 'dislike']:
            edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['type'] == edge_type]
            nx.draw_networkx_edges(
                G, pos,
                edgelist=edges,
                edge_color=edge_colors[edge_type],
                width=1,
                alpha=0.7,
                arrows=True,
                arrowsize=10,
                label=edge_type.capitalize()
            )
            
        # 绘制节点
        nx.draw_networkx_nodes(
            G, pos,
            node_size=100,
            node_color='lightgray',
            alpha=0.8
        )
        
        # 添加图例
        plt.legend()
        
        # 保存图形
        plt.savefig(
            os.path.join(self.output_dir, 'interaction_graph.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    def _generate_directed_graph_html(self, analysis: Dict) -> str:
        """Generate HTML for directed graph analysis"""
        html = """
            <div class="section directed-graph">
                <h2>User Interaction Analysis</h2>
                <div class="stats">
                    <h3>Basic Statistics</h3>
                    <ul>
        """
        
        for key, value in analysis['statistics'].items():
            if key != 'top_users':
                html += f"<li>{key}: {value}</li>"
                
        html += """
                    </ul>
                </div>
                
                <div class="top-users">
                    <h3>Most Influential Users</h3>
                    <table>
                        <tr>
                            <th>Username</th>
                            <th>Influence Score</th>
                            <th>Followers Count</th>
                            <th>Post Count</th>
                            <th>Likes Received</th>
                        </tr>
        """
        
        for user in analysis['statistics']['top_users']:
            html += f"""
                        <tr>
                            <td>{user['user_name']}</td>
                            <td>{user['influence_score']:.2f}</td>
                            <td>{user['followers_count']}</td>
                            <td>{user['post_count']}</td>
                            <td>{user['post_likes_received']}</td>
                        </tr>
            """
            
        html += """
                    </table>
                </div>
                
                <div class="graph-visualization">
                    <h3>Interaction Graph</h3>
                    <img src="interaction_graph.png" alt="User Interaction Graph" style="max-width: 100%;">
                    <div class="legend">
                        <p>Legend:</p>
                        <ul>
                            <li><span style="color: blue;">Blue arrows</span> - Follow relationships</li>
                            <li><span style="color: green;">Green arrows</span> - Like relationships</li>
                            <li><span style="color: red;">Red arrows</span> - Dislike relationships</li>
                        </ul>
                    </div>
                </div>
                
                <div class="metrics-visualization">
                    <h3>User Metrics Analysis</h3>
                    <img src="user_metrics.png" alt="User Metrics Analysis" style="max-width: 100%;">
                </div>
            </div>
        """
        
        return html

    def generate_html_report(self, output_path: str, results: Dict):
        """生成HTML格式的分析报告"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>数据库分析报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin-bottom: 30px; }
                h1, h2, h3 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin: 10px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f5f5f5; }
                .stats { background-color: #f9f9f9; padding: 15px; border-radius: 5px; }
                .visualization { margin: 20px 0; }
                img { max-width: 100%; height: auto; }
                .legend { margin: 10px 0; }
                .legend ul { list-style-type: none; padding: 0; }
                .legend li { margin: 5px 0; }
            </style>
        </head>
        <body>
            <h1>数据库分析报告</h1>
        """
        
        # 添加数据库结构分析
        if 'db_structure' in results:
            html += """
            <div class="section db-structure">
                <h2>数据库结构分析</h2>
                <div class="stats">
                    <p>总表数: {}</p>
                    <p>总大小: {:.2f} MB</p>
                </div>
            """.format(
                results['db_structure']['total_tables'],
                results['db_structure']['total_size'] / (1024 * 1024)
            )
            
            for table_name, table_info in results['db_structure']['tables'].items():
                html += f"""
                <h3>表: {table_name}</h3>
                <div class="stats">
                    <p>行数: {table_info['statistics']['row_count']}</p>
                    <p>大小: {table_info['statistics']['size_bytes'] / 1024:.2f} KB</p>
                </div>
                <table>
                    <tr>
                        <th>列名</th>
                        <th>类型</th>
                        <th>非空</th>
                        <th>默认值</th>
                    </tr>
                """
                
                for column in table_info['structure']['columns']:
                    html += f"""
                    <tr>
                        <td>{column['name']}</td>
                        <td>{column['type']}</td>
                        <td>{'是' if column['notnull'] else '否'}</td>
                        <td>{column['default'] if column['default'] is not None else ''}</td>
                    </tr>
                    """
                    
                html += "</table>"
                
            html += "</div>"
            
        # 添加帖子内容分析
        if 'post_analysis' in results:
            html += """
            <div class="section post-analysis">
                <h2>帖子内容分析</h2>
                <div class="stats">
                    <p>总帖子数: {}</p>
                    <p>平均点赞数: {:.2f}</p>
                    <p>平均点踩数: {:.2f}</p>
                    <p>平均分享数: {:.2f}</p>
                </div>
            """.format(
                results['post_analysis']['total_posts'],
                results['post_analysis']['avg_likes'],
                results['post_analysis']['avg_dislikes'],
                results['post_analysis']['avg_shares']
            )
            
            if 'word_freq' in results['post_analysis']:
                html += """
                <div class="visualization">
                    <h3>词频分析</h3>
                    <img src="post_wordcloud.png" alt="帖子词云">
                </div>
                """
                
            html += "</div>"
            
        # 添加用户行为分析
        if 'trace_analysis' in results:
            html += """
            <div class="section trace-analysis">
                <h2>用户行为分析</h2>
                <div class="stats">
                    <p>总行为数: {}</p>
                    <p>独立用户数: {}</p>
                </div>
            """.format(
                results['trace_analysis']['total_traces'],
                results['trace_analysis']['unique_users']
            )
            
            html += """
            <h3>行为类型分布</h3>
            <table>
                <tr>
                    <th>行为类型</th>
                    <th>次数</th>
                </tr>
            """
            
            for action, count in results['trace_analysis']['action_types'].items():
                html += f"""
                <tr>
                    <td>{action}</td>
                    <td>{count}</td>
                </tr>
                """
                
            html += "</table></div>"
            
        # 添加有向图分析
        if 'directed_graph' in results:
            html += self._generate_directed_graph_html(results['directed_graph'])
            
        html += """
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

def analyze_db_file(db_path: str, config: AnalysisConfig) -> Dict:
    """
    分析数据库文件
    
    Args:
        db_path: 数据库文件路径
        config: 分析配置
        
    Returns:
        Dict: 分析结果
    """
    if not os.path.exists(db_path):
        return {'error': '数据库文件不存在'}
        
    analyzer = DBAnalyzer(db_path)
    analyzer.output_dir = config.output_dir
    
    try:
        results = {}
        
        if config.analyze_db_structure:
            results['db_structure'] = analyzer.analyze_db_structure()
            
        if config.analyze_posts:
            results['post_analysis'] = analyzer.analyze_posts()
            
        if config.analyze_traces:
            results['trace_analysis'] = analyzer.analyze_traces()
            
        if config.analyze_directed_graph:
            results['directed_graph'] = analyzer.analyze_directed_graph()
            
        if config.output_dir:
            os.makedirs(config.output_dir, exist_ok=True)
            analyzer.generate_html_report(
                os.path.join(config.output_dir, 'db_analysis_report.html'),
                results
            )
            
        return results
    finally:
        analyzer.close()

if __name__ == "__main__":
    # 使用示例
    db_path = "data/simu_db/twitter_simulation_done1.db"
    config = AnalysisConfig(
        analyze_db_structure=True,
        analyze_posts=True,
        analyze_traces=True,
        analyze_directed_graph=True,  # 启用有向图分析
        output_dir="data/db_analysis"
    )
    analysis_result = analyze_db_file(db_path, config)
    print("分析完成！请查看输出目录中的可视化结果。")
