import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
# 设置matplotlib支持中文字体显示
import matplotlib.font_manager as fm
import os

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

# 科研绘图配色方案
scientific_colors = {
    'background': '#f8f9fa',
    'nodes': '#2c3e50',
    'edges': '#34495e',
    'highlight': '#e74c3c',
    'secondary': '#3498db',
    'tertiary': '#27ae60',
    'accent': '#f39c12'
}

# 创建自定义颜色映射
node_cmap = LinearSegmentedColormap.from_list('scientific_nodes', 
                                             ['#ecf0f1', '#2c3e50'], N=256)

# 创建一个有向图
G = nx.DiGraph()

# 添加节点（代表不同步骤）
steps = ['用户代理', '异步处理单元', '聚合分析模块']
G.add_nodes_from(steps)

# 添加边（表示步骤之间的流程）
edges = [('用户代理', '异步处理单元'), ('异步处理单元', '聚合分析模块')]
G.add_edges_from(edges)

# 设置节点位置（更合理的布局）
pos = {
    '用户代理': (0, 0),
    '异步处理单元': (2, 0),
    '聚合分析模块': (4, 0)
}

# 创建图形
plt.figure(figsize=(12, 6), facecolor=scientific_colors['background'])

# 使用FontProperties确保中文字体正确显示
try:
    font_prop = fm.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    
    # 绘制边（使用科研配色）
    nx.draw_networkx_edges(G, pos, 
                          edge_color=scientific_colors['edges'],
                          width=3,
                          alpha=0.7,
                          arrows=True,
                          arrowsize=20,
                          arrowstyle='->',
                          connectionstyle='arc3,rad=0.1',
                          min_source_margin=25,
                          min_target_margin=25)
    
    # 绘制节点（使用渐变色彩）
    node_colors = [0.2, 0.5, 0.8]  # 不同深浅的节点颜色
    nodes = nx.draw_networkx_nodes(G, pos,
                                  node_color=node_colors,
                                  node_size=3000,
                                  cmap=node_cmap,
                                  alpha=0.9,
                                  edgecolors=scientific_colors['nodes'],
                                  linewidths=2)
    
    # 添加节点标签
    nx.draw_networkx_labels(G, pos, 
                           font_size=14, 
                           font_color='white',
                           font_weight='bold',
                           font_family=font_prop.get_name())
    
    # 标题
    plt.title("异步识别流程图", fontsize=18, fontweight='bold', 
             color=scientific_colors['nodes'], fontproperties=font_prop, pad=20)
    
except Exception as e:
    print(f"使用FontProperties失败: {e}")
    # 回退到默认设置
    nx.draw_networkx_edges(G, pos, 
                          edge_color=scientific_colors['edges'],
                          width=3,
                          alpha=0.7,
                          arrows=True,
                          arrowsize=20,
                          arrowstyle='->',
                          connectionstyle='arc3,rad=0.1')
    
    nodes = nx.draw_networkx_nodes(G, pos,
                                  node_color=node_colors,
                                  node_size=3000,
                                  cmap=node_cmap,
                                  alpha=0.9,
                                  edgecolors=scientific_colors['nodes'],
                                  linewidths=2)
    
    nx.draw_networkx_labels(G, pos, 
                           font_size=14, 
                           font_color='white',
                           font_weight='bold')
    
    plt.title("Asynchronous Recognition Flow", fontsize=18, fontweight='bold', 
             color=scientific_colors['nodes'], pad=20)

# 添加流程说明
try:
    description_text = """
流程说明:
 用户代理: 接收用户输入并预处理
 异步处理单元: 并行处理多个识别任务
 聚合分析模块: 整合结果并输出最终分析
    """
    
    plt.figtext(0.02, 0.02, description_text, fontsize=10, 
               color=scientific_colors['nodes'], 
               bbox=dict(boxstyle="round,pad=0.5", 
                       facecolor=scientific_colors['background'], 
                       edgecolor=scientific_colors['secondary'], 
                       alpha=0.9),
               fontproperties=font_prop)
except:
    description_text = """
Process Description:
 User Agent: Receives user input and preprocessing
 Async Processing Unit: Parallel processing of recognition tasks
 Aggregation Analysis Module: Integrates results and outputs final analysis
    """
    
    plt.figtext(0.02, 0.02, description_text, fontsize=10, 
               color=scientific_colors['nodes'], 
               bbox=dict(boxstyle="round,pad=0.5", 
                       facecolor=scientific_colors['background'], 
                       edgecolor=scientific_colors['secondary'], 
                       alpha=0.9))

plt.axis('off')  # 关闭坐标轴

# 调整布局并保存
plt.tight_layout()
plt.savefig("./visualization/temp_ppt1.png", dpi=300, bbox_inches='tight', 
           facecolor=scientific_colors['background'])
plt.close()
print("科研风格图片已保存到: ./visualization/temp_ppt1.png")
