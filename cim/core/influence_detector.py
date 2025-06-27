from typing import List, Tuple, Dict, TYPE_CHECKING
import igraph as ig

from oasis.social_agent.agent_graph import AgentGraph

def analyze_degree_centrality(
    agent_graph: AgentGraph,
    k: int,
    ascending: bool = False
) -> List[Tuple[int, float]]:
    """
    分析智能体关注图结构中的度中心性，并返回度中心性最大的top-k个节点。

    Args:
        agent_graph: AgentGraph对象，包含社交网络图结构。
        k: 要返回的度中心性最大的节点数量。
        ascending: 如果为True，按升序排列；如果为False（默认），按降序排列。

    Returns:
        一个列表，其中每个元素是一个元组 (agent_id, degree_centrality_value)，
        表示度中心性最大的top-k个节点及其对应的度中心性值，按指定方向排列。
    """
    if agent_graph.backend == "igraph":
        graph = agent_graph.graph
    elif agent_graph.backend == "neo4j":
        # 对于Neo4j后端，需要从AgentGraph中获取节点和边来构建igraph图
        nodes_info = agent_graph.get_agents()
        edges = agent_graph.get_edges()
        
        # 从nodes_info中提取实际的agent_id
        agent_ids_for_igraph = [agent_id for agent_id, _ in nodes_info]
        
        # 创建一个igraph图，确保所有节点都包含在图中
        graph = ig.Graph(directed=True)
        # 添加所有节点，确保igraph内部ID与agent_id对应
        graph.add_vertices(len(agent_ids_for_igraph)) 
        # 将igraph的顶点索引映射到agent_id
        agent_id_to_igraph_id = {agent_id: i for i, agent_id in enumerate(agent_ids_for_igraph)}

        # 添加边
        igraph_edges = []
        for src, dst in edges:
            if src in agent_id_to_igraph_id and dst in agent_id_to_igraph_id:
                igraph_edges.append((agent_id_to_igraph_id[src], agent_id_to_igraph_id[dst]))
            else:
                pass 
        graph.add_edges(igraph_edges)
    else:
        raise ValueError(f"不支持的AgentGraph后端: {agent_graph.backend}")
    # 计算度中心性 (in-degree + out-degree)
    # 对于有向图，igraph的degree()默认返回in-degree + out-degree
    degrees = graph.degree()

    # 将igraph的顶点索引转换回agent_id，并存储度中心性
    degree_centralities: Dict[int, float] = {}
    if agent_graph.backend == "igraph":
        # igraph backend的顶点索引就是agent_id
        for i, deg in enumerate(degrees):
            # 获取实际的agent_id
            # 这里需要注意igraph.Graph的vertex index和agent_id可能不是直接对应的
            # 假设agent_graph.get_agents()返回的顺序和igraph的vertex index顺序一致
            # 更安全的做法是直接从agent_graph获取所有agent_id
            all_agents_info = agent_graph.get_agents()
            agent_id = all_agents_info[i][0] # (agent_id, SocialAgent)
            degree_centralities[agent_id] = deg
    elif agent_graph.backend == "neo4j":
        for i, deg in enumerate(degrees):
            agent_id = agent_ids_for_igraph[i] # nodes列表的顺序和igraph的顶点索引顺序一致
            degree_centralities[agent_id] = deg

    # 按度中心性降序排序
    sorted_centralities = sorted(
        degree_centralities.items(), key=lambda item: item[1], reverse=not ascending
    )

    return sorted_centralities[:k]

def analyze_closeness_centrality(
    agent_graph: AgentGraph,
    k: int,
    ascending: bool = False
) -> List[Tuple[int, float]]:
    """
    分析智能体关注图结构中的亲近中心性，并返回亲近中心性最大的top-k个节点。

    Args:
        agent_graph: AgentGraph对象，包含社交网络图结构。
        k: 要返回的亲近中心性最大的节点数量。
        ascending: 如果为True，按升序排列；如果为False（默认），按降序排列。

    Returns:
        一个列表，其中每个元素是一个元组 (agent_id, closeness_centrality_value)，
        表示亲近中心性最大的top-k个节点及其对应的亲近中心性值，按指定方向排列。
    """
    if agent_graph.backend == "igraph":
        graph = agent_graph.graph
    elif agent_graph.backend == "neo4j":
        nodes_info = agent_graph.get_agents()
        edges = agent_graph.get_edges()

        agent_ids_for_igraph = [agent_id for agent_id, _ in nodes_info]

        graph = ig.Graph(directed=True)
        graph.add_vertices(len(agent_ids_for_igraph))
        agent_id_to_igraph_id = {agent_id: i for i, agent_id in enumerate(agent_ids_for_igraph)}

        igraph_edges = []
        for src, dst in edges:
            if src in agent_id_to_igraph_id and dst in agent_id_to_igraph_id:
                igraph_edges.append((agent_id_to_igraph_id[src], agent_id_to_igraph_id[dst]))
        graph.add_edges(igraph_edges)
    else:
        raise ValueError(f"不支持的AgentGraph后端: {agent_graph.backend}")

    # 计算亲近中心性
    # 对于有向图，igraph的closeness()默认计算出度亲近中心性。
    # 如果需要无向图的亲近中心性，可以先将图转换为无向图再计算。
    # 这里我们假设计算有向图的亲近中心性。
    closenesses = graph.closeness()

    # 将igraph的顶点索引转换回agent_id，并存储亲近中心性
    closeness_centralities: Dict[int, float] = {}
    if agent_graph.backend == "igraph":
        all_agents_info = agent_graph.get_agents()
        for i, close in enumerate(closenesses):
            agent_id = all_agents_info[i][0]
            closeness_centralities[agent_id] = close
    elif agent_graph.backend == "neo4j":
        for i, close in enumerate(closenesses):
            agent_id = agent_ids_for_igraph[i]
            closeness_centralities[agent_id] = close

    # 按亲近中心性降序排序
    sorted_centralities = sorted(
        closeness_centralities.items(), key=lambda item: item[1], reverse=not ascending
    )

    return sorted_centralities[:k]

def analyze_betweenness_centrality(
    agent_graph: AgentGraph,
    k: int,
    ascending: bool = False
) -> List[Tuple[int, float]]:
    """
    分析智能体关注图结构中的中介中心性，并返回中介中心性最大的top-k个节点。

    Args:
        agent_graph: AgentGraph对象，包含社交网络图结构。
        k: 要返回的中介中心性最大的节点数量。
        ascending: 如果为True，按升序排列；如果为False（默认），按降序排列。

    Returns:
        一个列表，其中每个元素是一个元组 (agent_id, betweenness_centrality_value)，
        表示中介中心性最大的top-k个节点及其对应的中介中心性值，按指定方向排列。
    """
    if agent_graph.backend == "igraph":
        graph = agent_graph.graph
    elif agent_graph.backend == "neo4j":
        nodes_info = agent_graph.get_agents()
        edges = agent_graph.get_edges()

        agent_ids_for_igraph = [agent_id for agent_id, _ in nodes_info]

        graph = ig.Graph(directed=True)
        graph.add_vertices(len(agent_ids_for_igraph))
        agent_id_to_igraph_id = {agent_id: i for i, agent_id in enumerate(agent_ids_for_igraph)}

        igraph_edges = []
        for src, dst in edges:
            if src in agent_id_to_igraph_id and dst in agent_id_to_igraph_id:
                igraph_edges.append((agent_id_to_igraph_id[src], agent_id_to_igraph_id[dst]))
        graph.add_edges(igraph_edges)
    else:
        raise ValueError(f"不支持的AgentGraph后端: {agent_graph.backend}")

    # 计算中介中心性
    betweennesses = graph.betweenness()

    # 将igraph的顶点索引转换回agent_id，并存储中介中心性
    betweenness_centralities: Dict[int, float] = {}
    if agent_graph.backend == "igraph":
        all_agents_info = agent_graph.get_agents()
        for i, bet in enumerate(betweennesses):
            agent_id = all_agents_info[i][0]
            betweenness_centralities[agent_id] = bet
    elif agent_graph.backend == "neo4j":
        for i, bet in enumerate(betweennesses):
            agent_id = agent_ids_for_igraph[i]
            betweenness_centralities[agent_id] = bet

    # 按中介中心性降序排序
    sorted_centralities = sorted(
        betweenness_centralities.items(), key=lambda item: item[1], reverse=not ascending
    )

    return sorted_centralities[:k]
