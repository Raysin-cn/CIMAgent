# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from __future__ import annotations

from typing import Any, Literal

import igraph as ig
from neo4j import GraphDatabase
import scipy.sparse as sp
import numpy as np

# from oasis.social_agent.agent import SocialAgent
from oasis.social_platform.config import Neo4jConfig

# modified_oasis_cim
from oasis_cim.social_agent.agent import SocialAgent

# import diffusion models
from IMmodule.Diffusion import Diffusion

class Neo4jHandler:

    def __init__(self, nei4j_config: Neo4jConfig):
        self.driver = GraphDatabase.driver(
            nei4j_config.uri,
            auth=(nei4j_config.username, nei4j_config.password),
        )
        self.driver.verify_connectivity()

    def close(self):
        self.driver.close()

    def create_agent(self, agent_id: int):
        with self.driver.session() as session:
            session.write_transaction(self._create_and_return_agent, agent_id)

    def delete_agent(self, agent_id: int):
        with self.driver.session() as session:
            session.write_transaction(
                self._delete_agent_and_relationships,
                agent_id,
            )

    def get_number_of_nodes(self) -> int:
        with self.driver.session() as session:
            return session.read_transaction(self._get_number_of_nodes)

    def get_number_of_edges(self) -> int:
        with self.driver.session() as session:
            return session.read_transaction(self._get_number_of_edges)

    def add_edge(self, src_agent_id: int, dst_agent_id: int):
        with self.driver.session() as session:
            session.write_transaction(
                self._add_and_return_edge,
                src_agent_id,
                dst_agent_id,
            )

    def remove_edge(self, src_agent_id: int, dst_agent_id: int):
        with self.driver.session() as session:
            session.write_transaction(
                self._remove_and_return_edge,
                src_agent_id,
                dst_agent_id,
            )

    def get_all_nodes(self) -> list[int]:
        with self.driver.session() as session:
            return session.read_transaction(self._get_all_nodes)

    def get_all_edges(self) -> list[tuple[int, int]]:
        with self.driver.session() as session:
            return session.read_transaction(self._get_all_edges)

    def reset_graph(self):
        with self.driver.session() as session:
            session.write_transaction(self._reset_graph)

    @staticmethod
    def _create_and_return_agent(tx: Any, agent_id: int):
        query = """
        CREATE (a:Agent {id: $agent_id})
        RETURN a
        """
        result = tx.run(query, agent_id=agent_id)
        return result.single()

    @staticmethod
    def _delete_agent_and_relationships(tx: Any, agent_id: int):
        query = """
        MATCH (a:Agent {id: $agent_id})
        DETACH DELETE a
        RETURN count(a) AS deleted
        """
        result = tx.run(query, agent_id=agent_id)
        return result.single()

    @staticmethod
    def _add_and_return_edge(tx: Any, src_agent_id: int, dst_agent_id: int):
        query = """
        MATCH (a:Agent {id: $src_agent_id}), (b:Agent {id: $dst_agent_id})
        CREATE (a)-[r:FOLLOW]->(b)
        RETURN r
        """
        result = tx.run(query,
                        src_agent_id=src_agent_id,
                        dst_agent_id=dst_agent_id)
        return result.single()

    @staticmethod
    def _remove_and_return_edge(tx: Any, src_agent_id: int, dst_agent_id: int):
        query = """
        MATCH (a:Agent {id: $src_agent_id})
        MATCH (b:Agent {id: $dst_agent_id})
        MATCH (a)-[r:FOLLOW]->(b)
        DELETE r
        RETURN count(r) AS deleted
        """
        result = tx.run(query,
                        src_agent_id=src_agent_id,
                        dst_agent_id=dst_agent_id)
        return result.single()

    @staticmethod
    def _get_number_of_nodes(tx: Any) -> int:
        query = """
        MATCH (n)
        RETURN count(n) AS num_nodes
        """
        result = tx.run(query)
        return result.single()["num_nodes"]

    @staticmethod
    def _get_number_of_edges(tx: Any) -> int:
        query = """
        MATCH ()-[r]->()
        RETURN count(r) AS num_edges
        """
        result = tx.run(query)
        return result.single()["num_edges"]

    @staticmethod
    def _get_all_nodes(tx: Any) -> list[int]:
        query = """
        MATCH (a:Agent)
        RETURN a.id AS agent_id
        """
        result = tx.run(query)
        return [record["agent_id"] for record in result]

    @staticmethod
    def _get_all_edges(tx: Any) -> list[tuple[int, int]]:
        query = """
        MATCH (a:Agent)-[r:FOLLOW]->(b:Agent)
        RETURN a.id AS src_agent_id, b.id AS dst_agent_id
        """
        result = tx.run(query)
        return [(record["src_agent_id"], record["dst_agent_id"])
                for record in result]

    @staticmethod
    def _reset_graph(tx: Any):
        query = """
        MATCH (n)
        DETACH DELETE n
        """
        tx.run(query)


class AgentGraph():
    r"""AgentGraph class to manage the social graph of agents."""

    def __init__(
        self,
        backend: Literal["igraph", "neo4j"] = "igraph",
        neo4j_config: Neo4jConfig | None = None,
    ):
        self.backend = backend
        if self.backend == "igraph":
            self.graph = ig.Graph(directed=True)
        else:
            assert neo4j_config is not None
            assert neo4j_config.is_valid()
            self.graph = Neo4jHandler(neo4j_config)
        self.agent_mappings: dict[int, SocialAgent] = {}
        self.diffusion = Diffusion()

    def reset(self):
        if self.backend == "igraph":
            self.graph = ig.Graph(directed=True)
        else:
            self.graph.reset_graph()
        self.agent_mappings: dict[int, SocialAgent] = {}

    def add_agent(self, agent: SocialAgent):
        if self.backend == "igraph":
            self.graph.add_vertex(agent.social_agent_id)
        else:
            self.graph.create_agent(agent.social_agent_id)
        self.agent_mappings[agent.social_agent_id] = agent

    def add_edge(self, agent_id_0: int, agent_id_1: int):
        try:
            self.graph.add_edge(agent_id_0, agent_id_1)
        except Exception:
            pass

    def remove_agent(self, agent: SocialAgent):
        if self.backend == "igraph":
            self.graph.delete_vertices(agent.social_agent_id)
        else:
            self.graph.delete_agent(agent.social_agent_id)
        del self.agent_mappings[agent.social_agent_id]

    def remove_edge(self, agent_id_0: int, agent_id_1: int):
        if self.backend == "igraph":
            if self.graph.are_connected(agent_id_0, agent_id_1):
                self.graph.delete_edges([(agent_id_0, agent_id_1)])
        else:
            self.graph.remove_edge(agent_id_0, agent_id_1)

    def get_agent(self, agent_id: int) -> SocialAgent:
        return self.agent_mappings[agent_id]

    def get_agents(self) -> list[tuple[int, SocialAgent]]:
        if self.backend == "igraph":
            return [(node.index, self.agent_mappings[node.index])
                    for node in self.graph.vs]
        else:
            return [(agent_id, self.agent_mappings[agent_id])
                    for agent_id in self.graph.get_all_nodes()]

    def get_edges(self) -> list[tuple[int, int]]:
        if self.backend == "igraph":
            return [(edge.source, edge.target) for edge in self.graph.es]
        else:
            return self.graph.get_all_edges()

    def get_num_nodes(self) -> int:
        if self.backend == "igraph":
            return self.graph.vcount()
        else:
            return self.graph.get_number_of_nodes()

    def get_num_edges(self) -> int:
        if self.backend == "igraph":
            return self.graph.ecount()
        else:
            return self.graph.get_number_of_edges()

    def close(self) -> None:
        if self.backend == "neo4j":
            self.graph.close()

    def visualize(
        self,
        path: str,
        vertex_size: int = 20,
        edge_arrow_size: float = 0.5,
        with_labels: bool = True,
        vertex_color: str = "#f74f1b",
        vertex_frame_width: int = 2,
        width: int = 1000,
        height: int = 1000,
    ):
        if self.backend == "neo4j":
            raise ValueError("Neo4j backend does not support visualization.")
        layout = self.graph.layout("auto")
        if with_labels:
            labels = [node_id for node_id, _ in self.get_agents()]
        else:
            labels = None
        ig.plot(
            self.graph,
            target=path,
            layout=layout,
            vertex_label=labels,
            vertex_size=vertex_size,
            vertex_color=vertex_color,
            edge_arrow_size=edge_arrow_size,
            vertex_frame_width=vertex_frame_width,
            bbox=(width, height),
        )

    def get_adjacency_matrix(self) -> sp.csr_matrix:
        """
        获取图的邻接矩阵
        Get the adjacency matrix of the graph
        
        Returns:
            邻接矩阵（CSR格式） / Adjacency matrix in CSR format
        """
        if self.backend == "igraph":
            # 使用igraph的get_adjacency方法获取邻接矩阵
            adj_matrix = self.graph.get_adjacency()
            # 转换为scipy稀疏矩阵
            return sp.csr_matrix(adj_matrix)
        else:
            # 对于neo4j后端，需要手动构建邻接矩阵
            num_nodes = self.get_num_nodes()
            edges = self.get_edges()
            
            # 创建邻接矩阵
            row = []
            col = []
            data = []
            
            # 获取节点ID到索引的映射
            node_to_idx = {node_id: idx for idx, (node_id, _) in enumerate(self.get_agents())}
            
            # 填充邻接矩阵
            for src, dst in edges:
                row.append(node_to_idx[src])
                col.append(node_to_idx[dst])
                data.append(1.0)
            
            # 创建CSR格式的稀疏矩阵
            return sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

    def select_seeds(self, seed_nums_rate: float = 0.1) -> list[int]:
        """
        选择种子节点
        Select seed nodes
        
        Args:
            algos (str, optional): 选择种子节点使用的算法. 默认为 "DeepIM"
            seed_nums_rate (float, optional): 选择的种子节点数量占总节点数的比例. 默认为 0.1
            
        Returns:
            list[int]: 选中的种子节点ID列表
        """
        # 获取总节点数
        num_nodes = self.get_num_nodes()
        
        # 计算需要选择的种子节点数量
        seed_nums = max(1, int(num_nodes * seed_nums_rate))
        
        # 获取邻接矩阵
        adj_matrix = self.get_adjacency_matrix()
        
        # 使用Diffusion类选择种子节点
        if self.diffusion != "Random":
            selected_seeds = self.diffusion.select_seeds(adj_matrix, seed_nums)
        else:
            selected_seeds = np.random.choice(np.arange(0, num_nodes), size=seed_nums, replace=False).tolist()
        
        # 获取所有节点ID列表
        if self.backend == "igraph":
            node_ids = [v.index for v in self.graph.vs]
        else:
            node_ids = self.graph.get_all_nodes()
        
        # 将选中的索引转换为实际的节点ID
        selected_node_ids = [node_ids[idx] for idx in selected_seeds]
        
        return selected_node_ids

    
