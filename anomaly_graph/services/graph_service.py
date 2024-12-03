from neo4j import GraphDatabase
import logging
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyGraphService:
    def __init__(self, uri: str, user: str, password: str):
        """Initialize the Neo4j connection."""
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()

    def create_node(self, node_id: str, metrics: Dict[str, float], anomaly_prob: float):
        """Create or update a database node with its metrics and anomaly probability."""
        with self.driver.session() as session:
            try:
                session.write_transaction(self._create_node_tx, node_id, metrics, anomaly_prob)
                logger.info(f"Successfully created/updated node: {node_id}")
            except Exception as e:
                logger.error(f"Failed to create/update node {node_id}: {str(e)}")
                raise

    @staticmethod
    def _create_node_tx(tx, node_id: str, metrics: Dict[str, float], anomaly_prob: float):
        """Transaction function to create/update a node."""
        query = """
        MERGE (n:DatabaseNode {id: $node_id})
        SET n += $metrics,
            n.anomaly_probability = $anomaly_prob,
            n.last_updated = datetime()
        """
        tx.run(query, node_id=node_id, metrics=metrics, anomaly_prob=anomaly_prob)

    def create_edge(self, source_id: str, target_id: str, weight: float):
        """Create or update an edge between two nodes."""
        with self.driver.session() as session:
            try:
                session.write_transaction(self._create_edge_tx, source_id, target_id, weight)
                logger.info(f"Successfully created/updated edge: {source_id} -> {target_id}")
            except Exception as e:
                logger.error(f"Failed to create/update edge {source_id}->{target_id}: {str(e)}")
                raise

    @staticmethod
    def _create_edge_tx(tx, source_id: str, target_id: str, weight: float):
        """Transaction function to create/update an edge."""
        query = """
        MATCH (a:DatabaseNode {id: $source_id})
        MATCH (b:DatabaseNode {id: $target_id})
        MERGE (a)-[r:CONNECTED]->(b)
        SET r.weight = $weight,
            r.last_updated = datetime()
        """
        tx.run(query, source_id=source_id, target_id=target_id, weight=weight)

    def compute_pagerank(self) -> List[Tuple[str, float]]:
        """Compute PageRank scores for all nodes."""
        with self.driver.session() as session:
            try:
                result = session.read_transaction(self._compute_pagerank_tx)
                logger.info("Successfully computed PageRank scores")
                return result
            except Exception as e:
                logger.error(f"Failed to compute PageRank: {str(e)}")
                raise

    @staticmethod
    def _compute_pagerank_tx(tx) -> List[Tuple[str, float]]:
        """Transaction function to compute PageRank."""
        query = """
        CALL gds.pageRank.stream({
            nodeProjection: 'DatabaseNode',
            relationshipProjection: {
                CONNECTED: {
                    type: 'CONNECTED',
                    properties: 'weight',
                    orientation: 'UNDIRECTED'
                }
            },
            maxIterations: 20,
            dampingFactor: 0.85
        })
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).id AS node, score
        ORDER BY score DESC
        """
        return [(record["node"], record["score"]) for record in tx.run(query)]

    def get_root_causes(self, top_k: int = 5) -> List[Dict[str, float]]:
        """Get top-k root causes based on composite scores."""
        with self.driver.session() as session:
            try:
                pagerank_scores = dict(self.compute_pagerank())
                result = session.read_transaction(
                    self._get_root_causes_tx, pagerank_scores, top_k
                )
                logger.info(f"Successfully identified top {top_k} root causes")
                return result
            except Exception as e:
                logger.error(f"Failed to get root causes: {str(e)}")
                raise

    @staticmethod
    def _get_root_causes_tx(tx, pagerank_scores: Dict[str, float], top_k: int) -> List[Dict[str, float]]:
        """Transaction function to get root causes."""
        query = """
        MATCH (n:DatabaseNode)
        RETURN n.id AS node, n.anomaly_probability AS anomaly_prob
        """
        results = []
        for record in tx.run(query):
            node = record["node"]
            anomaly_prob = record["anomaly_prob"]
            influence = pagerank_scores.get(node, 0)
            composite_score = anomaly_prob * influence
            results.append({
                "node": node,
                "anomaly_probability": anomaly_prob,
                "influence_score": influence,
                "composite_score": composite_score
            })
        
        # Sort by composite score and return top-k
        results.sort(key=lambda x: x["composite_score"], reverse=True)
        return results[:top_k]

    def get_node_neighbors(self, node_id: str) -> List[Dict]:
        """Get all neighbors of a given node with their relationships."""
        with self.driver.session() as session:
            try:
                result = session.read_transaction(self._get_node_neighbors_tx, node_id)
                logger.info(f"Successfully retrieved neighbors for node: {node_id}")
                return result
            except Exception as e:
                logger.error(f"Failed to get neighbors for node {node_id}: {str(e)}")
                raise

    @staticmethod
    def _get_node_neighbors_tx(tx, node_id: str) -> List[Dict]:
        """Transaction function to get node neighbors."""
        query = """
        MATCH (n:DatabaseNode {id: $node_id})-[r:CONNECTED]-(neighbor:DatabaseNode)
        RETURN neighbor.id AS neighbor_id,
               neighbor.anomaly_probability AS neighbor_anomaly_prob,
               r.weight AS relationship_weight
        """
        return [dict(record) for record in tx.run(query, node_id=node_id)]

    def get_subgraph(self, node_ids: List[str]) -> Dict:
        """Get a subgraph containing specified nodes and their relationships."""
        with self.driver.session() as session:
            try:
                result = session.read_transaction(self._get_subgraph_tx, node_ids)
                logger.info(f"Successfully retrieved subgraph for nodes: {node_ids}")
                return result
            except Exception as e:
                logger.error(f"Failed to get subgraph: {str(e)}")
                raise

    @staticmethod
    def _get_subgraph_tx(tx, node_ids: List[str]) -> Dict:
        """Transaction function to get subgraph."""
        nodes_query = """
        MATCH (n:DatabaseNode)
        WHERE n.id IN $node_ids
        RETURN n.id AS id,
               n.anomaly_probability AS anomaly_prob,
               n.cpu_usage AS cpu,
               n.memory_usage AS memory,
               n.io_usage AS io
        """
        
        edges_query = """
        MATCH (n:DatabaseNode)-[r:CONNECTED]-(m:DatabaseNode)
        WHERE n.id IN $node_ids AND m.id IN $node_ids
        RETURN n.id AS source,
               m.id AS target,
               r.weight AS weight
        """
        
        nodes = [dict(record) for record in tx.run(nodes_query, node_ids=node_ids)]
        edges = [dict(record) for record in tx.run(edges_query, node_ids=node_ids)]
        
        return {
            "nodes": nodes,
            "edges": edges
        } 