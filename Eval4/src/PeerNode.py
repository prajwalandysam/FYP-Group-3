from math import radians, cos, sin, sqrt, atan2
import numpy as np
from datetime import datetime

class PeerNode:
    def _init_(self, node_id, latitude=None, longitude=None):
        """
        Initialize a peer node with its own local DAG and model.
        :param node_id: Unique identifier for the peer node.
        :param latitude: Latitude of the node (for proximity scoring).
        :param longitude: Longitude of the node (for proximity scoring).
        """
        self.node_id = node_id
        self.latitude = latitude
        self.longitude = longitude
        
        # Initialize the DAG
        self.adj_matrix = [[0]]  # Global model is the first node (index 0)
        self.nodes = [{
            "node_id": 0,  # Global model node
            "model": {"coef_": np.zeros(16).tolist(), "intercept_": 0.0},
            "metadata": {"is_global": True, "timestamp": datetime.now()},
        }]
        
    def calculate_score(self, current_node, candidate_node, criteria_weights):
        """
        Calculate a score based on multi-criteria (recency, similarity, etc.).
        :param current_node: Node where the score is being calculated.
        :param candidate_node: Candidate node to consider as a parent.
        :param criteria_weights: Dictionary with weights for recency and similarity.
        :return: The computed score.
        """
        # Extract criteria weights
        recency_weight = criteria_weights.get("recency", 0.5)
        similarity_weight = criteria_weights.get("similarity", 0.5)

        # Recency: Difference in timestamps (smaller is better)
        recency_diff = (current_node["metadata"]["timestamp"] - candidate_node["metadata"]["timestamp"]).total_seconds()
        recency_score = max(0, 1 / (1 + recency_diff))

        # Similarity: Euclidean distance between model coefficients (smaller is better)
        current_coef = np.array(current_node["model"]["coef_"])
        candidate_coef = np.array(candidate_node["model"]["coef_"])
        similarity = np.linalg.norm(current_coef - candidate_coef)
        similarity_score = max(0, 1 / (1 + similarity))

        # Combine criteria scores
        return recency_weight * recency_score + similarity_weight * similarity_score

    def add_node(self, model_update, metadata, criteria_weights):
        """
        Add a new node to the local DAG based on multi-criteria dependencies.
        :param model_update: Dictionary containing 'coef_' and 'intercept_' of the model.
        :param metadata: Metadata for the new node (e.g., timestamp, client details).
        :param criteria_weights: Weights for multi-criteria scoring (recency, similarity).
        """
        # Add the new node
        new_node = {
            "node_id": len(self.nodes),
            "model": model_update,
            "metadata": metadata,
        }
        self.nodes.append(new_node)

        # Expand adjacency matrix for the new node
        for row in self.adj_matrix:
            row.append(0)  # Add a new column
        self.adj_matrix.append([0] * len(self.nodes))  # Add a new row

        # Find parent nodes based on multi-criteria scoring
        parent_scores = []
        for i, candidate_node in enumerate(self.nodes[:-1]):  # Exclude the new node itself
            score = self.calculate_score(new_node, candidate_node, criteria_weights)
            parent_scores.append((i, score))

        # Sort by score and select top parents
        parent_scores = sorted(parent_scores, key=lambda x: x[1], reverse=True)
        top_parents = [index for index, _ in parent_scores[:2]]  # Select top 2 parents

        # Add edges to the DAG
        new_node_index = len(self.nodes) - 1
        for parent_index in top_parents:
            self.adj_matrix[parent_index][new_node_index] = 1  # Parent -> Child