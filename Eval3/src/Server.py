import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class Server:
    def __init__(self):
        """
        Initialize the server with a global linear regression model.
        The global model starts with zero coefficients and intercept.
        """
        # Create the global LinearRegression model
        self.global_model = LinearRegression()
        self.global_model.coef_ = np.zeros(16)  # 16 features initialized to 0
        self.global_model.intercept_ = 0.0  # Intercept initialized to 0.0
        
        # Initialize the DAG
        self.adj_matrix = [[0]]  # Start with only the global model (node 0)
        self.nodes = [{"model": {"coef_": self.global_model.coef_, 
                                 "intercept_": self.global_model.intercept_}, 
                       "metadata": {"is_global": True},
                       "client_id": None}]  # No client ID for the global model

    def add_node(self, model_update, metadata, client_id):
        """
        Add a new node to the DAG.
        :param model_update: Dictionary containing 'coef_' and 'intercept_' of a model
        :param metadata: Dictionary with additional node information
        :param client_id: ID or name of the client
        :return: Index of the added node
        """
        node_index = len(self.nodes)
        self.nodes.append({"model": model_update, 
                           "metadata": metadata, 
                           "client_id": client_id})
        
        # Expand adjacency matrix for the new node
        for row in self.adj_matrix:
            row.append(0)  # Add new column
        self.adj_matrix.append([0] * (node_index + 1))  # Add new row
        
        # Optionally connect the new node to the global model (index 0)
        self.adj_matrix[0][node_index] = 1
        return node_index

    def delete_node(self, node_index):
        """
        Delete a node from the DAG.
        :param node_index: Index of the node to delete
        """
        if node_index == 0:
            raise ValueError("Cannot delete the global model node.")
        
        # Remove the node from the adjacency matrix and the node list
        self.adj_matrix.pop(node_index)  # Remove row
        for row in self.adj_matrix:
            row.pop(node_index)  # Remove column
        self.nodes.pop(node_index)

    def update_edge(self, parent_index, child_index, weight=1):
        """
        Update an edge in the DAG.
        :param parent_index: Index of the parent node
        :param child_index: Index of the child node
        :param weight: Weight of the edge
        """
        self.adj_matrix[parent_index][child_index] = weight

    def aggregate(self):
        """
        Aggregate models from all leaf nodes into the global model.
        :return: The updated global model coefficients and intercept
        """
        leaf_indices = [i for i in range(1, len(self.nodes)) 
                        if sum(self.adj_matrix[i]) == 0]  # Nodes with no children
        aggregated_coef = np.zeros_like(self.global_model.coef_)
        aggregated_intercept = 0.0
        total_weight = 0
        
        for idx in leaf_indices:
            node = self.nodes[idx]
            model = node["model"]
            weight = node["metadata"].get("weight", 1)
            
            aggregated_coef += model["coef_"] * weight
            aggregated_intercept += model["intercept_"] * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            aggregated_coef /= total_weight
            aggregated_intercept /= total_weight
        
        # Update the global model
        self.global_model.coef_ = aggregated_coef
        self.global_model.intercept_ = aggregated_intercept
        
        # Update the global model in the nodes list
        self.nodes[0]["model"] = {"coef_": aggregated_coef, "intercept_": aggregated_intercept}
        return {"coef_": aggregated_coef, "intercept_": aggregated_intercept}

    def print_dag(self):
        """
        Print the adjacency matrix and nodes for debugging.
        """
        print("Adjacency Matrix:")
        for row in self.adj_matrix:
            print(row)
        print("\nNodes:")
        for i, node in enumerate(self.nodes):
            print(f"Node {i}: {node}")
