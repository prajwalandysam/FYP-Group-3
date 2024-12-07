import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from math import radians,cos,sin,sqrt,atan2

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the Earth."""
    R = 6371  # Earth's radius in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

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

    def add_node(self, model_update, metadata, client):
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
                           "client_id": client.client_id,
                           "lat":client.latitude,
                           "long":client.longitude})
        
        # Expand adjacency matrix for the new node
        for row in self.adj_matrix:
            row.append(0)  # Add new column
        self.adj_matrix.append([0] * (node_index + 1))  # Add new row
        
        # Optionally connect the new node to the global model (index 0)
        self.adj_matrix[0][node_index] = 1

        # Establish edges with relevant leaf nodes based on geographical proximity
        for i, node in enumerate(self.nodes):
           
            # Get latitude and longitude from metadata
            lat1, lon1 = client.latitude, client.longitude
            lat2, lon2 = node.get("lat"), node.get("long")

            if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
                continue  # Skip if any node lacks location data

            # Calculate geographical proximity
            distance = haversine(lat1, lon1, lat2, lon2)
            print(client.client_id, node["client_id"])

            # Define a distance threshold for connection (e.g., 50 km)
            if distance <= 50:  # Example threshold
                self.update_edge(node_index, i)
                self.update_edge(i, node_index)

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
        Aggregate models from all leaf nodes directly connected to the global model (node 0).
        :return: The updated global model coefficients and intercept
        """
        # Find nodes connected to node 0
        connected_indices = [i for i in range(1, len(self.nodes)) if self.adj_matrix[0][i] == 1]
        
        aggregated_coef = np.zeros_like(self.global_model.coef_)
        aggregated_intercept = 0.0
        total_weight = 0
        
        for idx in connected_indices:
            node = self.nodes[idx]
            model = node["model"]
            weight = node["metadata"].get("weight", 1)  # Default weight is 1 if not provided
            
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
