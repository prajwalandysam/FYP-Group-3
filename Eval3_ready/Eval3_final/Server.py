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
        self.nodes = [{"model": {"coef_": self.global_model.coef_.tolist(), 
                                 "intercept_": self.global_model.intercept_}, 
                       "metadata": {"is_global": True},
                       "client_id": None}]  # No client ID for the global model

    def add_node(self, model_update, metadata, client):
        """
        Add a new node to the DAG or update an existing one.
        :param model_update: Dictionary containing 'coef_' and 'intercept_' of a model
        :param metadata: Dictionary with additional node information
        :param client: Client object containing ID, lat, long, etc.
        :return: Index of the added/updated node
        """
        # Check if a node with the same client_id already exists
        existing_node_index = None
        for index, node in enumerate(self.nodes):
            if node.get("client_id") == client.client_id:
                existing_node_index = index
                break

        if existing_node_index is not None:
            # Update the existing node with the latest data
            self.nodes[existing_node_index]["lat"] = client.latitude
            self.nodes[existing_node_index]["long"] = client.longitude
            self.nodes[existing_node_index]["model"] = model_update
            self.nodes[existing_node_index]["metadata"] = metadata
            print(f"Updated node for client {client.client_id}")
            node_index = existing_node_index
        else:
            # Add a new node
            self.nodes.append({
                "model": model_update,
                "metadata": metadata,
                "client_id": client.client_id,
                "lat": client.latitude,
                "long": client.longitude
            })
            node_index = len(self.nodes) - 1

            # Expand adjacency matrix for the new node
            for row in self.adj_matrix:
                row.append(0)  # Add a new column to each row
            self.adj_matrix.append([0] * len(self.nodes))  # Add a new row

        # Connect the new/updated node to the global model (index 0)
        self.adj_matrix[0][node_index] = 1

        # Establish edges with relevant leaf nodes based on geographical proximity
        for i, node in enumerate(self.nodes):
            if i == node_index:
                continue  # Skip self

            lat1, lon1 = client.latitude, client.longitude
            lat2, lon2 = node.get("lat"), node.get("long")

            if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
                continue  # Skip if any node lacks location data

            # Calculate geographical proximity
            distance = haversine(lat1, lon1, lat2, lon2)
            print(f"Distance between {client.client_id} and {node['client_id']} is {distance:.2f} km")

            if distance <= 50:  # Check threshold
                self.update_edge(node_index, i)
                self.update_edge(i, node_index)

        print("Adjacency Matrix after adding/updating node:")
        self.print_dag()

        return node_index



    def delete_node(self, node_index):
        """
        Delete a node from the DAG.
        :param node_index: Index of the node to delete
        """
        print("printing the DAG before delete_node fn")
        self.print_dag()
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
        print("printing the DAG before update_edge fn")
        self.print_dag()
        self.adj_matrix[parent_index][child_index] = weight

    def aggregate(self):
        """
        Aggregate models from all leaf nodes directly connected to the global model (node 0).
        :return: The updated global model coefficients and intercept
        """

        print("Server.py : aggregating fn called")
        # Find nodes connected to node 0
        print(len(self.nodes))
        print("printing the DAG before aggregate fn")
        self.print_dag()
        connected_indices = [i for i in range(1, len(self.nodes)) if self.adj_matrix[0][i] == 1]
        print("connected_indices: "+ str(connected_indices))
        
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

        self.print_dag()
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

    def to_dict(self):
        """
        Serialize the server state for debugging or client communication.
        This method ensures all data is JSON serializable.
        """
        try:
            # Serialize the global model, adjacency matrix, and nodes
            return {
                "global_model": {
                    "coef_": self.global_model.coef_.tolist() if hasattr(self.global_model.coef_, "tolist") else self.global_model.coef_,
                    "intercept_": self.global_model.intercept_,
                },
                "adj_matrix": self.adj_matrix,  # Adjacency matrix is already JSON-compatible
                "nodes": [
                    {
                        "model": {
                            "coef_": node["model"]["coef_"] if isinstance(node["model"]["coef_"], list) else list(node["model"]["coef_"]),
                            "intercept_": node["model"]["intercept_"],
                        },
                        "metadata": node.get("metadata", {}),
                        "client_id": node.get("client_id"),
                        "lat": node.get("lat"),
                        "long": node.get("long"),
                    }
                    for node in self.nodes
                ],
            }
        except Exception as e:
            print(f"Error serializing server state: {e}")
            raise