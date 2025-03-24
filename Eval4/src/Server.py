from datetime import datetime, timezone
import logging
import requests
import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from math import radians,cos,sin,sqrt,atan2
import networkx as nx
import matplotlib.pyplot as plt
import heapq
from collections import deque

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
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Format timestamp to milliseconds
        self.nodes = [{"model": {"coef_": self.global_model.coef_.tolist(), 
                                 "intercept_": self.global_model.intercept_}, 
                       "metadata": {"is_global": True,"timestamp":self.timestamp},
                       "client_id": None}]  # No client ID for the global model

    
    
    def sync_global_model(self, urls):
        """Synchronize global model across nodes."""
        
        latest_coef_ = np.zeros(16)  # 16 features initialized to 0
        latest_intercept_ = 0.0 # Intercept initialized to 0.0
        latest_time = 0  # Latest timestamp to track newest model

        if not hasattr(self, "global_model"):
            logging.error("No global model found.")
            return {"error": "No global model available."}

        
        
       

        # for url in server_urls:
        for url in urls:
            try:
                response = requests.get(f"{url}/get-global-parameters")
                response.raise_for_status()  # Raise exception for HTTP errors (4xx, 5xx)
                data = response.json()  # Parse response JSON
                
                if "time" in data and "coef" in data and "intercept_" in data:
                    response_time = datetime.fromisoformat(data["time"])
                    latest_time_obj = datetime.fromisoformat(latest_time) if latest_time else None
                    
                    if latest_time_obj is None or response_time > latest_time_obj:
                        latest_time = data["time"]
                        latest_coef_ = np.array(data["coef"])
                        latest_intercept_ = data["intercept_"]
                else:
                    logging.warning(f"Invalid response format from {url}: {data}")
            
            except requests.exceptions.RequestException as e:
                logging.error(f"Failed to sync with {url}: {e}")
            except ValueError as e:
                logging.error(f"Invalid JSON response from {url}: {e}")

        # Update the global model with the latest parameters
        self.global_model.coef_ = latest_coef_
        self.global_model.intercept_ = latest_intercept_
        # Update the global model in the nodes list
        self.nodes[0]["model"] = {"coef_": latest_coef_, "intercept_": latest_intercept_}

        logging.info(f"Global model updated with latest parameters at {latest_time}")
        self.print_dag()

        return {"coef_": latest_coef_.tolist(), "intercept_": latest_intercept_, "latest_time": latest_time}

    
    
    
    
    def add_node(self, model_update, metadata, client, num_parents=2):
        """
        Add a new node to the DAG and randomly assign parent nodes.
        
        :param model_update: Dictionary containing 'coef_' and 'intercept_' of the model.
        :param metadata: Metadata for the new node (e.g., timestamp, client details).
        :param client: Client object containing client_id, latitude, longitude.
        :param num_parents: Number of random parent nodes to assign.
        :return: Index of the added/updated node.
        """
        

        # Add the new node
        new_node = {
            "model": model_update,
            "metadata": metadata,
            "client_id": client.client_id,
            "lat": client.latitude,
            "long": client.longitude
        }
        self.nodes.append(new_node)
        new_node_index = len(self.nodes) - 1

        # Expand adjacency matrix for the new node
        for row in self.adj_matrix:
            row.append(0)  # Add a new column
        self.adj_matrix.append([0] * len(self.nodes))  # Add a new row

        # Randomly pick parent nodes
        if len(self.nodes) > 1:
            parent_indices = random.sample(range(len(self.nodes) - 1), min(num_parents, len(self.nodes) - 1))
        else:
            parent_indices = []

        print(f"Randomly selected parents for node {new_node_index}: {parent_indices}")

        # Add edges to the DAG
        for parent_index in parent_indices:
            self.adj_matrix[parent_index][new_node_index] = 1  # Parent -> Child

        print("Adjacency Matrix after adding/updating node:")
        self.print_dag()

        return new_node_index
    



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
    def plot_graph(self, removed_nodes=None, title="Graph Visualization"):
        G = nx.DiGraph()
        current_time = datetime.utcnow()
        node_ages = {i: (current_time - datetime.strptime(self.nodes[i]['metadata']['timestamp'], "%Y-%m-%d %H:%M:%S.%f")).total_seconds()
                     for i in range(len(self.nodes))}
        
        for i in range(len(self.nodes)):
            color = 'red' if removed_nodes and i in removed_nodes else 'green'
            G.add_node(i, age=node_ages[i], color=color)
        
        for i in range(len(self.adj_matrix)):
            for j in range(len(self.adj_matrix[i])):
                if self.adj_matrix[i][j] == 1:
                    G.add_edge(i, j)
        
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)
        colors = [G.nodes[node]['color'] for node in G.nodes]
        
        nx.draw(G, pos, with_labels=True, node_color=colors, node_size=700, edge_color='gray', font_size=10)
        plt.title(title)
        plt.show()

    def prune_graph(self, prune_count=2):
        print("DEBUG: Pruning Graph")
        current_time = datetime.utcnow()
        
        # BFS Traversal to collect all nodes
        queue = deque([0])  # Start BFS from the global node (Node 0)
        visited = set([0])
        node_heap = []  # Min heap for pruning candidates
        
        while queue:
            node_idx = queue.popleft()
            if node_idx != 0:  # Skip global node
                timestamp = datetime.strptime(self.nodes[node_idx]['metadata']['timestamp'], "%Y-%m-%d %H:%M:%S.%f")
                heapq.heappush(node_heap, (timestamp, node_idx))
            
            for child_idx in range(len(self.adj_matrix[node_idx])):
                if self.adj_matrix[node_idx][child_idx] == 1 and child_idx not in visited:
                    visited.add(child_idx)
                    queue.append(child_idx)
        
        # Remove the two oldest nodes (skip if fewer available)
        nodes_to_remove = []
        while len(nodes_to_remove) < prune_count and node_heap:
            _, node_idx = heapq.heappop(node_heap)
            nodes_to_remove.append(node_idx)
        
        if not nodes_to_remove:
            print("DEBUG: No nodes to prune.")
            return []
        
        print(f"DEBUG: Pruning nodes {nodes_to_remove}")
        
        # Reconnect children of removed nodes to random remaining nodes
        remaining_nodes = [i for i in range(len(self.nodes)) if i not in nodes_to_remove]
        
        for node_idx in nodes_to_remove:
            children = [i for i in range(len(self.adj_matrix[node_idx])) if self.adj_matrix[node_idx][i] == 1]
            if not remaining_nodes:
                break  # No remaining nodes to reconnect to
            
            for child in children:
                new_parent = random.choice(remaining_nodes)  # Random reassignment
                self.adj_matrix[new_parent][child] = 1  # Connect new parent to child
        
        # Create new adjacency matrix and node list
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(remaining_nodes)}
        new_adj_matrix = [[self.adj_matrix[i][j] for j in remaining_nodes] for i in remaining_nodes]
        self.nodes = [self.nodes[i] for i in remaining_nodes]
        self.adj_matrix = new_adj_matrix
        
        print("DEBUG: DAG after pruning")
        self.print_dag()
        
        
        
        return nodes_to_remove

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
            
            
    # def plot_graph(adj_matrix, node_list, timestamps, removed_nodes=None, title="Graph Visualization"):
    #     G = nx.DiGraph()
        
    #     # Convert timestamps to ages
    #     current_time = datetime.utcnow()
    #     node_ages = {node: (current_time - datetime.strptime(timestamps[node], "%Y-%m-%d %H:%M:%S.%f")).total_seconds()
    #                 for node in node_list}
        
    #     # Add nodes with attributes
    #     for i, node in enumerate(node_list):
    #         color = 'red' if removed_nodes and node in removed_nodes else 'green'
    #         G.add_node(node, age=node_ages[node], color=color)
        
    #     # Add edges based on adjacency matrix
    #     for i in range(len(adj_matrix)):
    #         for j in range(len(adj_matrix[i])):
    #             if adj_matrix[i][j] == 1:
    #                 G.add_edge(node_list[i], node_list[j])
        
    #     # Draw graph
    #     plt.figure(figsize=(8, 6))
    #     pos = nx.spring_layout(G)
    #     colors = [G.nodes[node]['color'] for node in G.nodes]
        
    #     nx.draw(G, pos, with_labels=True, node_color=colors, node_size=700, edge_color='gray', font_size=10)
    #     plt.title(title)
    #     plt.show()
    
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