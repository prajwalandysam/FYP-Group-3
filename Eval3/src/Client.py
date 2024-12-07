import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression



class Client:
    def __init__(self, client_id, local_data, local_labels):
        """
        Initialize a client with local data and labels.
        :param client_id: Unique identifier for the client
        :param local_data: Local training data (numpy array or similar)
        :param local_labels: Local labels (numpy array or similar)
        """
        self.client_id = client_id
        self.local_data = local_data
        self.local_labels = local_labels
        self.local_model = None  # Placeholder for the local LinearRegression model
        self.server_node_index = None  # Keeps track of the node in the server's DAG

    def get_global_model(self, server, full_graph=False):
        """
        Retrieve the global model or the full DAG from the server.
        :param server: The Server object to fetch the model from
        :param full_graph: If True, return the entire DAG. Otherwise, return the global model.
        :return: Global model (dict) or full DAG (list of nodes)
        """
        if full_graph:
            return {"adjacency_matrix": server.adj_matrix, "nodes": server.nodes}
        else:
            return server.nodes[0]["model"]  # Return the global model (Node 0)

    def train_local_model(self, global_model):
        """
        Train a local linear regression model using the global model as the starting point.
        :param global_model: Global model (dict with 'coef_' and 'intercept_')
        :return: Updated local model coefficients and intercept
        """
        # Initialize local model with global coefficients and intercept
        self.local_model = LinearRegression()
        self.local_model.coef_ = global_model["coef_"].copy()
        self.local_model.intercept_ = global_model["intercept_"]

        # Train the model on the client's local data
        self.local_model.fit(self.local_data, self.local_labels)

        # Return updated coefficients and intercept
        return {"coef_": self.local_model.coef_, "intercept_": self.local_model.intercept_}

    def send_update_to_server(self, server, metadata):
        """
        Send the local model's updates to the server.
        If the client has already submitted an update, overwrite the existing node in the server's DAG.
        :param server: The Server object to send updates to
        :param metadata: Metadata for the update (e.g., weight, timestamp)
        """
        local_update = {"coef_": self.local_model.coef_, "intercept_": self.local_model.intercept_}

        if self.server_node_index is None:
            # First time: Add a new node with the client ID
            self.server_node_index = server.add_node(local_update, metadata, client_id=self.client_id)
        else:
            # Overwrite the existing node, retaining the client ID
            server.nodes[self.server_node_index] = {
                "model": local_update,
                "metadata": metadata,
                "client_id": self.client_id
            }
