import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

# Load the Diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
Y = diabetes.target

# Create a DataFrame with feature names
data = pd.DataFrame(X, columns=diabetes.feature_names)
data["DiseaseProgression"] = Y

# Splitting the data into training and testing sets
X = data.drop("DiseaseProgression", axis=1)
Y = data["DiseaseProgression"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 1: Define a simple Linear Regression model in PyTorch
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# Initialize the model
input_dim = X_train.shape[1]
model = LinearRegressionModel(input_dim)

# Step 2: Simulate federated learning by manually splitting the data
datasets = [
    (torch.tensor(X_train.values[:100], dtype=torch.float32), torch.tensor(Y_train.values[:100], dtype=torch.float32)),
    (torch.tensor(X_train.values[100:200], dtype=torch.float32), torch.tensor(Y_train.values[100:200], dtype=torch.float32)),
    (torch.tensor(X_train.values[200:], dtype=torch.float32), torch.tensor(Y_train.values[200:], dtype=torch.float32))
]

# Step 3: Federated Training Function
def train_on_node(model, data, target, epochs=1):
    """Trains the model on a specific node."""
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(), target)
        loss.backward()
        optimizer.step()

    return model

def aggregate_models(models):
    """Aggregates models by averaging their parameters."""
    # Initialize a new model with the same architecture
    aggregated_model = LinearRegressionModel(input_dim)
    state_dict = aggregated_model.state_dict()

    # Sum up all model weights
    for k in state_dict.keys():
        state_dict[k] = torch.stack([models[i].state_dict()[k].float() for i in range(len(models))], 0).mean(0)

    # Load the averaged weights into the new model
    aggregated_model.load_state_dict(state_dict)
    return aggregated_model

def federated_train(nodes, datasets, epochs=5, rounds=3):
    """Federated training over multiple rounds."""
    for round_num in range(rounds):
        print(f"\n--- Round {round_num + 1} ---")
        node_models = []

        # Train each node independently
        for i, (data, target) in enumerate(datasets):
            print(f"\nNode {i+1} training started...")
            model = LinearRegressionModel(input_dim)
            trained_model = train_on_node(model, data, target, epochs=epochs)
            node_models.append(trained_model)
            print(f"Node {i+1} training completed.")

        # Aggregation on the central server
        print("\nCentral server: Aggregating models...")
        aggregated_model = aggregate_models(node_models)
        print("Central server: Model aggregation completed.")

        # Send the aggregated model back to the nodes
        for i in range(len(nodes)):
            nodes[i] = aggregated_model
        print("Aggregated model sent back to all nodes.\n")

    return aggregated_model

# Initialize nodes with the same model
nodes = [LinearRegressionModel(input_dim) for _ in range(3)]

# Train the model using federated learning simulation
aggregated_model = federated_train(nodes, datasets, epochs=5, rounds=3)

# Make predictions on the test set with the aggregated model
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
predictions = aggregated_model(X_test_tensor).detach().numpy()

# Plotting Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, predictions, color='blue', alpha=0.5, label='Predicted vs Actual')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.grid(True)
plt.show()
