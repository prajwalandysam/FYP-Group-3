import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

server_url = "http://127.0.0.1:8000"  # Server endpoint
response = requests.get(f"{server_url}/get-global-parameters")
print(response.json())
global_model=response.json()
coef = np.array(global_model['coef']).flatten()  # Use flatten to convert 2D array to 1D if needed
intercept = global_model['intercept_']

node_file = "./Datasets/node8.csv"
data = pd.read_csv(node_file)

# Extract features (all columns except the 9th one)
local_dataset = data.drop(columns=data.columns[9]).values  # Drop the 9th column for features

# Extract labels (9th column)
y_test = data.iloc[:, 9].values  # Select the 9th column for labels

FLregressor = LinearRegression()

FLregressor.coef_ = coef.copy()
FLregressor.intercept_ = intercept


# FLregressor.fit(local_dataset,targetvar)

y_pred= FLregressor.predict(local_dataset)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2) #With FL


