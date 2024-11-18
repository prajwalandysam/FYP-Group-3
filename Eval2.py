# %% [markdown]
# ## Import the Necessary Libraries

# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## Data Preprocessing

# %%
data = pd.read_csv("Steel_industry_data.csv")
print(data.isnull().sum())  # Count of missing values in each column

data = data.drop(columns=['date'])
X_train=data.values

data.head()


# %%
print(X_train)

# %% [markdown]
# Feature Scaling

# %%
#Feature Scaling
scaler = StandardScaler()


X_train[:, 1:7] = scaler.fit_transform(X_train[:, 1:7]).astype('float64')


print(X_train)
X_train.shape


# %% [markdown]
# Column tranforming and label encoding

# %%
# Transforming Categorical data into number matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[7,8])],remainder='passthrough')
X_train=ct.fit_transform(X_train)
print(X_train)
print(X_train.shape)



# %%
#label encoding on the suitable column
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X_train[:,-1]=le.fit_transform(X_train[:,-1])
print(X_train)
type(X_train)
df=pd.DataFrame(X_train)
df.to_csv('scaled_Steel_industry_data.csv', index=False, header=True)

# %% [markdown]
# ## Federated Learning

# %%
#splitting the entire data into 8 nodes to perform FL
node_data = np.array_split(X_train, 8)


# %%
global_model = LinearRegression() #Global Model

# %% [markdown]
# Funtions for calculating and aggregating the local models 

# %%

def federated_averaging(models):
    """Aggregate model weights using the average."""
    coef_avg = np.mean([model.coef_ for model in models], axis=0)
    intercept_avg = np.mean([model.intercept_ for model in models])
    return coef_avg, intercept_avg

# Function to train the local model on each node's data
def train_local_model(data):
  
    y = data[:, 11]   # Target (Usage_kWh)
  
    X = np.concatenate((data[:, :11], data[:, 12:]), axis=1)
    model = LinearRegression()
    model.fit(X, y)
    return model

# %% [markdown]
# Federated Learning Process

# %%

for iteration in range(3):
    print(f"Iteration {iteration + 1}:")

    # Local models for each node
    local_models = []

    # Train each node's model using 1460 rows in this iteration
    for i, node in enumerate(node_data):
        start = iteration * 1460
        end = start + 1460
        node_subset = node[start:end]

        # Train the local model on the subset
        local_model = train_local_model(node_subset)
        local_models.append(local_model)

        # Evaluate the local model
        y_pred = local_model.predict(np.concatenate((node_subset[:, :11], node_subset[:, 12:]), axis=1))
        mse = mean_squared_error(node_subset[:, 11], y_pred)
        print(f"  Node {i + 1} - MSE: {mse:.4f}")

    # Aggregate the local model updates using Federated Averaging
    coef_avg, intercept_avg = federated_averaging(local_models)

    # Update the global model with aggregated parameters
    global_model.coef_ = coef_avg
    global_model.intercept_ = intercept_avg

    print(f"Global model updated after iteration {iteration + 1}.\n")


# %% [markdown]
# Final Evaluation of our Global model

# %%
# Final Evaluation (Optional): Use some test data for evaluation
test_data = node_data[0][:1460]  # Example: Using first 1460 rows from node 1 as test data
# X_test = test_data[:, 1:]
# y_test = test_data[:, 1]
test_data

y_test = test_data[:, 11]   # Target (Usage_kWh)
    # Select all columns except the 12th column using slicing
X_test= np.concatenate((test_data[:, :11], test_data[:, 12:]), axis=1)

df=pd.DataFrame(y_test)
df.to_csv('test_data.csv', index=False, header=True)

y_pred = global_model.predict(X_test)
final_mse = mean_squared_error(y_test, y_pred)
print(f"Final Global Model MSE on Test Data: {final_mse:.4f}")

# %% [markdown]
# comparing the predicted results vs the actual results

# %%
output=np.concatenate((y_pred.reshape(-1,1),y_test.reshape(-1,1)),1)

df=pd.DataFrame(output)
df.to_csv("output.csv", index=False, header=True)

# %%
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2)

# %% [markdown]
# ## Visualizing the Results (FL)

# %%
import matplotlib.pyplot as plt

# %%
plt.scatter(y_pred,y_test,color='red')

plt.title("FL using Linear reg")
plt.xlabel('y_pred')
plt.ylabel('y_test')
plt.show()

# %% [markdown]
# ## Comparision of Global model if done Without FL

# %%
regressor=LinearRegression()
without_FL=X_train

X_withoutFL= np.concatenate((without_FL[:, :11], without_FL[:, 12:]), axis=1)
y_withoutFL = without_FL[:, 11]   # Target (Usage_kWh)


# %%
from sklearn.model_selection import train_test_split

X_withoutFL_train, X_withoutFL_test,y_withoutFL_train,y_withoutFL_test = train_test_split(X_withoutFL,y_withoutFL,test_size=0.25,random_state=0)

# %%
X_withoutFL_test.shape

# %%
X_withoutFL_train.shape

# %%
y_withoutFL_test.shape

# %%
y_withoutFL_train.shape

# %%
regressor.fit(X_withoutFL_train,y_withoutFL_train)

# %%
pred=regressor.predict(X_withoutFL_test)

print(pred)

# %% [markdown]
# ## Visualising the results (Without FL)

# %%
plt.scatter(pred,y_withoutFL_test,color='red')

plt.title("Training the same dataset without FL (regular LR)")
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()

# %%
pred.shape

# %%
from sklearn.metrics import r2_score
r2 = r2_score(y_withoutFL_test, pred)
print(r2)

# %%



