import requests
from sklearn.metrics import mean_squared_error
from datetime import datetime


def client_task(client, server_urls):
    """
    Function to execute the task for each client.
    :param client: The Client object
    :param server_url: URL of the server API to interact with
    :return: Tuple with client_id and Mean Squared Error (MSE)
    """
    try:
        # Step 1:  client fetches the global model from its server
        response = requests.get(f"{server_urls[0]}/get_global_model")
        if response.status_code == 200:
            payload = response.json()
            global_model= payload["server_obj"]["global_model"]
            print(global_model)
            print(f"Client {client.client_id} fetched the global model.")
        else:
            raise Exception(f"Failed to fetch global model. Status code: {response.status_code}")

        # Step 2: Client trains its local model based on the global model
        updated_coeff,updated_intercept = client.train_local_model(global_model)

        # Step 3: Create metadata with the current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Format timestamp to milliseconds
        metadata = {"weight": 1, "timestamp": timestamp}

        # Step 4: Client sends its update to the server
        update_data = {
            "metadata": metadata,
            "client": {
                "client_id": client.client_id,
                "address": client.address
            },
            "new_coeff": updated_coeff,
            "updated_intercept": updated_intercept
            
        }
        for url in server_urls:
            update_response = requests.post(f"{url}/send_update", json=update_data)
            if update_response.status_code == 200:
                print(f"Client {client.client_id} successfully sent update to server url {url}.")
            else:
                raise Exception(f"Failed to send update. Status code: {update_response.text}")

            # Step 5: Evaluate the client's local model using MSE
            y_pred = client.local_model.predict(client.local_data)
            mse = mean_squared_error(client.local_labels, y_pred)
            print("mse is " + str(mse))

            # Step 6: Request the server to perform aggregation
            agg_response = requests.get(f"{url}/aggregate")
            print(f"aggregation done by url: {url}")
            if agg_response.status_code == 200:
                aggregated_model = agg_response.json()
                print(f"Client {client.client_id} received aggregated model.")
            else:
                raise Exception(f"Failed to aggregate models. Status code: {agg_response.status_code}")

            # Return client ID and MSE
        return client.client_id, mse

    except Exception as e:
        print(f"Error in client task for {client.client_id}: {str(e)}")
        return client.client_id, None
