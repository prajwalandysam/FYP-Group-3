
import traceback
import numpy as np
from fastapi import FastAPI, Request
from typing import List 
import sys
sys.path.append("../src")  # Add the parent directory to the module search path

from pydantic import BaseModel
# Now import from the parent directory
from Client import Client
from Server import Server
from Client_task import client_task
from asyncio import Lock
import logging
import uvicorn  # Import uvicorn to run the app

# Initialize FastAPI app and dependencies
app = FastAPI()
server_obj = Server()
lock = Lock()

# Configure logging for detailed debugging
logging.basicConfig(level=logging.DEBUG)


@app.get("/get_global_model")
async def get_new_server_obj():
    """Endpoint to retrieve the current global model."""
    async with lock:
        try:
            logging.debug("Fetching global model state.")
            dict_obj = server_obj.to_dict()  # Ensure Server has a proper to_dict method
            logging.debug(f"Serialized server object: {dict_obj}")
            return {"server_obj": dict_obj}
        except Exception as e:
            logging.error("Error while serializing server object", exc_info=True)
            return {"error": str(e), "details": traceback.format_exc()}


@app.post("/send_update")
async def receive_update(request: Request):
    """Endpoint to receive client updates."""
    async with lock:
        try:
            data = await request.json()
            logging.debug(f"Received data: {data}")

            metadata = data["metadata"]
            client_data = data["client"]
            new_coeff = data.get("new_coeff")
            new_update_intercept = data.get("updated_intercept")
            d1 = {"coef_": new_coeff, "intercept_": new_update_intercept}

            # Create and add client object
            client_obj = Client(
                client_id=client_data["client_id"],
                address=client_data["address"],
                local_data=None,
                local_labels=None,
            )
            logging.debug("Adding client node to server.")
            node_index = server_obj.add_node(d1, metadata, client_obj)
            logging.info(f"Client node added at index: {node_index}")

            return {"node_index": node_index}
        except Exception as e:
            logging.error("Error in /send_update endpoint", exc_info=True)
            return {"error": str(e), "details": traceback.format_exc()}


@app.get("/aggregate")
async def aggregate():
    """Endpoint to aggregate the global model."""
    async with lock:
        try:
            logging.debug("Starting model aggregation.")
            aggregated_model = server_obj.aggregate()
            logging.info("Aggregation complete.")
            
            # Serialize aggregated model
            if hasattr(aggregated_model, 'to_dict'):
                model_data = aggregated_model.to_dict()
            elif hasattr(aggregated_model, '_dict_'):
                model_data = aggregated_model._dict_
            else:
                model_data = str(aggregated_model)
            
            return {"model": model_data}
        except Exception as e:
            logging.error("Error during aggregation", exc_info=True)
            return {"error": str(e), "details": traceback.format_exc()}
        
        

class SyncRequest(BaseModel):
    urls: List[str]  # Ensure the request expects a JSON list

@app.post("/sync-global-model")
async def sync_global(request: SyncRequest):
    response = server_obj.sync_global_model(request.urls)
    return response
    
    
@app.get("/print-dag")
async def print_dag():
    """Endpoint to print the DAG."""
    async with lock:
        try:
            logging.debug("Printing DAG.")
            server_obj.print_dag()
            return {"message": "DAG printed on FastAPI server console"}
        except Exception as e:
            logging.error("Error while printing DAG", exc_info=True)
            return {"error": str(e), "details": traceback.format_exc()}
        
@app.get("/prune-graph")
async def prune_graph():
    """Endpoint to prune the graph."""
    async with lock:
        try:
            logging.debug("pruning Graph.")
            server_obj.prune_graph()
            return {"message": "DAG pruned"}
        except Exception as e:
            logging.error("Error while pruning  DAG", exc_info=True)
            return {"error": str(e), "details": traceback.format_exc()}  
        
@app.get("/plot-graph")
async def plot_graph():
    """Endpoint to plot the graph."""
    async with lock:
        try:
            logging.debug("plotting Graph.")
            server_obj.plot_graph()
            return {"message": "DAG plotted"}
        except Exception as e:
            logging.error("Error while plotting  DAG", exc_info=True)
            return {"error": str(e), "details": traceback.format_exc()}               
        
@app.get("/get-global-parameters")
async def calculate_accuracy():
    async with lock:
        try:
            logging.debug("calculating accuracy")
            coeff=server_obj.global_model.coef_
            intercept=server_obj.global_model.intercept_
            time=server_obj.timestamp
            coef_= server_obj.global_model.coef_.tolist() if hasattr(server_obj.global_model.coef_, "tolist") else server_obj.global_model.coef_,
            intercept_= server_obj.global_model.intercept_
            return {"coef":coef_,"intercept_":intercept,"time":time}
        except Exception as e:
            logging.error("Error while caluating DAG", exc_info=True)
            return {"error": str(e), "details": traceback.format_exc()}
@app.post("/trigger_sync")
async def trigger_sync(request: Request):
    """API to sync the global model across nodes."""
    data = await request.json()
    server_urls = data.get("server_urls", [])
    
    if not server_urls:
        return {"error": "No server URLs provided for synchronization."}
    
    sync_result = server_obj.sync_global_model(server_urls)  # Ensure this matches the new function
    return sync_result

@app.post("/update_global_model")
async def update_global_model(request: Request):
    """Endpoint to update the server’s global model with a new version."""
    async with lock:
        try:
            data = await request.json()
            logging.debug(f"Received global model update: {data}")

            # Extract model parameters
            new_coeff = data["model"]["coef_"]
            new_intercept = data["model"]["intercept_"]
            timestamp = data["timestamp"]

            # Update the server's global model
            server_obj.global_model.coef_ = np.array(new_coeff)
            server_obj.global_model.intercept_ = new_intercept

            # Log update
            logging.info(f"Global model updated successfully at {timestamp}")

            return {"message": "Global model updated successfully", "timestamp": timestamp}

        except Exception as e:
            logging.error("Error while updating global model", exc_info=True)
            return {"error": str(e), "details": traceback.format_exc()}
        
        
def start_server(port):
    """Function to start FastAPI server on a given port."""
    uvicorn.run(app, host="0.0.0.0", port=port)