# import traceback
# from fastapi import FastAPI,Request
# from Client import *
# import json
# from Server import *
# from asyncio import Lock

# lock = Lock()
# app=FastAPI()


# server_obj=Server()

# # @app.get("/get_global_model")
# # async def get_new_server_obj():
# #     async with lock:
# #         try:
# #             dict_obj = server_obj.to_dict()
# #             print(f"Serialized server object: {dict_obj}")
# #             return {"server_obj": dict_obj}
# #         except Exception as e:
# #             return {"error": str(e)}

# @app.get("/get_global_model")
# async def get_new_server_obj():
#     async with lock:
#         try:
#             print("Server Object State Diagnostics:")
#             print(f"Type of global model coefficients: {type(server_obj.global_model.coef_)}")
#             print(f"Type of global model intercept: {type(server_obj.global_model.intercept_)}")
#             dict_obj = server_obj.to_dict()
#             print(f"Serialized server object: {dict_obj}")
#             return {"server_obj": dict_obj}
#         except Exception as e:
#             print(f"Serialization Error Details:")
#             print(f"Error type: {type(e)}")
#             print(f"Error message: {str(e)}")
#             print(f"Full traceback: {traceback.format_exc()}")
#             return {"error": str(e)}
        
# @app.post("/send_update")
# async def receive_update(request:Request):
#     data=await request.json()
#     print(data)
#     metadata = data["metadata"]
#     client_data = data["client"]
#     new_coeff=data.get("new_coeff")
#     new_update_intercept=data.get("updated_intercept")
#     d1={"coef_":new_coeff,"intercept_":new_update_intercept}

#     print("creating dummy objjj to add client--------")

#     client_obj = Client(
#             client_id=client_data["client_id"],
#             address=client_data["address"],
#             local_data = None , 
#             local_labels = None
#         )
    
#     print("adding client node to Server")
#     node_index = server_obj.add_node(d1, metadata, client_obj)

#     print(node_index)
#     return {"node_index":node_index}


# @app.get("/aggregate")
# def aggregate():
#     try:
#         print("aggregating..")
#         aggregated_model = server_obj.aggregate()
#         print("Aggregation complete, returning model.")
#         if hasattr(aggregated_model, 'to_dict'):
#             return {"model": aggregated_model.to_dict()}
#         elif hasattr(aggregated_model, '__dict__'):
#             return {"model": aggregated_model.__dict__}
#         else:
#             # If the model doesn't have a built-in dictionary conversion method
#             return {"model": str(aggregated_model)}
#     except Exception as e:
#         return {"error": str(e)}
    
# @app.get("/print-dag")
# def print_dag():
#     try:
#         server_obj.print_dag()
#         return {"message": "DAG printed on fastapi server console"}
#     except Exception as e:
#         return {"error": str(e)}

# @app.get("/get-global")
# def get_global():
    

# @app.post("/addedge")
# def add_edge():

import traceback
from fastapi import FastAPI, Request
from Client import Client
from Server import Server
from asyncio import Lock
import logging

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
        
@app.get("/get-global-parameters")
async def calculate_accuracy():
    async with lock:
        try:
            logging.debug("calculating accuracy")
            coeff=server_obj.global_model.coef_
            intercept=server_obj.global_model.intercept_
            coef_= server_obj.global_model.coef_.tolist() if hasattr(server_obj.global_model.coef_, "tolist") else server_obj.global_model.coef_,
            intercept_= server_obj.global_model.intercept_
            return {"coef":coef_,"intercept_":intercept_}
        except Exception as e:
            logging.error("Error while caluating DAG", exc_info=True)
            return {"error": str(e), "details": traceback.format_exc()}