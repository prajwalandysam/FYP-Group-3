# Node1_main.py
import sys
sys.path.append("../src")
from main import start_server

if __name__ == "__main__":
    print("Starting Node1 on port 8001...")
    start_server(8001)
