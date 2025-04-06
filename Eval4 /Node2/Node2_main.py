# Node2_main.py
import sys
sys.path.append("../src")
from main import start_server

if __name__ == "__main__":
    print("Starting Node2 on port 8002...")
    start_server(8002)
