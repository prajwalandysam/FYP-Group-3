# Node3_main.py
import sys
sys.path.append("../src")
from main import start_server

if __name__ == "__main__":
    print("Starting Node3 on port 8003...")
    start_server(8003)
