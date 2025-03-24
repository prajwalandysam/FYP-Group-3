# Node6_main.py
import sys
sys.path.append("../src")
from main import start_server

if __name__ == "__main__":
    print("Starting Node6 on port 8006...")
    start_server(8006)
