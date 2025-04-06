# Node4_main.py
import sys
sys.path.append("../src")
from main import start_server

if __name__ == "__main__":
    print("Starting Node4 on port 8004...")
    start_server(8004)
