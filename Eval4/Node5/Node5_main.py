# Node5_main.py
import sys
sys.path.append("../src")
from main import start_server

if __name__ == "__main__":
    print("Starting Node5 on port 8005...")
    start_server(8005)
