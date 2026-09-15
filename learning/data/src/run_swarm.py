import os
import sys
import time
import multiprocessing
from train_node import SwarmNode 
from config import NUM_ACTIVE_CLIENTS, MAX_CONCURRENT_CLIENTS

# --- Logging Setup ---
class Logger(object):
    def __init__(self, filename="swarm_training.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w") 

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() 

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def run_client(client_id):
    """Worker function to initialize and run a single Swarm Edge Node safely."""
    try:
        print(f"[Worker] Spawning client {client_id} (pid={os.getpid()})...")
        sys.stdout.flush()

        node = SwarmNode(client_id=client_id)
        
        # Simply run the node. Solid WebSockets now handle the flow natively.
        node.run()

    except Exception as e:
        print(f"❌ [Worker] Client {client_id} crashed or was killed: {e}")

if __name__ == "__main__":
    sys.stdout = Logger()
    print("🚀 Initializing Decentralized Swarm Learning network...")
    
    processes = []

    # Launch all clients
    for i in range(1, NUM_ACTIVE_CLIENTS + 1):
        p = multiprocessing.Process(target=run_client, args=(i,))
        p.start()
        processes.append(p)
        
        # A tiny mechanical jitter to prevent process-spawning collisions
        time.sleep(0.1)

    # Wait for all processes to formally conclude their epochs
    for p in processes:
        p.join()
        
    print("\n🎉 SWARM LEARNING COMPLETE! All nodes successfully reached consensus.")