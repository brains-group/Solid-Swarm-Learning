import os
import time
import multiprocessing
from train_node import SwarmNode
from config import NUM_ACTIVE_CLIENTS
import sys
import os

# --- Logging Setup ---
class Logger(object):
    def __init__(self, filename="swarm_training.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w") # Use "a" to append instead of overwrite

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Force write to disk immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Route all print statements to our custom Logger
sys.stdout = Logger()

# Configuration

def run_client(client_id):
    """Worker function to initialize and run a single Swarm Edge Node."""
    try:
        node = SwarmNode(client_id=client_id)
        node.run()
    except Exception as e:
        print(f"❌ [Client {client_id}] Crashed or encountered an error: {e}")

def main():
    print(f"🚀 Initializing Swarm Learning network with {NUM_ACTIVE_CLIENTS} nodes...")
    
    # Ensure the global directory exists for the leaders to save aggregated models
    os.makedirs("../global", exist_ok=True)
    
    processes = []

    # Spawn a separate process for each client
    for i in range(1, NUM_ACTIVE_CLIENTS + 1):
        p = multiprocessing.Process(target=run_client, args=(i,))
        processes.append(p)
        p.start()
        
        # Stagger the startups by 0.5 seconds to prevent overwhelming the Anvil RPC 
        # and your laptop's memory allocation all at the exact same millisecond.
        time.sleep(0.5)
        
    print(f"✅ All {NUM_ACTIVE_CLIENTS} nodes are now live and training independently!")
    
    # Keep the main script alive until all clients finish their epochs
    for p in processes:
        p.join()
        
    print("\n🎉 SWARM LEARNING COMPLETE! All epochs finished and global models synchronized.")

if __name__ == "__main__":
    main()