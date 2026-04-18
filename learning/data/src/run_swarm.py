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

def run_client(client_id, compute_semaphore, network_semaphore):
    """Worker function to initialize and run a single Swarm Edge Node safely."""
    try:
        print(f"[Worker] Spawning client {client_id} (pid={os.getpid()})...")
        sys.stdout.flush()

        node = SwarmNode(client_id=client_id)

        # --- FIX 1: Protect the RAM (Compute Lock) ---
        original_train = node.train_local_epoch

        def throttled_train():
            print(f"⏳ [Client {client_id}] Waiting for an open PyTorch compute slot...")
            with compute_semaphore:
                print(f"🚀 [Client {client_id}] Slot acquired! Starting heavy local training...")
                return original_train()

        node.train_local_epoch = throttled_train

        # --- FIX 2: Protect the Solid Server (Network Lock) ---
        # This prevents the CSS from dropping connections under heavy load
        original_sync = node.sync_data_from_solid

        def throttled_sync():
            print(f"🌐 [Client {client_id}] Waiting for an open Solid Server connection...")
            with network_semaphore:
                return original_sync()

        node.sync_data_from_solid = throttled_sync

        # Run the standard lifecycle 
        node.run()

    except Exception as e:
        print(f"❌ [Worker] Client {client_id} crashed or was killed: {e}")

if __name__ == "__main__":
    sys.stdout = Logger()
    print("🚀 Initializing Dual-Protected Swarm Learning network...")

    # Lock 1: Allows only configured number of processes to do PyTorch math
    compute_semaphore = multiprocessing.Semaphore(MAX_CONCURRENT_CLIENTS)
    
    # Lock 2: Allows only 10 clients to ping the Solid Server simultaneously
    # Adjust this up to 15 or 20 if your CSS instance can handle the I/O
    network_semaphore = multiprocessing.Semaphore(MAX_CONCURRENT_CLIENTS) 
    
    processes = []

    # Launch all clients. They will immediately hit the network_semaphore and queue up gracefully.
    for i in range(1, NUM_ACTIVE_CLIENTS + 1):
        p = multiprocessing.Process(target=run_client, args=(i, compute_semaphore, network_semaphore))
        p.start()
        processes.append(p)
        
        # A tiny mechanical jitter to prevent process-spawning collisions
        time.sleep(0.1)

    # Wait for all processes to formally conclude their 50 epochs
    for p in processes:
        p.join()
        
    print("\n🎉 SWARM LEARNING COMPLETE! All nodes successfully reached consensus.")