import os
import time
import multiprocessing
from train_node import SwarmNode
from config import NUM_ACTIVE_CLIENTS, MAX_CONCURRENT_CLIENTS
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

def run_client(client_id, init_queue=None):
    """Worker function to initialize and run a single Swarm Edge Node."""
    try:
        print(f"[Worker] Initializing client {client_id} (pid={os.getpid()})...")
        # Flush to ensure prompt visibility
        sys.stdout.flush()
        node = SwarmNode(client_id=client_id)
        print(f"[Worker] Client {client_id} initialized (pid={os.getpid()}). Starting run...")
        # Notify parent that initialization completed so it can start another initializer
        try:
            if init_queue is not None:
                init_queue.put(client_id)
        except Exception:
            pass
        sys.stdout.flush()
        node.run()
    except Exception as e:
        print(f"❌ [Client {client_id}] Crashed or encountered an error: {e}")

def main():
    print(f"🚀 Initializing Swarm Learning network with {NUM_ACTIVE_CLIENTS} nodes...")

    # Ensure the global directory exists for the leaders to save aggregated models
    os.makedirs("../global", exist_ok=True)

    # Determine concurrency limit. Allow an environment variable to override
    # the config file for quick experiments: set MAX_CONCURRENT_CLIENTS to
    # an integer. Treat 0 or negative as "no limit" (i.e., all clients).
    env_val = os.getenv('MAX_CONCURRENT_CLIENTS')
    if env_val is not None:
        try:
            env_limit = int(env_val)
        except ValueError:
            print(f"⚠️ Invalid MAX_CONCURRENT_CLIENTS env='{env_val}', falling back to config value {MAX_CONCURRENT_CLIENTS}")
            env_limit = MAX_CONCURRENT_CLIENTS
    else:
        env_limit = MAX_CONCURRENT_CLIENTS

    if env_limit <= 0:
        max_workers = NUM_ACTIVE_CLIENTS
    else:
        max_workers = min(NUM_ACTIVE_CLIENTS, env_limit)

    print(f"🔁 Using up to {max_workers} concurrent training processes (env/config: {env_limit}/{MAX_CONCURRENT_CLIENTS})")

    # Manual process pool: limit concurrent *initializations* to `max_workers`.
    # Once a child signals it finished initialization, its slot is freed while
    # the child continues training in the background.
    manager = multiprocessing.Manager()
    init_queue = manager.Queue()

    processes = []
    initializing = set()
    for i in range(1, NUM_ACTIVE_CLIENTS + 1):
        # Process any initialization-complete notifications
        try:
            while True:
                ready_id = init_queue.get_nowait()
                if ready_id in initializing:
                    initializing.remove(ready_id)
                    print(f"🔁 Client {ready_id} finished initialization; freeing an init slot.")
        except Exception:
            pass

        # Wait until we have a free init slot
        while len(initializing) >= max_workers:
            # Drain queue while waiting
            try:
                while True:
                    ready_id = init_queue.get_nowait()
                    if ready_id in initializing:
                        initializing.remove(ready_id)
                        print(f"🔁 Client {ready_id} finished initialization; freeing an init slot.")
            except Exception:
                pass

            # Print status of currently initializing PIDs
            alive_init = [p.pid for p in processes if p.is_alive() and p._identity and p._identity[0] in range(1, NUM_ACTIVE_CLIENTS+1)]
            print(f"🔁 Active initializing PIDs: {alive_init}. Waiting for an init slot...")
            time.sleep(1)

        p = multiprocessing.Process(target=run_client, args=(i, init_queue))
        p.start()
        print(f"▶️ Started training process for client {i} (pid={p.pid})")
        processes.append(p)
        initializing.add(i)

    # Wait for remaining processes to finish
    for p in processes:
        p.join()
        print(f"✅ Training process pid={p.pid} finished.")
        
    print("\n🎉 SWARM LEARNING COMPLETE! All epochs finished and global models synchronized.")

if __name__ == "__main__":
    main()