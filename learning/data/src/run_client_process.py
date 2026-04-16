#!/usr/bin/env python3
import sys
import time
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: run_client_process.py <client_id>")
        sys.exit(2)

    client_id = int(sys.argv[1])

    try:
        # Import heavy modules inside the child process
        from train_node import SwarmNode

        # Jitter to avoid simultaneous heavy startup
        time.sleep(np.random.uniform(1, 15))

        node = SwarmNode(client_id=client_id)
        node.run()
    except Exception as e:
        print(f"❌ [Client {client_id}] Crashed or encountered an error: {e}")
        sys.exit(1)
