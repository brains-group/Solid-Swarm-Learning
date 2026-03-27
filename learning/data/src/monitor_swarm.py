import time
import json
import os
from web3 import Web3
from config import NUM_ACTIVE_CLIENTS, TOTAL_EPOCHS

# --- Configuration ---

ANVIL_RPC_URL = "http://127.0.0.1:8545"
CONTRACT_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3" 
ABI_PATH = "../../../swarm_orchestrator/out/SwarmCoordinator.sol/SwarmCoordinator.json"

def main():
    w3 = Web3(Web3.HTTPProvider(ANVIL_RPC_URL))

    with open(ABI_PATH, 'r') as f:
        contract_json = json.load(f)
        contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=contract_json['abi'])

    print("Connecting to blockchain...")
    total_expected = NUM_ACTIVE_CLIENTS

    while True:
        try:
            

            os.system('cls' if os.name == 'nt' else 'clear')
            print("==========================================================================")
            print("                         LIVE GLOBAL SWARM MONITOR                          ")
            print("==========================================================================")

            # We only have one global swarm with ID 0 now.
            swarm_id = 0
            swarm_data = contract.functions.swarms(swarm_id).call()

            nodes_submitted, current_epoch, leader = swarm_data[0], swarm_data[1], swarm_data[2]

            if current_epoch >= TOTAL_EPOCHS:
                print(f"\n🎉 GLOBAL SWARM COMPLETE: Reached final epoch.")
                print("All models synchronized. Exiting monitor...")
                break

            # Determine the live status of the swarm
            if total_expected == 0:
                status = "Idle (No clients expected)"
            elif nodes_submitted >= total_expected*0.8:
                status = "AGGREGATING"
            else:
                status = f"Waiting on {total_expected - nodes_submitted} nodes..."

            leader_str = f"{leader[:8]}..." if leader != "0x0000000000000000000000000000000000000000" else "None      "

            print(f"Global Swarm | Epoch: {current_epoch} / {TOTAL_EPOCHS} | Submitted: {nodes_submitted}/{total_expected} | Leader: {leader_str} | Status: {status}")

            print("\n==========================================================================")
            print("Press Ctrl+C to exit")
            time.sleep(2)
        except KeyboardInterrupt:
            print("\nExiting monitor.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    main()