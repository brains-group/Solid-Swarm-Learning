import os
import json
import argparse
from web3 import Web3

# --- Configuration (Must match your other scripts) ---
ANVIL_RPC_URL = "http://127.0.0.1:8545"
CONTRACT_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3" 
ABI_PATH = "../../../swarm_orchestrator/out/SwarmCoordinator.sol/SwarmCoordinator.json"
PODS_DIR = "../pods"

def main(): # python3 toggle_privacy.py #(1/2/3/...) exclude/include
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Toggle vulnerable data privacy settings for a Solid Pod / Edge Node.")
    parser.add_argument("client_id", type=int, help="The ID of the client to target (e.g., 1)")
    parser.add_argument("action", choices=["exclude", "include"], help="Type 'exclude' to hide vulnerable data, or 'include' to share it.")
    args = parser.parse_args()

    client_id = args.client_id
    exclude_flag = True if args.action == "exclude" else False
    
    pod_dir = os.path.join(PODS_DIR, f"client_{client_id}")
    profile_path = os.path.join(pod_dir, 'profile.json')

    if not os.path.exists(profile_path):
        print(f"❌ Error: Cannot find profile for Client {client_id} at {profile_path}")
        return

    # 1. Load the Client's Private Key
    with open(profile_path, 'r') as f:
        profile = json.load(f)
        wallet_address = profile['wallet_address']
        private_key = profile['private_key']

    # 2. Connect to Blockchain
    w3 = Web3(Web3.HTTPProvider(ANVIL_RPC_URL))
    if not w3.is_connected():
        print("❌ Error: Failed to connect to Anvil RPC!")
        return

    with open(ABI_PATH, 'r') as f:
        contract_json = json.load(f)
        contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=contract_json['abi'])

    # 3. Build and Sign the Transaction
    print(f"🔒 Requesting Smart Contract update for Client {client_id} ({wallet_address})...")
    print(f"   Setting excludeVulnerableData to: {exclude_flag}")

    try:
        tx = contract.functions.setPrivacyPreference(exclude_flag).build_transaction({
            'from': wallet_address,
            'nonce': w3.eth.get_transaction_count(wallet_address),
            'gas': 100000, # Lightweight state change
            'gasPrice': w3.eth.gas_price
        })

        signed_tx = w3.eth.account.sign_transaction(tx, private_key=private_key)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        
        # Wait for the block to be mined
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

        if receipt.status == 1:
            status_str = "HIDDEN" if exclude_flag else "SHARED"
            print(f"✅ Success! Client {client_id}'s vulnerable data is now {status_str}.")
        else:
            print(f"❌ Transaction failed. Check your Anvil logs.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()