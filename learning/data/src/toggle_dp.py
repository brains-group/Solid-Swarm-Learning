import os
import json
import argparse
from web3 import Web3

ANVIL_RPC_URL = "http://127.0.0.1:8545"
CONTRACT_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3" # Ensure this matches your deployment!
ABI_PATH = "../../../swarm_orchestrator/out/SwarmCoordinator.sol/SwarmCoordinator.json"
PODS_DIR = "../pods"

def main():
    parser = argparse.ArgumentParser(description="Set the Differential Privacy epsilon budget for an Edge Node.")
    parser.add_argument("client_id", type=int, help="The ID of the client (e.g., 1)")
    parser.add_argument("epsilon", type=float, help="Privacy budget: 0 (Off), 0.1 (Max Noise), 1.0 (Standard), 10.0+ (Low Noise)")
    args = parser.parse_args()

    client_id = args.client_id
    epsilon = args.epsilon
    
    # Scale the float by 100 to safely store it as an integer in Solidity
    scaled_epsilon = int(epsilon * 100)
    
    pod_dir = os.path.join(PODS_DIR, f"client_{client_id}")
    profile_path = os.path.join(pod_dir, 'profile.json')

    if not os.path.exists(profile_path):
        print(f"❌ Error: Cannot find profile for Client {client_id}")
        return

    with open(profile_path, 'r') as f:
        profile = json.load(f)
        wallet_address = profile['wallet_address']
        private_key = profile['private_key']

    w3 = Web3(Web3.HTTPProvider(ANVIL_RPC_URL))
    with open(ABI_PATH, 'r') as f:
        contract_json = json.load(f)
        contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=contract_json['abi'])

    print(f"🔒 Updating DP Epsilon for Client {client_id} to {epsilon} (Scaled on-chain as {scaled_epsilon})...")

    try:
        tx = contract.functions.setDPBudget(scaled_epsilon).build_transaction({
            'from': wallet_address,
            'nonce': w3.eth.get_transaction_count(wallet_address),
            'gas': 100000,
            'gasPrice': w3.eth.gas_price
        })

        signed_tx = w3.eth.account.sign_transaction(tx, private_key=private_key)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

        if receipt.status == 1:
            print(f"✅ Success! Client {client_id} DP Epsilon is now {epsilon}.")
        else:
            print(f"❌ Transaction failed.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()