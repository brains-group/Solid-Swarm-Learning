import os
import json
from web3 import Web3

from config import NUM_ACTIVE_CLIENTS

# --- Configuration ---
ANVIL_RPC_URL = "http://127.0.0.1:8545"
CONTRACT_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3" 

# Go up 3 levels from /learning/data/src to reach /Swarm, then down into the orchestrator
ABI_PATH = "../../../swarm_orchestrator/out/SwarmCoordinator.sol/SwarmCoordinator.json"

# Go up 1 level from /learning/data/src to reach /learning/data, then into pods
PODS_DIR = "../pods"

# Anvil's default Account #0 (The Whale)
WHALE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

def main():
    w3 = Web3(Web3.HTTPProvider(ANVIL_RPC_URL))
    assert w3.is_connected(), "Failed to connect to Anvil!"
    
    whale_account = w3.eth.account.from_key(WHALE_KEY)

    # Load the Contract ABI
    with open(ABI_PATH, 'r') as f:
        contract_json = json.load(f)
        contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=contract_json['abi'])

    # Get a list of all client folders (client_1, client_2, ... client_50)
    client_folders = [f for f in os.listdir(PODS_DIR) if f.startswith('client_')]
    client_folders.sort(key=lambda x: int(x.split('_')[1]))

    # ---> Apply the global limit here <---
    client_folders = client_folders[:NUM_ACTIVE_CLIENTS]

    print(f"Found and selected {len(client_folders)} clients. Beginning network registration...")

    for client_folder in client_folders:
        client_path = os.path.join(PODS_DIR, client_folder)
        
        # 1. Read the profile to get the sub-swarm IDs
        with open(os.path.join(client_path, 'profile.json'), 'r') as f:
            profile = json.load(f)
            swarm_ids = profile['swarm_ids']  # Grab the array directly

        # 2. Generate a unique wallet for this client
        new_account = w3.eth.account.create()
        
        # Save the private key to the pod so the client can use it later for submitting weights!
        profile['wallet_address'] = new_account.address
        profile['private_key'] = new_account.key.hex()
        with open(os.path.join(client_path, 'profile.json'), 'w') as f:
            json.dump(profile, f, indent=4)

        # 3. Fund the new wallet with fake ETH from the Whale so it can pay gas
        fund_tx = {
            'nonce': w3.eth.get_transaction_count(whale_account.address),
            'to': new_account.address,
            'value': w3.to_wei(174,'ether'),
            'gas': 21000,
            'gasPrice': w3.eth.gas_price
        }
        signed_fund_tx = w3.eth.account.sign_transaction(fund_tx, WHALE_KEY)
        w3.eth.send_raw_transaction(signed_fund_tx.raw_transaction)

        # 4. Register the client to the smart contract
        reg_tx = contract.functions.registerNode(swarm_ids).build_transaction({
            'from': new_account.address,
            'nonce': w3.eth.get_transaction_count(new_account.address),
            'gas': 500000,
            'gasPrice': w3.eth.gas_price
        })
        
        signed_reg_tx = w3.eth.account.sign_transaction(reg_tx, new_account.key)
        tx_hash = w3.eth.send_raw_transaction(signed_reg_tx.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

        if receipt.status == 1:
            print(f"✅ {client_folder} registered to Swarms {swarm_ids} with address {new_account.address[:8]}...")
        else:
            print(f"❌ {client_folder} failed to register.")

if __name__ == "__main__":
    main()