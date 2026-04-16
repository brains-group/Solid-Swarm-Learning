import os
import json
from web3 import Web3

from config import NUM_ACTIVE_CLIENTS
import time
import math

# Batch/register throttling settings
# Number of clients to register before sleeping (to avoid overwhelming Anvil)
BATCH_SIZE = 5
# Seconds to wait after each batch
BATCH_DELAY = 0.8
# How many times to retry a failing on-chain transaction
TX_RETRIES = 2
RETRY_DELAY = 0.5

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

    for idx, client_folder in enumerate(client_folders, start=1):
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
        # Use a conservative value; wait for the funding receipt before continuing
        fund_value = w3.to_wei(10, 'ether')
        fund_nonce = w3.eth.get_transaction_count(whale_account.address)
        fund_tx = {
            'nonce': fund_nonce,
            'to': new_account.address,
            'value': fund_value,
            'gas': 21000,
            'gasPrice': w3.eth.gas_price
        }
        signed_fund_tx = w3.eth.account.sign_transaction(fund_tx, WHALE_KEY)
        try:
            fund_tx_hash = w3.eth.send_raw_transaction(signed_fund_tx.raw_transaction)
            # Wait for funding to be mined so the new account has balance
            w3.eth.wait_for_transaction_receipt(fund_tx_hash)
        except Exception as e:
            print(f"⚠️ Funding failed for {client_folder}: {e}")
            # Attempt small pause and continue; registration will likely fail without funds
            time.sleep(0.2)

        # 4. Register the client to the smart contract. Retry a couple times if necessary.
        success = False
        for attempt in range(TX_RETRIES + 1):
            try:
                reg_nonce = w3.eth.get_transaction_count(new_account.address)
                reg_tx = contract.functions.registerNode(swarm_ids).build_transaction({
                    'from': new_account.address,
                    'nonce': reg_nonce,
                    'gas': 500000,
                    'gasPrice': w3.eth.gas_price
                })

                signed_reg_tx = w3.eth.account.sign_transaction(reg_tx, new_account.key)
                tx_hash = w3.eth.send_raw_transaction(signed_reg_tx.raw_transaction)
                receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

                if receipt.status == 1:
                    print(f"✅ {client_folder} registered to Swarms {swarm_ids} with address {new_account.address[:8]}...")
                    success = True
                    break
                else:
                    print(f"❌ {client_folder} registration transaction reverted.")
            except Exception as e:
                print(f"⚠️ Registration attempt {attempt+1} failed for {client_folder}: {e}")
                time.sleep(RETRY_DELAY)

        if not success:
            print(f"❌ {client_folder} failed to register after {TX_RETRIES+1} attempts.")

        # Throttle in batches to avoid overwhelming the local node
        if idx % BATCH_SIZE == 0:
            print(f"⏸️ Registered {idx} clients, sleeping {BATCH_DELAY}s to avoid flooding Anvil...")
            time.sleep(BATCH_DELAY)

if __name__ == "__main__":
    main()