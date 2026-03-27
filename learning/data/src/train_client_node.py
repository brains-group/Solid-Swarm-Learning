import os
import json
import time
import argparse
from web3 import Web3

# --- Configuration ---

# This would be your contract's ABI. You get this after compiling.
CONTRACT_ABI = [] 
# This is the address the contract was deployed to.
CONTRACT_ADDRESS = "0x..." 
# The RPC endpoint of your local Anvil node.
RPC_URL = "http://127.0.0.1:8545"

class ClientNode:
    def __init__(self, client_id, private_key):
        self.client_id = client_id
        self.pod_path = os.path.join('../data/pods', f'client_{client_id}')
        self.profile = self._load_profile()
        self.swarm_ids = self.profile.get("swarm_ids", [])

        # --- Web3 Setup ---
        self.w3 = Web3(Web3.HTTPProvider(RPC_URL))
        self.account = self.w3.eth.account.from_key(private_key)
        self.contract = self.w3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)
        print(f"Client {client_id} | Address: {self.account.address} | Swarms: {self.swarm_ids}")

    def _load_profile(self):
        """Loads the client's profile.json file."""
        profile_path = os.path.join(self.pod_path, 'profile.json')
        if not os.path.exists(profile_path):
            raise FileNotFoundError(f"Profile not found for client {self.client_id}")
        with open(profile_path, 'r') as f:
            return json.load(f)

    def register_on_chain(self):
        """Registers this node with the SwarmCoordinator for its swarms."""
        print(f"Client {self.client_id}: Registering for swarms {self.swarm_ids}...")
        # TODO: Build and send a transaction to the `registerNode` function.
        # You'll need to handle nonce, gas, etc.
        # tx = self.contract.functions.registerNode(self.swarm_ids).build_transaction({...})
        # signed_tx = self.w3.eth.account.sign_transaction(tx, self.account.privateKey)
        # tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        # print(f"Registration sent! Tx hash: {tx_hash.hex()}")
        pass

    def local_training_epoch(self):
        """Simulates a local training epoch."""
        print(f"\nClient {self.client_id}: Starting local training epoch...")
        # TODO: Load `train.csv` from self.pod_path
        # TODO: Run your ML model training logic here.
        time.sleep(5) # Simulate work

        # After training, save the weights and get a URI
        weights_uri = self._save_local_weights()
        print(f"Client {self.client_id}: Local training complete. Weights saved to {weights_uri}")
        return weights_uri

    def _save_local_weights(self):
        """Saves model weights to a file and returns its path (URI)."""
        # In a real system, this would be an IPFS hash or a secure URL.
        # For our simulation, the file path is sufficient.
        weights_path = os.path.join(self.pod_path, 'local_weights.json')
        mock_weights = {"param1": self.client_id * 0.1, "param2": list(range(self.client_id))}
        with open(weights_path, 'w') as f:
            json.dump(mock_weights, f)
        return weights_path

    def submit_weights_to_chain(self, weights_uri):
        """Submits the URI of the trained weights to the contract for each swarm."""
        for swarm_id in self.swarm_ids:
            print(f"Client {self.client_id}: Submitting weights for swarm {swarm_id}...")
            # TODO: Build and send a transaction to the `submitWeights` function.
            # tx = self.contract.functions.submitWeights(swarm_id, weights_uri).build_transaction({...})
            pass

    def listen_for_leadership(self):
        """A simple loop to check if this node has been elected leader."""
        print(f"Client {self.client_id}: Listening for leadership role...")
        # TODO: This should be a more robust event listener.
        # For now, we can just read from the contract state in a loop.
        for swarm_id in self.swarm_ids:
            # swarm_state = self.contract.functions.swarms(swarm_id).call()
            # current_leader = swarm_state[4] # Index of currentLeader in Swarm struct
            # if current_leader == self.account.address:
            #     print(f"I AM THE LEADER for swarm {swarm_id}!")
            #     self.perform_leader_duties(swarm_id)
            pass

    def perform_leader_duties(self, swarm_id):
        """The leader's logic to aggregate weights and update the global model."""
        print(f"Leader {self.client_id}: Performing duties for swarm {swarm_id}")
        # TODO: Call `getAllWeightsUris(swarm_id)`
        # TODO: "Download" all weights (read the files)
        # TODO: Average the weights
        # TODO: Save the new global model and get its URI
        # TODO: Call `submitGlobalModel(swarm_id, new_global_model_uri)`
        pass

if __name__ == "__main__":
    # In a real simulation, you'd launch this script multiple times,
    # passing in a different client_id and private_key for each.
    # Anvil provides 10 default accounts and private keys.
    node = ClientNode(client_id=1, private_key="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80")
    node.register_on_chain()
    weights_uri = node.local_training_epoch()
    node.submit_weights_to_chain(weights_uri)
    node.listen_for_leadership()
