import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import gc
from web3 import Web3
from torch.utils.data import TensorDataset, DataLoader
from solid_integration import SolidTokenClient
from config import TOTAL_EPOCHS, TOTAL_RECIPES
import numpy as np

# Absolute import to avoid relative import errors
from config import TOTAL_EPOCHS 

torch.set_num_threads(1)

# --- Configuration ---
ANVIL_RPC_URL = "http://127.0.0.1:8545"
CONTRACT_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3" # Ensure this is your deployed address
ABI_PATH = "../../../swarm_orchestrator/out/SwarmCoordinator.sol/SwarmCoordinator.json"
GLOBAL_DIR = "../global"
PODS_DIR = "../pods" 

# --- Decentralized Top-N Model ---
class DecentralizedFoodRecommender(nn.Module):
    def __init__(self, num_recipes=TOTAL_RECIPES, embedding_dim=32):
        super(DecentralizedFoodRecommender, self).__init__()
        
        # PRIVATE: This client's personal taste vector. 
        # Stays trapped in the Solid Pod and is NEVER sent to the blockchain.
        self.my_personal_embedding = nn.Parameter(torch.randn(1, embedding_dim))
        
        # PUBLIC: The shared understanding of how ingredients relate.
        # This is the ONLY thing aggregated by the Swarm.
        self.recipe_embedding = nn.Embedding(num_recipes, embedding_dim)

    def forward(self, recipe_idx):
        # We only need the recipe index now, the user is intrinsic to the model
        r = self.recipe_embedding(recipe_idx)
        
        # Dot product between the user's private taste and the recipe
        scores = torch.sum(self.my_personal_embedding * r, dim=1)
        
        # Output shape [batch_size, 1] to match the BCELoss expected target shape
        return torch.sigmoid(scores).unsqueeze(1)

class SwarmNode:
    def __init__(self, client_id):
        self.client_id = client_id
        self.pod_dir = os.path.join(PODS_DIR, f"client_{client_id}")
        
        # Load profile and credentials
        with open(os.path.join(self.pod_dir, 'profile.json'), 'r') as f:
            self.profile = json.load(f)
            self.swarm_ids = self.profile['swarm_ids']
            self.wallet_address = self.profile['wallet_address']
            self.private_key = self.profile['private_key']
            
            # Extract Solid Credentials
            self.solid_url = self.profile.get('solid_pod_url', 'http://localhost:3000')
            self.solid_username = self.profile.get('solid_username', '')
            self.solid_token_id = self.profile.get('solid_token_id', '')
            self.solid_token_secret = self.profile.get('solid_token_secret', '')

        # Blockchain setup
        self.w3 = Web3(Web3.HTTPProvider(ANVIL_RPC_URL))
        with open(ABI_PATH, 'r') as f:
            self.contract = self.w3.eth.contract(
                address=CONTRACT_ADDRESS, 
                abi=json.load(f)['abi']
            )

        # ML setup
        self.model = DecentralizedFoodRecommender()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.BCELoss() 

    def sync_data_from_solid(self):
        """Authenticates with the Solid Pod and downloads the latest training data."""
        print(f"\n[Client {self.client_id}] Initiating Solid Pod Sync...")
        
        # Initialize our new auth client
        solid_client = SolidTokenClient(
            pod_url=self.solid_url,
            username=self.solid_username,
            token_id=self.solid_token_id,
            token_secret=self.solid_token_secret
        )
        
        # Attempt Authentication
        if solid_client.authenticate():
            # If successful, download the CSV files directly into the local edge pod directory
            common_dest = os.path.join(self.pod_dir, 'train_common.csv')
            solid_client.download_file("/swarm_data/train_common.csv", common_dest)
            
            vuln_dest = os.path.join(self.pod_dir, 'train_vulnerable.csv')
            solid_client.download_file("/swarm_data/train_vulnerable.csv", vuln_dest)
            
            test_dest = os.path.join(self.pod_dir, 'test.csv')
            solid_client.download_file("/swarm_data/test.csv", test_dest)
        else:
            print(f"⚠️ [Client {self.client_id}] Solid Auth failed. Falling back to local offline data if available.")

    def train_local_epoch(self):
        # 1. READ BLOCKCHAIN STATE: Should we exclude vulnerable data?
        exclude_vuln = self.contract.functions.excludeVulnerableData(self.wallet_address).call()
        status_msg = "EXCLUDING" if exclude_vuln else "INCLUDING"
        print(f"\n[Client {self.client_id}] Local training... Privacy Flag is {status_msg} vulnerable data.")

        # 2. LOCATE FILES
        common_path = os.path.join(self.pod_dir, 'train_common.csv')
        vuln_path = os.path.join(self.pod_dir, 'train_vulnerable.csv')
        test_path = os.path.join(self.pod_dir, 'test.csv')
        
        if not os.path.exists(common_path):
            print(f"[Client {self.client_id}] No train_common.csv found. Skipping.")
            return

        # 3. DYNAMICALLY BUILD THE DATAFRAME
        train_df = pd.read_csv(common_path)
        
        if not exclude_vuln and os.path.exists(vuln_path):
            # If the user permits it, concatenate the vulnerable data
            vuln_df = pd.read_csv(vuln_path)
            train_df = pd.concat([train_df, vuln_df], ignore_index=True)

        if train_df.empty:
            print(f"[Client {self.client_id}] Active training data is empty. Skipping.")
            return

        # 4. Build the "Forbidden List" to prevent Data Leakage
        global_seen_set = set(train_df['recipe_id'].values)
        
        # Make sure to forbid the vulnerable data even if we aren't training on it
        if exclude_vuln and os.path.exists(vuln_path):
            vuln_df = pd.read_csv(vuln_path)
            global_seen_set.update(vuln_df['recipe_id'].values)
            
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
            if not test_df.empty:
                global_seen_set.update(test_df['recipe_id'].values)

        # 5. Format Data
        pos_recipes = train_df['recipe_id'].values % TOTAL_RECIPES
        pos_labels = (train_df['rating'].values >= 4).astype(float)
        
        # 6. Generate Strict Negative Samples
        num_interactions = len(train_df)
        neg_recipes_list = []
        
        while len(neg_recipes_list) < num_interactions:
            rand_r = np.random.randint(0, TOTAL_RECIPES)
            if rand_r not in global_seen_set:
                neg_recipes_list.append(rand_r)
                
        neg_recipes = np.array(neg_recipes_list)
        neg_labels = np.zeros(num_interactions) # Label as 0
        
        # 7. Combine Real data with Fake data
        all_recipes = np.concatenate([pos_recipes, neg_recipes])
        all_labels = np.concatenate([pos_labels, neg_labels])
        
        recipes_tensor = torch.tensor(all_recipes, dtype=torch.long)
        labels_tensor = torch.tensor(all_labels, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(recipes_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        # 8. Training Loop
        self.model.train()
        total_loss = 0.0
        
        for batch_recipes, batch_labels in dataloader:
            self.optimizer.zero_grad()
            predictions = self.model(batch_recipes) 
            loss = self.criterion(predictions, batch_labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"[Client {self.client_id}] Local training complete. Avg Loss: {avg_loss:.4f}")

    def sign_and_send(self, tx):
        signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=self.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # THE FIX: Explicitly raise an error if the smart contract rejects the transaction
        if receipt.status == 0:
            raise Exception("Transaction reverted on-chain (Likely too late for this epoch).")
            
        return receipt

    def submit_to_swarm(self, swarm_id, epoch):
        # Strip out the private embedding BEFORE saving
        state_dict = self.model.state_dict()
        public_weights_only = {k: v for k, v in state_dict.items() if 'my_personal_embedding' not in k}
        
        weights_uri = os.path.join(self.pod_dir, f"weights_s{swarm_id}_e{epoch}.pt")
        torch.save(public_weights_only, weights_uri)

        print(f"[Client {self.client_id}] Submitting public recipe weights for Swarm {swarm_id}...")
        
        try:
            tx = self.contract.functions.submitWeights(swarm_id, epoch, weights_uri).build_transaction({
                'from': self.wallet_address,
                'nonce': self.w3.eth.get_transaction_count(self.wallet_address),
                'gas': 2000000,
                'gasPrice': self.w3.eth.gas_price
            })
            self.sign_and_send(tx)
        except Exception as e:
            print(f"⚠️ [Client {self.client_id}] Submission rejected (Likely too late for Epoch {epoch}). Dropping local weights and preparing to sync.")

    def wait_for_consensus(self, swarm_id, epoch):
        print(f"[Client {self.client_id}] Waiting for consensus on swarm {swarm_id}...")
        import re
        
        while True:
            swarm_data = self.contract.functions.swarms(swarm_id).call()
            current_leader = swarm_data[2]
            global_uri = swarm_data[3]     
            
            # Condition 1: We are the leader!
            if current_leader == self.wallet_address:
                self.execute_leader_duties(swarm_id, epoch)
                break 
                
            # Condition 2: Someone else was leader and updated the model
            elif global_uri:
                # Extract the epoch number from the URI string (e.g., "..._e5.pt" -> 5)
                match = re.search(r'_e(\d+)\.pt', global_uri)
                if match:
                    global_epoch = int(match.group(1))
                    
                    # If the network is AT or AHEAD of us, grab the latest weights to catch up!
                    if global_epoch >= epoch:
                        print(f"[Client {self.client_id}] Global model at Epoch {global_epoch} found. Downloading...")
                        global_state = torch.load(global_uri, weights_only=True)
                        
                        # strict=False tells PyTorch to only update the public recipe weights
                        self.model.load_state_dict(global_state, strict=False)
                        break 
            
            time.sleep(2)

    def execute_leader_duties(self, swarm_id, epoch):
        print(f"[Client {self.client_id}] ELECTED LEADER ✅ for Swarm {swarm_id}! Aggregating weights...")
        uris = self.contract.functions.getAllWeightsUris(swarm_id).call()
        
        # Federated Averaging (FedAvg)
        aggregated_state = None

        for uri in uris:
            state = torch.load(uri, weights_only=True)
            if aggregated_state is None:
                aggregated_state = state
            else:
                for key in aggregated_state.keys():
                    aggregated_state[key] += state[key]
            
            # Clear memory immediately
            del state      
            gc.collect()   
                    
        for key in aggregated_state.keys():
            aggregated_state[key] = torch.div(aggregated_state[key], len(uris))

        # Save global model
        global_uri = os.path.join(GLOBAL_DIR, f"global_model_s{swarm_id}_e{epoch}.pt")
        os.makedirs(GLOBAL_DIR, exist_ok=True)
        torch.save(aggregated_state, global_uri)

        # Submit back to blockchain
        print(f"[Client {self.client_id}] Submitting aggregated global model for Swarm {swarm_id}...")
        tx = self.contract.functions.submitGlobalModel(swarm_id, global_uri).build_transaction({
            'from': self.wallet_address,
            'nonce': self.w3.eth.get_transaction_count(self.wallet_address),
            'gas': 2000000,
            'gasPrice': self.w3.eth.gas_price
        })
        self.sign_and_send(tx)
        
        # strict=False ensures the leader doesn't overwrite its own private taste vector!
        self.model.load_state_dict(aggregated_state, strict=False)

    def run(self):
        # Step 1: Securely fetch the latest data from the Solid Pod
        self.sync_data_from_solid()

        global_swarm_id = 0

        # Step 2: Begin the Swarm Training Loop
        for epoch in range(TOTAL_EPOCHS):
            self.train_local_epoch()
            self.submit_to_swarm(global_swarm_id, epoch)
            self.wait_for_consensus(global_swarm_id, epoch)
            
        # Step 3: Save the full private brain to the pod at the end of training for evaluation
        full_local_path = os.path.join(self.pod_dir, 'local_full_model.pt')
        torch.save(self.model.state_dict(), full_local_path)
        print(f"🎉 [Client {self.client_id}] Finished all {TOTAL_EPOCHS} epochs successfully! Saved private local model to {full_local_path}")

if __name__ == "__main__":
    # In practice, this is handled by run_swarm.py mapping to different clients
    node = SwarmNode(client_id=1)
    node.run()