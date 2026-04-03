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
import torch.nn.functional as F

# Absolute import to avoid relative import errors
from config import TOTAL_EPOCHS 

torch.set_num_threads(1)

# --- Configuration ---
ANVIL_RPC_URL = "http://127.0.0.1:8545"
CONTRACT_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3" # Ensure this is your deployed address
ABI_PATH = "../../../swarm_orchestrator/out/SwarmCoordinator.sol/SwarmCoordinator.json"
GLOBAL_DIR = "../global"
PODS_DIR = "../pods" 

# --- Decentralized Top-N Model that implements Highway Networks---
class HighwayLayer(nn.Module):
    def __init__(self, size, gate_bias=-1.0):
        super(HighwayLayer, self).__init__()
        # H(x) - The non-linear transformation
        self.transform = nn.Linear(size, size)
        # T(x) - The transform gate
        self.gate = nn.Linear(size, size)
        
        # Pro-Tip: Initialize the gate bias negatively so the layer 
        # initially behaves like a simple pass-through (carry) connection.
        nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, x):
        h = F.relu(self.transform(x))
        t = torch.sigmoid(self.gate(x))
        c = 1.0 - t  # Carry gate
        
        # Output is a dynamic blend of the transformed data and the raw input
        return h * t + x * c

class DecentralizedFoodRecommender(nn.Module):
    def __init__(self, num_recipes=TOTAL_RECIPES, embedding_dim=32, num_highway_layers=2):
        super(DecentralizedFoodRecommender, self).__init__()
        
        # 1. The Embeddings (Still using 0.1 scaling to prevent gradient death!)
        self.my_personal_embedding = nn.Parameter(torch.randn(1, embedding_dim) * 0.1)
        self.recipe_embedding = nn.Embedding(num_recipes, embedding_dim)
        self.recipe_embedding.weight.data.normal_(0, 0.1)

        # 2. Highway Networks require input and output dimensions to match.
        # Since we concatenate User (32) + Recipe (32), our dimension is 64.
        input_dim = embedding_dim * 2
        
        # ModuleList ensures PyTorch registers the parameters of our custom layers
        self.highway_layers = nn.ModuleList([
            HighwayLayer(input_dim) for _ in range(num_highway_layers)
        ])
        
        # 3. The Final Scoring Layer
        self.output = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, recipe_idx):
        # Fetch global recipe embeddings
        r = self.recipe_embedding(recipe_idx)
        
        # Expand the local user's private embedding to match batch size
        batch_size = r.size(0)
        u = self.my_personal_embedding.expand(batch_size, -1)
        
        # Concatenate: [batch_size, 64]
        x = torch.cat([u, r], dim=1)
        
        # Route through the Highway architecture
        for layer in self.highway_layers:
            x = layer(x)
            x = self.dropout(x)
            
        scores = self.output(x)
        return torch.sigmoid(scores)

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
        print(f"[Client {self.client_id}] Local training... Privacy Flag is {status_msg} vulnerable data.")

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

        # 9. Take a snapshot of the global weights BEFORE the batch loop starts
        global_weights = {name: param.clone().detach() for name, param in self.model.named_parameters()}
        mu = 0.01 # FedProx penalty hyperparameter
        
        for batch_recipes, batch_labels in dataloader:
            self.optimizer.zero_grad()
            predictions = self.model(batch_recipes) 
            base_loss = self.criterion(predictions, batch_labels)
            
            # Calculate the FedProx Penalty (L2 Norm)
            proximal_term = 0.0
            for name, param in self.model.named_parameters():
                # Only penalize the public recipe weights, not the private user embedding!
                if 'my_personal_embedding' not in name:
                    proximal_term += (param - global_weights[name]).norm(2)
            
            # Add the penalty to the base loss
            loss = base_loss + (mu / 2) * proximal_term

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"\n[Client {self.client_id}] Local training complete. Avg Loss: {avg_loss:.4f}")

    def sign_and_send(self, tx):
        signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=self.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # THE FIX: Explicitly raise an error if the smart contract rejects the transaction
        if receipt.status == 0:
            raise Exception("Transaction reverted on-chain (Likely too late for this epoch).")
            
        return receipt

    def submit_to_swarm(self, swarm_id, epoch):
        # Fetch the user's DP Budget from the Smart Contract
        scaled_dp_budget = self.contract.functions.dpBudgets(self.wallet_address).call()
        
        # Strip out the private embedding BEFORE saving
        state_dict = self.model.state_dict()
        public_weights_only = {k: v for k, v in state_dict.items() if 'my_personal_embedding' not in k}
        
        # Apply Local Differential Privacy (Continuous Epsilon)
        if scaled_dp_budget > 0:
            # Reverse the Solidity integer scaling back to a float
            epsilon = scaled_dp_budget / 100.0
            #print(f"   [Client {self.client_id}] 🛡️ Applying DP Noise (Epsilon: {epsilon})...")
            
            # The standard deviation is inversely proportional to epsilon.
            # (Assuming a base sensitivity constant of 0.1 for the clipping bound)
            std_dev = 0.1 / epsilon 
            
            for key in public_weights_only.keys():
                noise = torch.randn_like(public_weights_only[key]) * std_dev
                public_weights_only[key] += noise

        # Save and Submit
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
                return epoch + 1 # Move to the next epoch normally
                
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
                        # FAST-FORWARD: Return the global epoch + 1 so the local loop catches up instantly!
                        return global_epoch + 1 
            
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
        # We changed this from a 'for' loop to a dynamic 'while' loop
        current_epoch = 0
        while current_epoch < TOTAL_EPOCHS:
            # train harder locally to survive global dilution
            #print(f"\n--- Starting Local Epochs for Global Round {epoch} ---")
            for local_step in range(3): 
                self.train_local_epoch()
            self.submit_to_swarm(global_swarm_id, current_epoch)
            # The node syncs, and the function returns the NEW epoch it should jump to
            current_epoch = self.wait_for_consensus(global_swarm_id, current_epoch)
            
        # Step 3: Save the full private brain to the pod at the end of training for evaluation
        full_local_path = os.path.join(self.pod_dir, 'local_full_model.pt')
        torch.save(self.model.state_dict(), full_local_path)
        print(f"🎉 [Client {self.client_id}] Finished all {TOTAL_EPOCHS} epochs successfully! Saved private local model to {full_local_path}")

if __name__ == "__main__":
    # In practice, this is handled by run_swarm.py mapping to different clients
    node = SwarmNode(client_id=1)
    node.run()