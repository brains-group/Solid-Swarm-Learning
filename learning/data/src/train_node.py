import os
import json
from pathlib import Path
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import gc
from web3 import Web3
from torch.utils.data import TensorDataset, DataLoader
from solid_integration import SolidTokenClient
from config import TOTAL_EPOCHS, TOTAL_RECIPES, ROUNDS_PER_EPOCH, TOP_N, NUM_NEGATIVE_SAMPLES
import numpy as np
import torch.nn.functional as F
import math
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

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
    def __init__(self, num_recipes=TOTAL_RECIPES, embedding_dim=64, num_highway_layers=3):
        super(DecentralizedFoodRecommender, self).__init__()
        
        # 1. Higher Capacity Embeddings
        self.my_personal_embedding = nn.Parameter(torch.randn(1, embedding_dim) * 0.01)
        self.recipe_embedding = nn.Embedding(num_recipes, embedding_dim)
        
        # 2. THE FIX: Recipe Bias (Popularity Variable)
        # This learnable parameter represents the "Global Baseline" for each recipe
        self.recipe_bias = nn.Embedding(num_recipes, 1)
        
        # Initialization: Xavier for embeddings, Zero for bias
        nn.init.xavier_uniform_(self.recipe_embedding.weight)
        nn.init.zeros_(self.recipe_bias.weight)

        # 3. Increased Network Depth
        input_dim = embedding_dim * 2
        self.highway_layers = nn.ModuleList([
            HighwayLayer(input_dim) for _ in range(num_highway_layers)
        ])
        
        self.output = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(0.3) # Slightly higher dropout for higher capacity

    def forward(self, recipe_idx):
        r = self.recipe_embedding(recipe_idx)
        # THE FIX: Fetch the independent bias/popularity score
        b = self.recipe_bias(recipe_idx).squeeze() 
        
        batch_size = r.size(0)
        u = self.my_personal_embedding.expand(batch_size, -1)
        
        # Dot product (Geometric similarity)
        dot_product = (u * r).sum(dim=1)
        
        # Highway Path (Feature interaction)
        x = torch.cat([u, r], dim=1)
        for layer in self.highway_layers:
            x = layer(x)
            x = self.dropout(x)
            
        mlp_score = self.output(x).squeeze()
        
        # THE FIX: Final Score = Interaction (MLP) + Similarity (Dot) + Popularity (Bias)
        # Return raw logits; apply sigmoid at inference/time of thresholding.
        return mlp_score + dot_product + b

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.2, gamma=2.0):
        super(FocalLoss, self).__init__()
        # alpha=0.2 weights the rare negative class heavier (since positives outnumber 4:1)
        # gamma=2.0 forces the network to focus on hard, misclassified examples
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Clamp inputs to prevent log(0) NaN crashes
        inputs = torch.clamp(inputs, min=1e-7, max=1.0 - 1e-7)
        
        # Standard BCE
        bce_loss = - (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        
        # Dynamic Focal Weights
        p_t = targets * inputs + (1 - targets) * (1 - inputs)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Final Focal Equation
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        return focal_loss.mean()

class SwarmNode:
    def __init__(self, client_id):
        self.client_id = client_id
        self.pod_dir = os.path.join(PODS_DIR, f"client_{client_id}")
        print(f"[Client {self.client_id}] Initializing client (pid={os.getpid()})...")
        
        # Load profile and credentials. Prefer local profile.json for backward
        # compatibility, but fall back to secret/user_accounts.json which is the
        # canonical source produced by `create_accounts.js` (Solid pods).
        self.profile = {}
        profile_path = os.path.join(self.pod_dir, 'profile.json')
        if os.path.exists(profile_path):
            try:
                with open(profile_path, 'r') as f:
                    self.profile = json.load(f)
            except Exception:
                print(f"⚠️ [Client {self.client_id}] Failed to read local profile.json, falling back to secret/user_accounts.json")

        # If some fields are missing, attempt to load Solid credentials from secret/user_accounts.json
        if not self.profile or 'webid' not in self.profile:
            # Resolve repository root and accounts path
            repo_root = Path(__file__).resolve().parents[3]
            accounts_path = repo_root / 'secret' / 'user_accounts.json'
            if accounts_path.exists():
                try:
                    with open(accounts_path, 'r') as f:
                        accounts = json.load(f)
                    username = f"user{client_id}"
                    acct = accounts.get(username, {})
                    # Map fields from create_accounts.js structure
                    self.profile.update({
                        'webid': acct.get('webid', ''),
                        'pod': acct.get('pod', ''),
                        'email': acct.get('email', ''),
                        'client_credentials_token_identifier': acct.get('client_credentials_token_identifier', ''),
                        'client_credentials_token_secret': acct.get('client_credentials_token_secret', ''),
                    })
                except Exception as e:
                    print(f"⚠️ [Client {self.client_id}] Could not read user_accounts.json: {e}")

        # Swarm-related fields (may be absent for pods-only setup)
        self.swarm_ids = self.profile.get('swarm_ids', [])
        self.wallet_address = self.profile.get('wallet_address', '')
        self.private_key = self.profile.get('private_key', '')

        # Extract Solid Credentials (canonical names)
        self.solid_url = self.profile.get('pod', self.profile.get('solid_pod_url', 'http://localhost:3000'))
        # create_accounts uses webid/email as username; fall back cleanly
        self.solid_username = self.profile.get('email', self.profile.get('webid', ''))
        self.solid_token_id = self.profile.get('client_credentials_token_identifier', self.profile.get('solid_token_id', ''))
        self.solid_token_secret = self.profile.get('client_credentials_token_secret', self.profile.get('solid_token_secret', ''))

        # Warn if blockchain credentials are missing; training can proceed but
        # submission/signing will be skipped or fail later unless provided.
        if not self.wallet_address or not self.private_key:
            print(f"⚠️ [Client {self.client_id}] Wallet credentials missing. On-chain submission may fail for this client.")

        # Blockchain setup
        self.w3 = Web3(Web3.HTTPProvider(ANVIL_RPC_URL))
        with open(ABI_PATH, 'r') as f:
            self.contract = self.w3.eth.contract(
                address=CONTRACT_ADDRESS, 
                abi=json.load(f)['abi']
            )

        # ML setup
        self.model = DecentralizedFoodRecommender()
        # Reduce LR for stability
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-5)
        self.criterion = FocalLoss(alpha=0.05, gamma=2.0)

        # Signal that initialization finished (helps diagnose hangs during model construction)
        try:
            print(f"[Client {self.client_id}] Initialized model and environment (pid={os.getpid()}).")
        except Exception:
            pass

        # Immediately attempt to sync training data from the user's Solid Pod.
        try:
            self.sync_data_from_solid()
        except Exception as e:
            print(f"⚠️ [Client {self.client_id}] Error during Solid sync: {e}")

    def sync_data_from_solid(self):
        """Authenticates with the Solid Pod and downloads the latest training data."""
        # Stagger auth attempts slightly to avoid thundering herd against the Solid server
        time.sleep(random.uniform(0.0, 0.5))
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
            # Report which files were written (or missing)
            for fname in ('train_common.csv', 'train_vulnerable.csv', 'test.csv'):
                p = os.path.join(self.pod_dir, fname)
                if os.path.exists(p):
                    print(f"[Client {self.client_id}] Synced: {fname}")
                else:
                    print(f"[Client {self.client_id}] Missing after sync: {fname}")
        else:
            print(f"⚠️ [Client {self.client_id}] Solid Auth failed. Falling back to local offline data if available.")

    def train_local_epoch(self):
        # 1. READ BLOCKCHAIN STATE: Should we exclude vulnerable data?
        try:
            exclude_vuln = self.contract.functions.excludeVulnerableData(self.wallet_address).call()
        except Exception as e:
            print(f"⚠️ [Client {self.client_id}] RPC Error fetching privacy preference, defaulting to True: {e}")
            exclude_vuln = True
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

        # 6. Generate negative samples to create a balanced dataset
        pos_count = len(pos_recipes)
        num_negatives = pos_count if pos_count > 0 else 1

        all_possible = np.arange(TOTAL_RECIPES)
        seen_array = np.array(list(global_seen_set)) if len(global_seen_set) > 0 else np.array([], dtype=int)
        valid_pool = np.setdiff1d(all_possible, seen_array)

        if valid_pool.size == 0:
            # fallback to random sampling with replacement across all recipes
            neg_recipes = np.random.randint(0, TOTAL_RECIPES, size=num_negatives)
        else:
            replace = valid_pool.size < num_negatives
            neg_recipes = np.random.choice(valid_pool, size=num_negatives, replace=replace)

        neg_labels = np.zeros(len(neg_recipes), dtype=float)

        # 7. Combine positive and negative samples and shuffle
        all_recipes = np.concatenate([pos_recipes, neg_recipes])
        all_labels = np.concatenate([pos_labels, neg_labels])
        perm = np.random.permutation(len(all_recipes))
        all_recipes = all_recipes[perm]
        all_labels = all_labels[perm]

        recipes_tensor = torch.tensor(all_recipes, dtype=torch.long)
        labels_tensor = torch.tensor(all_labels, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(recipes_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        # 8. Training Loop
        num_batches = max(1, math.ceil(len(dataset) / 64))
        print(f"[Client {self.client_id}] Prepared training data: {len(dataset)} samples, {num_batches} batches.")
        self.model.train()
        total_loss = 0.0

        # 9. Take a snapshot of the global weights BEFORE the batch loop starts
        global_weights = {name: param.clone().detach() for name, param in self.model.named_parameters()}
        mu = 0.01 # FedProx penalty hyperparameter (kept small)

        # Use BCEWithLogitsLoss for numerical stability with logits output
        pos = float(np.sum(all_labels == 1.0))
        neg = float(np.sum(all_labels == 0.0))

        pos_weight = 1.25
        if pos <= 0.0:
            pos_weight = torch.tensor(pos_weight)
        else:
            pos_weight = torch.tensor(max(1.0, neg / pos) * pos_weight)

        local_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Precompute normalization factor for proximal term
        total_param_count = sum(p.numel() for name, p in self.model.named_parameters() if 'my_personal_embedding' not in name)

        for batch_recipes, batch_labels in dataloader:
            self.optimizer.zero_grad()
            predictions = self.model(batch_recipes).unsqueeze(1)
            base_loss = local_criterion(predictions, batch_labels)

            # Calculate the FedProx Penalty (squared L2 Norm), normalized
            proximal_term = 0.0
            for name, param in self.model.named_parameters():
                if 'my_personal_embedding' not in name:
                    gw = global_weights[name].to(param.device)
                    proximal_term = proximal_term + torch.sum((param - gw) ** 2)

            if total_param_count > 0:
                proximal_term = proximal_term / float(total_param_count)

            # Add the penalty to the base loss
            loss = base_loss + (mu / 2.0) * proximal_term

            loss.backward()
            #Prevent gradient explosion in the Highway Layers
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"\n[Client {self.client_id}] Local training complete. Avg Loss: {avg_loss:.4f}")

        return avg_loss

    def sign_and_send(self, tx):
        signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=self.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # THE FIX: Explicitly raise an error if the smart contract rejects the transaction
        if receipt.status == 0:
            raise Exception(f"Transaction reverted on-chain! Gas used: {receipt.gasUsed}. If it hit the limit, increase the gas.")
            
        return receipt

    def submit_to_swarm(self, swarm_id, epoch):
        # Fetch the user's DP Budget from the Smart Contract
        try:
            scaled_dp_budget = self.contract.functions.dpBudgets(self.wallet_address).call()
        except Exception as e:
            print(f"⚠️ [Client {self.client_id}] RPC Error fetching DP budget, defaulting to 0: {e}")
            scaled_dp_budget = 0
        
        # Strip out the private embedding BEFORE saving
        state_dict = self.model.state_dict()
        public_weights_only = {k: v for k, v in state_dict.items() if 'my_personal_embedding' not in k}
        
        # Apply Local Differential Privacy (Continuous Epsilon)
        if scaled_dp_budget > 0:
            # Reverse the Solidity integer scaling back to a float
            epsilon = scaled_dp_budget / 100.0
            
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
            base_tx = {
                'from': self.wallet_address,
                'nonce': self.w3.eth.get_transaction_count(self.wallet_address),
                'gasPrice': self.w3.eth.gas_price
            }
            # Automatically estimate gas required based on current swarm state and add 20% safety buffer
            gas_estimate = self.contract.functions.submitWeights(swarm_id, epoch, weights_uri).estimate_gas(base_tx)
            base_tx['gas'] = int(gas_estimate * 1.2)
            
            tx = self.contract.functions.submitWeights(swarm_id, epoch, weights_uri).build_transaction(base_tx)
            receipt = self.sign_and_send(tx)
            try:
                print(f"[Client {self.client_id}] Submission receipt: {receipt.transactionHash.hex()}")
            except Exception:
                print(f"[Client {self.client_id}] Submission sent (no receipt details).")
        except Exception as e:
            print(f"⚠️ [Client {self.client_id}] Submission rejected (Likely too late for Epoch {epoch}): {e}. Dropping local weights and preparing to sync.")

    def wait_for_consensus(self, swarm_id, epoch):
        print(f"[Client {self.client_id}] Waiting for consensus on swarm {swarm_id} (local epoch {epoch})...")
        import re
        
        while True:
            try:
                swarm_data = self.contract.functions.swarms(swarm_id).call()
                current_leader = swarm_data[2]
                global_uri = swarm_data[3]     
            except Exception as e:
                print(f"⚠️ [Client {self.client_id}] RPC Error polling consensus: {e}")
                time.sleep(random.uniform(1.0, 3.0))
                continue
            
            # Condition 1: We are the leader!
            if current_leader == self.wallet_address:
                print(f"[Client {self.client_id}] I am the leader for epoch {epoch} (wallet {self.wallet_address}).")
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

                        # --- THE FIX: Replaced File Tokens with safe Jitter ---
                        # Prevent all nodes from trying to read the `.pt` file simultaneously
                        time.sleep(random.uniform(0.1, 3.0))

                        try:
                            global_state = torch.load(global_uri, weights_only=True)
                            # strict=False tells PyTorch to only update the public recipe weights
                            self.model.load_state_dict(global_state, strict=False)
                            # FAST-FORWARD: Return the global epoch + 1 so the local loop catches up instantly!
                            return global_epoch + 1
                        except Exception as load_err:
                            print(f"⚠️ [Client {self.client_id}] Failed to load global model: {load_err}. Retrying...")
            
            time.sleep(2)

    def execute_leader_duties(self, swarm_id, epoch):
        print(f"[Client {self.client_id}] ELECTED LEADER ✅ for Swarm {swarm_id}! Aggregating weights...")
        try:
            uris = self.contract.functions.getAllWeightsUris(swarm_id).call()
        except Exception as e:
            print(f"⚠️ [Client {self.client_id}] Leader RPC Error fetching URIs: {e}")
            return
            
        # Filter to only current epoch to prevent memory leaks if contract doesn't clear
        uris = [uri for uri in uris if f"_e{epoch}.pt" in uri]
        if not uris:
            print(f"⚠️ [Client {self.client_id}] No valid URIs found for epoch {epoch}. Skipping aggregation.")
            return
        
        # Federated Averaging (FedAvg)
        aggregated_state = None

        for uri in uris:
            try:
                state = torch.load(uri, weights_only=True)
                if aggregated_state is None:
                    aggregated_state = state
                else:
                    for key in aggregated_state.keys():
                        aggregated_state[key] += state[key]
                
                # Clear memory immediately
                del state      
                gc.collect()   
            except Exception as e:
                print(f"⚠️ [Client {self.client_id}] Leader failed to read client weight file at {uri}: {e}")
                continue
                    
        for key in aggregated_state.keys():
            aggregated_state[key] = torch.div(aggregated_state[key], len(uris))

        # Save global model
        global_uri = os.path.join(GLOBAL_DIR, f"global_model_s{swarm_id}_e{epoch}.pt")
        os.makedirs(GLOBAL_DIR, exist_ok=True)
        torch.save(aggregated_state, global_uri)

        # Submit back to blockchain
        print(f"[Client {self.client_id}] Submitting aggregated global model for Swarm {swarm_id}...")
        try:
            base_tx = {
                'from': self.wallet_address,
                'nonce': self.w3.eth.get_transaction_count(self.wallet_address),
                'gasPrice': self.w3.eth.gas_price
            }
            # Estimate gas dynamically (especially important for the leader loop that resets all node states)
            gas_estimate = self.contract.functions.submitGlobalModel(swarm_id, global_uri).estimate_gas(base_tx)
            base_tx['gas'] = int(gas_estimate * 1.2)
            
            tx = self.contract.functions.submitGlobalModel(swarm_id, global_uri).build_transaction(base_tx)
            self.sign_and_send(tx)
        except Exception as e:
            print(f"⚠️ [Client {self.client_id}] Failed to submit global model: {e}")
        
        # strict=False ensures the leader doesn't overwrite its own private taste vector!
        self.model.load_state_dict(aggregated_state, strict=False)

    def evaluate(self, epoch):
        test_path = os.path.join(self.pod_dir, 'test.csv')
        train_path = os.path.join(self.pod_dir, 'train_common.csv')
        vuln_path = os.path.join(self.pod_dir, 'train_vulnerable.csv')
        
        if not os.path.exists(test_path):
            return
            
        test_df = pd.read_csv(test_path)
        if test_df.empty:
            return
            
        train_df = pd.read_csv(train_path) if os.path.exists(train_path) else pd.DataFrame(columns=['recipe_id'])
        
        try:
            exclude_vuln = self.contract.functions.excludeVulnerableData(self.wallet_address).call()
        except Exception:
            exclude_vuln = True

        if not exclude_vuln and os.path.exists(vuln_path):
            vuln_df = pd.read_csv(vuln_path)
            train_df = pd.concat([train_df, vuln_df], ignore_index=True)
            
        self.model.eval()
        
        # =========================================================
        # PHASE 1: Classification Evaluation (Thresholding) ONLY
        # =========================================================
        recipes_tensor = torch.tensor(test_df['recipe_id'].values % TOTAL_RECIPES, dtype=torch.long)
        y_true = (test_df['rating'].values >= 4).astype(float)
        
        with torch.no_grad():
            logits = self.model(recipes_tensor).squeeze()
            probs = torch.sigmoid(logits).numpy()
            if probs.ndim == 0:
                probs = np.expand_dims(probs, 0)
            y_pred = (probs >= 0.5).astype(float)
            
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'epoch': epoch,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
            # Notice we removed hits, ndcg, and total_ranking_tests
        }
        
        metrics_path = os.path.join(self.pod_dir, 'metrics.csv')
        df = pd.DataFrame([metrics])
        if not os.path.exists(metrics_path):
            df.to_csv(metrics_path, index=False)
        else:
            df.to_csv(metrics_path, mode='a', header=False, index=False)

        print(f"   [Client {self.client_id}] Evaluated Epoch {epoch} and saved metrics to pod.")
        # --- ROC-AUC Curve (classification) ---
        try:
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, probs)
                fpr, tpr, _ = roc_curve(y_true, probs)
                roc_path = os.path.join(self.pod_dir, f"roc_epoch_{epoch}.png")
                plt.figure()
                plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
                plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - Client {self.client_id} Epoch {epoch}')
                plt.legend(loc='lower right')
                plt.tight_layout()
                plt.savefig(roc_path)
                plt.close()
                print(f"   [Client {self.client_id}] ROC-AUC: {auc:.4f} saved to {roc_path}")
        except Exception as e:
            print(f"   [Client {self.client_id}] Could not compute ROC-AUC: {e}")

    def evaluate_final_ranking(self):
        """Runs the heavy Top-N evaluation using Actual Negatives first, then padding with Randoms."""
        print(f"   [Client {self.client_id}] Running Final Top-{TOP_N} Ranking Evaluation...")
        test_path = os.path.join(self.pod_dir, 'test.csv')
        train_path = os.path.join(self.pod_dir, 'train_common.csv')
        
        if not os.path.exists(test_path):
            return
            
        test_df = pd.read_csv(test_path)
        if test_df.empty:
            return
            
        train_df = pd.read_csv(train_path) if os.path.exists(train_path) else pd.DataFrame(columns=['recipe_id'])
        self.model.eval()
        
        # 1. Build the sets of what to include/exclude
        seen_recipes = set(test_df['recipe_id'].values % TOTAL_RECIPES).union(set(train_df['recipe_id'].values % TOTAL_RECIPES))
        
        # Split test set into Actual Positives and Actual Negatives
        liked_test_df = test_df[test_df['rating'] >= 4]
        disliked_test_df = test_df[test_df['rating'] < 4]
        
        # Extract unique actual negative IDs
        actual_test_negatives = np.unique(disliked_test_df['recipe_id'].values % TOTAL_RECIPES)
        
        client_hits = 0
        client_ndcg = 0
        total_ranking_tests = len(liked_test_df)
        
        if total_ranking_tests == 0:
            return

        # 2. Build the backup pool of Random Negatives (Recipes never seen)
        all_possible_recipes = np.arange(TOTAL_RECIPES)
        seen_array = np.array(list(seen_recipes))
        valid_negatives_pool = np.setdiff1d(all_possible_recipes, seen_array)

        with torch.no_grad():
            for index, row in liked_test_df.iterrows():
                true_id = int(row['recipe_id'] % TOTAL_RECIPES)
                true_recipe = torch.tensor([true_id], dtype=torch.long)

                # 3. Smart Negative Sampling Logic
                if len(actual_test_negatives) >= NUM_NEGATIVE_SAMPLES:
                    # Scenario A: We have enough Actual Negatives in the test set
                    fake_recipes_array = np.random.choice(actual_test_negatives, NUM_NEGATIVE_SAMPLES, replace=False)
                else:
                    # Scenario B: Take ALL Actual Negatives, then pad with Random Negatives
                    fake_from_test = actual_test_negatives
                    shortfall = NUM_NEGATIVE_SAMPLES - len(actual_test_negatives)
                    
                    if valid_negatives_pool.size >= shortfall:
                        fake_from_pool = np.random.choice(valid_negatives_pool, shortfall, replace=False)
                    else:
                        # Extreme edgecase: Almost no valid pool left (highly unlikely with 500k recipes)
                        reps = int(np.ceil(shortfall / max(1, valid_negatives_pool.size)))
                        fake_from_pool = np.tile(valid_negatives_pool, reps)[:shortfall]
                        
                    fake_recipes_array = np.concatenate([fake_from_test, fake_from_pool])

                fake_recipes = torch.tensor(fake_recipes_array, dtype=torch.long)
                all_recipes = torch.cat([true_recipe, fake_recipes])

                # 4. Scoring and Ranking
                scores = torch.sigmoid(self.model(all_recipes)).squeeze().numpy()
                
                if scores.ndim == 0:
                    scores = np.expand_dims(scores, 0)
                    
                ranked_indices = scores.argsort()[::-1]
                rank_of_true_item = (ranked_indices == 0).nonzero()[0][0]

                if rank_of_true_item < TOP_N:
                    client_hits += 1
                    client_ndcg += 1.0 / math.log2(rank_of_true_item + 2)
                    
        ranking_metrics = {
            'hits': int(client_hits),
            'ndcg': float(client_ndcg),
            'total_ranking_tests': int(total_ranking_tests)
        }
        
        pd.DataFrame([ranking_metrics]).to_csv(os.path.join(self.pod_dir, 'final_ranking.csv'), index=False)
        print(f"   [Client {self.client_id}] Final ranking metrics saved (Used {len(actual_test_negatives)} actual negatives per test).")

    def run(self):
        self.sync_data_from_solid()

        # Remove old metrics to start fresh
        metrics_path = os.path.join(self.pod_dir, 'metrics.csv')
        ranking_path = os.path.join(self.pod_dir, 'final_ranking.csv')
        if os.path.exists(metrics_path): os.remove(metrics_path)
        if os.path.exists(ranking_path): os.remove(ranking_path)

        patience = 2
        min_delta = 0.0001
        global_swarm_id = 0
        # Allow resuming from environment variable START_EPOCH (exported by master_swarm_fix.sh)
        try:
            env_start = os.environ.get('START_EPOCH', None)
            current_epoch = int(env_start) if env_start is not None else 0
        except Exception:
            current_epoch = 0

        if current_epoch >= TOTAL_EPOCHS:
            print(f"[Client {self.client_id}] START_EPOCH={current_epoch} >= TOTAL_EPOCHS={TOTAL_EPOCHS}. Nothing to run.")
            return
        
        while current_epoch < TOTAL_EPOCHS:
            best_loss = float('inf')
            patience_counter = 0
            
            for local_step in range(ROUNDS_PER_EPOCH): 
                avg_loss = self.train_local_epoch()
                
                if avg_loss is None: 
                    break 
                
                if avg_loss < (best_loss - min_delta):
                    best_loss = avg_loss
                    patience_counter = 0 
                else:
                    patience_counter += 1
                    print(f"   [Client {self.client_id}] Loss stagnated. Patience: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        print(f"   [Client {self.client_id}] 🛑 Early stopping triggered at local step {local_step + 1}!")
                        patience_counter = 0
                        break
                        
            self.submit_to_swarm(global_swarm_id, current_epoch)
            new_epoch = self.wait_for_consensus(global_swarm_id, current_epoch)
            
            self.evaluate(new_epoch - 1)
            current_epoch = new_epoch
            
        full_local_path = os.path.join(self.pod_dir, 'local_full_model.pt')
        torch.save(self.model.state_dict(), full_local_path)
        
        # --- THE FIX: Run Phase 2 ONLY when training finishes ---
        self.evaluate_final_ranking()
        
        print(f"🎉 [Client {self.client_id}] Finished all {TOTAL_EPOCHS} epochs successfully! Saved private local model.")

if __name__ == "__main__":
    # In practice, this is handled by run_swarm.py mapping to different clients
    node = SwarmNode(client_id=1)
    node.run()