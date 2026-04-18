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
from torch.utils.data import TensorDataset, DataLoader, Dataset
from solid_integration import SolidTokenClient
from config import TOTAL_EPOCHS, TOTAL_RECIPES, ROUNDS_PER_EPOCH, TOP_N, NUM_NEGATIVE_SAMPLES, BATCH_SIZE_GLOBAL, EMBEDDING_DIM
import numpy as np
import torch.nn.functional as F
import math
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

torch.set_num_threads(1)

# --- Configuration ---
ANVIL_RPC_URL = "http://127.0.0.1:8545"
CONTRACT_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3" 
ABI_PATH = "../../../swarm_orchestrator/out/SwarmCoordinator.sol/SwarmCoordinator.json"
GLOBAL_DIR = "../global"
PODS_DIR = "../pods" 


GENRE_MAP = {
    'Action': 1, 'Adventure': 2, 'Animation': 3, 'Children': 4, 
    'Comedy': 5, 'Crime': 6, 'Documentary': 7, 'Drama': 8, 'Fantasy': 9, 
    'Film-Noir': 10, 'Horror': 11, 'Musical': 12, 'Mystery': 13, 'Romance': 14, 
    'Sci-Fi': 15, 'Thriller': 16, 'War': 17, 'Western': 18, 'IMAX': 19, 
    '(no genres listed)': 0, '': 0
}

def genre_collate_fn(batch):
    recipes = []
    labels = []
    all_genres = []
    offsets = [0]
    
    for recipe_id, label, genre_list in batch:
        recipes.append(recipe_id)
        labels.append(label)
        all_genres.extend(genre_list)
        offsets.append(len(all_genres))
        
    offsets = torch.tensor(offsets[:-1], dtype=torch.long)
    recipes = torch.tensor(recipes, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    
    genres_tensor = torch.tensor(all_genres, dtype=torch.long) if all_genres else torch.empty(0, dtype=torch.long)
    
    return recipes, genres_tensor, offsets, labels

class RecipeGenreDataset(Dataset):
    def __init__(self, recipes, labels, genres):
        self.recipes = recipes
        self.labels = labels
        self.genres = genres 
        
    def __len__(self):
        return len(self.recipes)
        
    def __getitem__(self, idx):
        return self.recipes[idx], self.labels[idx], self.genres[idx]

class HighwayLayer(nn.Module):
    def __init__(self, size, gate_bias=-1.0):
        super(HighwayLayer, self).__init__()
        self.transform = nn.Linear(size, size)
        self.gate = nn.Linear(size, size)
        nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, x):
        h = F.relu(self.transform(x))
        t = torch.sigmoid(self.gate(x))
        c = 1.0 - t 
        return h * t + x * c

class DecentralizedFoodRecommender(nn.Module):
    def __init__(self, num_recipes=TOTAL_RECIPES, embedding_dim=EMBEDDING_DIM, num_highway_layers=3):
        super(DecentralizedFoodRecommender, self).__init__()
        
        self.my_personal_embedding = nn.Parameter(torch.randn(1, embedding_dim) * 0.01)
        self.recipe_embedding = nn.Embedding(num_recipes, embedding_dim)
        self.genre_embedding = nn.EmbeddingBag(25, embedding_dim, mode='mean') 
        self.recipe_bias = nn.Embedding(num_recipes, 1)
        
        nn.init.kaiming_uniform_(self.recipe_embedding.weight)
        nn.init.zeros_(self.recipe_bias.weight)

        # NeuMF Width: User Embed + Recipe Embed + Element-wise Interaction
        input_dim = embedding_dim * 3
        self.highway_layers = nn.ModuleList([
            HighwayLayer(input_dim) for _ in range(num_highway_layers)
        ])
        
        self.output = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, recipe_idx, genre_indices, genre_offsets):
        r_pre = self.recipe_embedding(recipe_idx)
        g = self.genre_embedding(genre_indices, genre_offsets)
        r = r_pre + g # Content-injected recipe embedding

        b = self.recipe_bias(recipe_idx).squeeze(-1) 
        batch_size = r.size(0)
        u = self.my_personal_embedding.expand(batch_size, -1)
        
        # Matrix Factorization (Dot Product)
        dot_product = (u * r).sum(dim=1)
        
        # Neural Collaborative Filtering Path
        dot_interaction = u * r 
        x = torch.cat([u, r, dot_interaction], dim=1) 
        
        x = self.dropout(x)
        for layer in self.highway_layers:
            x = layer(x)
            
        mlp_score = self.output(x).squeeze(-1)
        return mlp_score + dot_product + b

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=1.0):
        super(WeightedBCELoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, inputs, targets):
        return self.criterion(inputs, targets)

class SwarmNode:
    def __init__(self, client_id):
        self.client_id = client_id
        self.pod_dir = os.path.join(PODS_DIR, f"client_{client_id}")
        
        self.profile = {}
        profile_path = os.path.join(self.pod_dir, 'profile.json')
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                self.profile = json.load(f)

        if not self.profile or 'webid' not in self.profile:
            repo_root = Path(__file__).resolve().parents[3]
            accounts_path = repo_root / 'secret' / 'user_accounts.json'
            if accounts_path.exists():
                with open(accounts_path, 'r') as f:
                    accounts = json.load(f)
                username = f"user{client_id}"
                acct = accounts.get(username, {})
                self.profile.update({
                    'webid': acct.get('webid', ''),
                    'pod': acct.get('pod', ''),
                    'email': acct.get('email', ''),
                    'client_credentials_token_identifier': acct.get('client_credentials_token_identifier', ''),
                    'client_credentials_token_secret': acct.get('client_credentials_token_secret', ''),
                })

        self.swarm_ids = self.profile.get('swarm_ids', [])
        self.wallet_address = self.profile.get('wallet_address', '')
        self.private_key = self.profile.get('private_key', '')

        self.solid_url = self.profile.get('pod', self.profile.get('solid_pod_url', 'http://localhost:3000'))
        self.solid_username = self.profile.get('email', self.profile.get('webid', ''))
        self.solid_token_id = self.profile.get('client_credentials_token_identifier', self.profile.get('solid_token_id', ''))
        self.solid_token_secret = self.profile.get('client_credentials_token_secret', self.profile.get('solid_token_secret', ''))

        self.w3 = Web3(Web3.HTTPProvider(ANVIL_RPC_URL))
        with open(ABI_PATH, 'r') as f:
            self.contract = self.w3.eth.contract(
                address=CONTRACT_ADDRESS, 
                abi=json.load(f)['abi']
            )

        self.model = DecentralizedFoodRecommender()
        # THE FIX: Apply weight decay ONLY to the Highway and Output layers to prevent runaway positive logits
        self.optimizer = optim.Adam([
            {'params': self.model.my_personal_embedding, 'weight_decay': 0.0},
            {'params': self.model.recipe_embedding.parameters(), 'weight_decay': 0.0},
            {'params': self.model.genre_embedding.parameters(), 'weight_decay': 0.0},
            {'params': self.model.recipe_bias.parameters(), 'weight_decay': 0.0},
            {'params': self.model.highway_layers.parameters(), 'weight_decay': 1e-4},
            {'params': self.model.output.parameters(), 'weight_decay': 1e-4}
        ], lr=5e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-5)

        # Removed sync_data_from_solid() from __init__ to prevent the thundering herd lockup.
        print(f"[Client {self.client_id}] Initialized model and environment (pid={os.getpid()}).")

    def sync_data_from_solid(self):
        for fname in ('train_common.csv', 'train_vulnerable.csv', 'test.csv'):
            p = os.path.join(self.pod_dir, fname)
            if os.path.exists(p):
                os.remove(p)

        # 2. Deterministic Modulo Scaffolding
        # We group the 150 clients into 15 buckets.
        # Clients 1, 16, 31... are in Bucket 1 (Wait 2.0s)
        # Clients 2, 17, 32... are in Bucket 2 (Wait 4.0s)
        # This guarantees exactly 10 clients wake up every 2 seconds.
        bucket = self.client_id % 15
        
        # Multiply the bucket by a 2.0 second spacing interval
        stagger_time = bucket * 2.0 
        
        print(f"\n[Client {self.client_id}] Staggering Solid Pod Sync by {stagger_time:.1f}s (Bucket {bucket}) to avoid server overload...")
        time.sleep(stagger_time)
        
        solid_client = SolidTokenClient(
            pod_url=self.solid_url,
            username=self.solid_username,
            token_id=self.solid_token_id,
            token_secret=self.solid_token_secret
        )
        
        if solid_client.authenticate():
            common_dest = os.path.join(self.pod_dir, 'train_common.csv')
            solid_client.download_file("/swarm_data/train_common.csv", common_dest)
            
            vuln_dest = os.path.join(self.pod_dir, 'train_vulnerable.csv')
            solid_client.download_file("/swarm_data/train_vulnerable.csv", vuln_dest)
            
            test_dest = os.path.join(self.pod_dir, 'test.csv')
            solid_client.download_file("/swarm_data/test.csv", test_dest)
        else:
            print(f"⚠️ [Client {self.client_id}] Solid Auth failed. Falling back to local offline data if available.")

    def train_local_epoch(self):
        try:
            exclude_vuln = self.contract.functions.excludeVulnerableData(self.wallet_address).call()
        except Exception:
            exclude_vuln = True
            
        status_msg = "EXCLUDING" if exclude_vuln else "INCLUDING"
        print(f"[Client {self.client_id}] Local training... Privacy Flag is {status_msg} vulnerable data.")

        common_path = os.path.join(self.pod_dir, 'train_common.csv')
        vuln_path = os.path.join(self.pod_dir, 'train_vulnerable.csv')
        test_path = os.path.join(self.pod_dir, 'test.csv')
        
        if not os.path.exists(common_path):
            return

        train_df = pd.read_csv(common_path)
        if not exclude_vuln and os.path.exists(vuln_path):
            vuln_df = pd.read_csv(vuln_path)
            train_df = pd.concat([train_df, vuln_df], ignore_index=True)

        if train_df.empty:
            return

        global_seen_set = set(train_df['recipe_id'].values)
        if exclude_vuln and os.path.exists(vuln_path):
            vuln_df = pd.read_csv(vuln_path)
            global_seen_set.update(vuln_df['recipe_id'].values)
            
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
            if not test_df.empty:
                global_seen_set.update(test_df['recipe_id'].values)

        pos_recipes = train_df['recipe_id'].values % TOTAL_RECIPES
        pos_labels = (train_df['rating'].values >= 4).astype(float)

        pos_count = len(pos_recipes)
        num_negatives = pos_count * 1 if pos_count > 0 else 1

        all_possible = np.arange(TOTAL_RECIPES)
        seen_array = np.array(list(global_seen_set)) if len(global_seen_set) > 0 else np.array([], dtype=int)
        valid_pool = np.setdiff1d(all_possible, seen_array)

        if valid_pool.size == 0:
            neg_recipes = np.random.randint(0, TOTAL_RECIPES, size=num_negatives)
        else:
            replace = valid_pool.size < num_negatives
            neg_recipes = np.random.choice(valid_pool, size=num_negatives, replace=replace)

        neg_labels = np.zeros(len(neg_recipes), dtype=float)

        all_recipes = np.concatenate([pos_recipes, neg_recipes])
        all_labels = np.concatenate([pos_labels, neg_labels])
        perm = np.random.permutation(len(all_recipes))
        all_recipes = all_recipes[perm]
        all_labels = all_labels[perm]

        local_genre_lookup = {}
        for _, row in train_df.iterrows():
            rec_id = int(row['recipe_id']) % TOTAL_RECIPES
            genre_str = str(row['genres']).split('|')
            local_genre_lookup[rec_id] = [GENRE_MAP.get(g.strip(), 0) for g in genre_str]

        all_genres_list = [local_genre_lookup.get(int(r_id), [0]) for r_id in all_recipes]

        dataset = RecipeGenreDataset(all_recipes, all_labels, all_genres_list)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_GLOBAL, shuffle=True, collate_fn=genre_collate_fn)

        self.model.train()
        total_loss = 0.0
        global_weights = {name: param.clone().detach().to(param.device) for name, param in self.model.named_parameters()}
        mu = 0.01 

        #pos = float(np.sum(all_labels == 1.0))
        #neg = float(np.sum(all_labels == 0.0))
        #pos_weight = min(neg / pos if pos > 0 else 1.0, 10.0)

        local_criterion = nn.BCEWithLogitsLoss()

        for batch_recipes, batch_genres, batch_offsets, batch_labels in dataloader:
            self.optimizer.zero_grad()
            predictions = self.model(batch_recipes, batch_genres, batch_offsets).unsqueeze(1)
            base_loss = local_criterion(predictions, batch_labels)

            proximal_term = 0.0
            unique_recipes = torch.unique(batch_recipes)
            for name, param in self.model.named_parameters():
                if 'my_personal_embedding' not in name:
                    gw = global_weights[name]
                    if 'recipe_embedding' in name or 'recipe_bias' in name:
                        proximal_term = proximal_term + torch.sum((param[unique_recipes] - gw[unique_recipes]) ** 2)
                    else:
                        proximal_term = proximal_term + torch.sum((param - gw) ** 2)

            loss = base_loss + (mu / 2.0) * proximal_term
            loss.backward()
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
        if receipt.status == 0:
            raise Exception(f"Transaction reverted on-chain! Gas used: {receipt.gasUsed}. If it hit the limit, increase the gas.")
        return receipt

    def submit_to_swarm(self, swarm_id, epoch):
        try:
            scaled_dp_budget = self.contract.functions.dpBudgets(self.wallet_address).call()
        except Exception:
            scaled_dp_budget = 0
        
        state_dict = self.model.state_dict()
        public_weights_only = {k: v.clone() for k, v in state_dict.items() if 'my_personal_embedding' not in k}
        
        if scaled_dp_budget > 0:
            epsilon = scaled_dp_budget / 100.0
            std_dev = 0.1 / epsilon 
            for key in public_weights_only.keys():
                noise = torch.randn_like(public_weights_only[key]) * std_dev
                public_weights_only[key] += noise

        weights_uri = os.path.join(self.pod_dir, f"weights_s{swarm_id}_e{epoch}.pt")
        torch.save(public_weights_only, weights_uri)

        try:
            base_tx = {
                'from': self.wallet_address,
                'nonce': self.w3.eth.get_transaction_count(self.wallet_address),
                'gasPrice': self.w3.eth.gas_price
            }
            gas_estimate = self.contract.functions.submitWeights(swarm_id, epoch, weights_uri).estimate_gas(base_tx)
            base_tx['gas'] = int(gas_estimate * 1.2)
            
            tx = self.contract.functions.submitWeights(swarm_id, epoch, weights_uri).build_transaction(base_tx)
            self.sign_and_send(tx)
        except Exception as e:
            print(f"⚠️ [Client {self.client_id}] Submission rejected (Likely too late for Epoch {epoch}): {e}")

    def wait_for_consensus(self, swarm_id, epoch):
        import re
        while True:
            try:
                swarm_data = self.contract.functions.swarms(swarm_id).call()
                current_leader = swarm_data[2]
                global_uri = swarm_data[3]     
            except Exception as e:
                time.sleep(random.uniform(5.0, 10.0))
                continue
            
            if current_leader == self.wallet_address:
                self.execute_leader_duties(swarm_id, epoch)
                return epoch + 1 
                
            elif global_uri:
                match = re.search(r'_e(\d+)\.pt', global_uri)
                if match:
                    global_epoch = int(match.group(1))
                    if global_epoch >= epoch:
                        time.sleep(random.uniform(0.1, 3.0))
                        try:
                            global_state = torch.load(global_uri, weights_only=True)
                            self.model.load_state_dict(global_state, strict=False)
                            return global_epoch + 1
                        except Exception:
                            pass
            time.sleep(2)

    def execute_leader_duties(self, swarm_id, epoch):
        try:
            uris = self.contract.functions.getAllWeightsUris(swarm_id).call()
        except Exception:
            return
            
        uris = [uri for uri in uris if f"_e{epoch}.pt" in uri]
        if not uris: return
        
        aggregated_state = None
        for uri in uris:
            try:
                state = torch.load(uri, weights_only=True)
                if aggregated_state is None:
                    aggregated_state = state
                else:
                    for key in aggregated_state.keys():
                        aggregated_state[key] += state[key]
                del state      
                gc.collect()   
            except Exception:
                continue
                    
        for key in aggregated_state.keys():
            aggregated_state[key] = torch.div(aggregated_state[key], len(uris))

        global_uri = os.path.join(GLOBAL_DIR, f"global_model_s{swarm_id}_e{epoch}.pt")
        os.makedirs(GLOBAL_DIR, exist_ok=True)
        torch.save(aggregated_state, global_uri)

        try:
            base_tx = {
                'from': self.wallet_address,
                'nonce': self.w3.eth.get_transaction_count(self.wallet_address),
                'gasPrice': self.w3.eth.gas_price
            }
            gas_estimate = self.contract.functions.submitGlobalModel(swarm_id, global_uri).estimate_gas(base_tx)
            base_tx['gas'] = int(gas_estimate * 1.2)
            
            tx = self.contract.functions.submitGlobalModel(swarm_id, global_uri).build_transaction(base_tx)
            self.sign_and_send(tx)
        except Exception:
            pass
        
        self.model.load_state_dict(aggregated_state, strict=False)

    def evaluate(self, epoch):
        test_path = os.path.join(self.pod_dir, 'test.csv')
        train_path = os.path.join(self.pod_dir, 'train_common.csv')
        vuln_path = os.path.join(self.pod_dir, 'train_vulnerable.csv')
        
        if not os.path.exists(test_path): return
            
        test_df = pd.read_csv(test_path)
        if test_df.empty: return
            
        train_df = pd.read_csv(train_path) if os.path.exists(train_path) else pd.DataFrame(columns=['recipe_id'])
        
        try:
            exclude_vuln = self.contract.functions.excludeVulnerableData(self.wallet_address).call()
        except Exception:
            exclude_vuln = True

        if not exclude_vuln and os.path.exists(vuln_path):
            vuln_df = pd.read_csv(vuln_path)
            train_df = pd.concat([train_df, vuln_df], ignore_index=True)
            
        self.model.eval()

        local_genre_lookup = {}
        for df in [train_df, test_df]:
            if not df.empty and 'genres' in df.columns:
                for _, row in df.iterrows():
                    rec_id = int(row['recipe_id']) % TOTAL_RECIPES
                    genre_str = str(row['genres']).split('|')
                    local_genre_lookup[rec_id] = [GENRE_MAP.get(g.strip(), 0) for g in genre_str]

        test_recipes = test_df['recipe_id'].values % TOTAL_RECIPES
        y_true = (test_df['rating'].values >= 4).astype(float)
        test_genres_list = [local_genre_lookup.get(int(r), [0]) for r in test_recipes]
        
        eval_dataset = RecipeGenreDataset(test_recipes, y_true, test_genres_list)
        eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE_GLOBAL, shuffle=False, collate_fn=genre_collate_fn)
        
        all_probs = []
        with torch.no_grad():
            for batch_recipes, batch_genres, batch_offsets, _ in eval_loader:
                logits = self.model(batch_recipes, batch_genres, batch_offsets).squeeze(-1)
                probs = torch.sigmoid(logits).numpy()
                if probs.ndim == 0:
                    probs = np.expand_dims(probs, 0)
                all_probs.extend(probs.tolist())
                
        probs = np.array(all_probs)
        y_pred = (probs >= 0.5).astype(float)
            
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {'epoch': epoch, 'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}
        
        metrics_path = os.path.join(self.pod_dir, 'metrics.csv')
        df = pd.DataFrame([metrics])
        if not os.path.exists(metrics_path):
            df.to_csv(metrics_path, index=False)
        else:
            df.to_csv(metrics_path, mode='a', header=False, index=False)

    def evaluate_final_ranking(self):
        test_path = os.path.join(self.pod_dir, 'test.csv')
        train_path = os.path.join(self.pod_dir, 'train_common.csv')
        
        if not os.path.exists(test_path): return
            
        test_df = pd.read_csv(test_path)
        if test_df.empty: return
            
        train_df = pd.read_csv(train_path) if os.path.exists(train_path) else pd.DataFrame(columns=['recipe_id'])
        self.model.eval()

        local_genre_lookup = {}
        for df in [train_df, test_df]:
            if not df.empty and 'genres' in df.columns:
                for _, row in df.iterrows():
                    rec_id = int(row['recipe_id']) % TOTAL_RECIPES
                    genre_str = str(row['genres']).split('|')
                    local_genre_lookup[rec_id] = [GENRE_MAP.get(g.strip(), 0) for g in genre_str]
        
        seen_recipes = set(test_df['recipe_id'].values % TOTAL_RECIPES).union(set(train_df['recipe_id'].values % TOTAL_RECIPES))
        
        liked_test_df = test_df[test_df['rating'] >= 4]
        disliked_test_df = test_df[test_df['rating'] < 4]
        actual_test_negatives = np.unique(disliked_test_df['recipe_id'].values % TOTAL_RECIPES)
        
        client_hits = 0
        client_ndcg = 0
        total_ranking_tests = len(liked_test_df)
        
        if total_ranking_tests == 0: return

        all_possible_recipes = np.arange(TOTAL_RECIPES)
        seen_array = np.array(list(seen_recipes))
        valid_negatives_pool = np.setdiff1d(all_possible_recipes, seen_array)

        with torch.no_grad():
            for index, row in liked_test_df.iterrows():
                true_id = int(row['recipe_id'] % TOTAL_RECIPES)
                
                if len(actual_test_negatives) >= NUM_NEGATIVE_SAMPLES:
                    fake_recipes_array = np.random.choice(actual_test_negatives, NUM_NEGATIVE_SAMPLES, replace=False)
                else:
                    fake_from_test = actual_test_negatives
                    shortfall = NUM_NEGATIVE_SAMPLES - len(actual_test_negatives)
                    if valid_negatives_pool.size >= shortfall:
                        fake_from_pool = np.random.choice(valid_negatives_pool, shortfall, replace=False)
                    else:
                        reps = int(np.ceil(shortfall / max(1, valid_negatives_pool.size)))
                        fake_from_pool = np.tile(valid_negatives_pool, reps)[:shortfall]
                    fake_recipes_array = np.concatenate([fake_from_test, fake_from_pool])

                all_recipes_array = np.concatenate([[true_id], fake_recipes_array])
                batch_genres_list = [local_genre_lookup.get(int(r), [0]) for r in all_recipes_array]
                
                flattened_genres = []
                offsets = [0]
                for g_list in batch_genres_list:
                    flattened_genres.extend(g_list)
                    offsets.append(len(flattened_genres))
                    
                offsets = torch.tensor(offsets[:-1], dtype=torch.long)
                genres_tensor = torch.tensor(flattened_genres, dtype=torch.long) if flattened_genres else torch.empty(0, dtype=torch.long)
                all_recipes_tensor = torch.tensor(all_recipes_array, dtype=torch.long)

                scores = torch.sigmoid(self.model(all_recipes_tensor, genres_tensor, offsets)).squeeze(-1).numpy()
                if scores.ndim == 0:
                    scores = np.expand_dims(scores, 0)
                    
                ranked_indices = scores.argsort()[::-1]
                rank_of_true_item = (ranked_indices == 0).nonzero()[0][0]

                if rank_of_true_item < TOP_N:
                    client_hits += 1
                    client_ndcg += 1.0 / math.log2(rank_of_true_item + 2)
                    
        ranking_metrics = {'hits': int(client_hits), 'ndcg': float(client_ndcg), 'total_ranking_tests': int(total_ranking_tests)}
        pd.DataFrame([ranking_metrics]).to_csv(os.path.join(self.pod_dir, 'final_ranking.csv'), index=False)

    def run(self):
        # 1. Sync is now performed here in the run loop to respect the orchestrator's concurrency limits
        self.sync_data_from_solid()

        metrics_path = os.path.join(self.pod_dir, 'metrics.csv')
        ranking_path = os.path.join(self.pod_dir, 'final_ranking.csv')
        if os.path.exists(metrics_path): os.remove(metrics_path)
        if os.path.exists(ranking_path): os.remove(ranking_path)

        patience = 2
        min_delta = 0.0001
        global_swarm_id = 0
        try:
            env_start = os.environ.get('START_EPOCH', None)
            current_epoch = int(env_start) if env_start is not None else 0
        except Exception:
            current_epoch = 0

        if current_epoch >= TOTAL_EPOCHS: return
        
        while current_epoch < TOTAL_EPOCHS:
            best_loss = float('inf')
            patience_counter = 0
            
            for local_step in range(ROUNDS_PER_EPOCH): 
                avg_loss = self.train_local_epoch()
                if avg_loss is None: break 
                
                if avg_loss < (best_loss - min_delta):
                    best_loss = avg_loss
                    patience_counter = 0 
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        patience_counter = 0
                        break
                        
            self.submit_to_swarm(global_swarm_id, current_epoch)
            new_epoch = self.wait_for_consensus(global_swarm_id, current_epoch)
            
            self.evaluate(new_epoch - 1)
            current_epoch = new_epoch

            self.scheduler.step()
            
        full_local_path = os.path.join(self.pod_dir, 'local_full_model.pt')
        torch.save(self.model.state_dict(), full_local_path)
        
        self.evaluate_final_ranking()

if __name__ == "__main__":
    node = SwarmNode(client_id=1)
    node.run()