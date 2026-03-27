import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math
from sklearn.metrics import confusion_matrix
from config import NUM_ACTIVE_CLIENTS, TOTAL_EPOCHS, TOP_N, NUM_NEGATIVE_SAMPLES, TOTAL_RECIPES

# --- Configuration ---
PODS_DIR = "../pods"
GLOBAL_DIR = "../global"

# --- Decentralized Model Definition ---
class DecentralizedFoodRecommender(nn.Module):
    def __init__(self, num_recipes=TOTAL_RECIPES, embedding_dim=32):
        super(DecentralizedFoodRecommender, self).__init__()
        self.my_personal_embedding = nn.Parameter(torch.randn(1, embedding_dim))
        self.recipe_embedding = nn.Embedding(num_recipes, embedding_dim)

    def forward(self, recipe_idx):
        r = self.recipe_embedding(recipe_idx)
        scores = torch.sum(self.my_personal_embedding * r, dim=1)
        return torch.sigmoid(scores).unsqueeze(1)

def main():
    print("📊 Initializing Unified Decentralized Swarm Evaluation...\n")
    
    swarm_id = 0 
    global_model_path = os.path.join(GLOBAL_DIR, f"global_model_s{swarm_id}_e{TOTAL_EPOCHS - 1}.pt")
    
    if not os.path.exists(global_model_path):
        print(f"Global model not found at {global_model_path}. Exiting.")
        return

    print(f"{'='*60}")
    print(f"--- Simulating Edge Evaluation across {NUM_ACTIVE_CLIENTS} Pods ---")
    
    # 1. Global Classification Trackers
    global_tp, global_tn, global_fp, global_fn = 0, 0, 0, 0
    
    # 2. Global Ranking Trackers
    global_hits = 0
    global_ndcg = 0
    total_ranking_tests = 0

    for client_id in range(1, NUM_ACTIVE_CLIENTS + 1):
        pod_dir = os.path.join(PODS_DIR, f"client_{client_id}")
        test_path = os.path.join(pod_dir, 'test.csv')
        train_path = os.path.join(pod_dir, 'train.csv')
        local_model_path = os.path.join(pod_dir, 'local_full_model.pt')
        
        if not (os.path.exists(test_path) and os.path.exists(local_model_path)):
            continue
            
        test_df = pd.read_csv(test_path)
        train_df = pd.read_csv(train_path) if os.path.exists(train_path) else pd.DataFrame(columns=['recipe_id'])
        
        if test_df.empty:
            continue

        # Initialize the model for this specific client
        local_model = DecentralizedFoodRecommender()
        local_model.load_state_dict(torch.load(local_model_path, weights_only=True))
        global_state = torch.load(global_model_path, weights_only=True)
        local_model.load_state_dict(global_state, strict=False)
        local_model.eval()
        
        # =========================================================
        # PHASE 1: Classification Evaluation (Thresholding)
        # =========================================================
        recipes_tensor = torch.tensor(test_df['recipe_id'].values % TOTAL_RECIPES, dtype=torch.long)
        y_true = (test_df['rating'].values >= 4).astype(float)

        with torch.no_grad():
            preds = local_model(recipes_tensor).squeeze().numpy()
            
            # Handle edge case where a user only has 1 item in their test set
            if preds.ndim == 0: 
                preds = np.expand_dims(preds, 0)
                
            y_pred = (preds >= 0.5).astype(float)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        global_tn += tn
        global_fp += fp
        global_fn += fn
        global_tp += tp

        # =========================================================
        # PHASE 2: Ranking Evaluation (Leave-One-Out Top-N)
        # =========================================================
        seen_recipes = set(test_df['recipe_id'].values % TOTAL_RECIPES).union(set(train_df['recipe_id'].values % TOTAL_RECIPES))
        client_hits = 0
        client_ndcg = 0
        
        # --- THE FIX: Only rank items the user ACTUALLY liked ---
        liked_test_df = test_df[test_df['rating'] >= 4]
        
        with torch.no_grad():
            # Iterate over the filtered dataframe instead of the full test_df
            for index, row in liked_test_df.iterrows():
                true_recipe = torch.tensor([row['recipe_id'] % TOTAL_RECIPES], dtype=torch.long)
                
                fake_recipes_list = []
                while len(fake_recipes_list) < NUM_NEGATIVE_SAMPLES:
                    rand_id = np.random.randint(0, TOTAL_RECIPES)
                    if rand_id not in seen_recipes:
                        fake_recipes_list.append(rand_id)
                
                fake_recipes = torch.tensor(fake_recipes_list, dtype=torch.long)
                all_recipes = torch.cat([true_recipe, fake_recipes])
                
                scores = local_model(all_recipes).squeeze().numpy()
                ranked_indices = scores.argsort()[::-1]
                rank_of_true_item = (ranked_indices == 0).nonzero()[0][0]
                
                if rank_of_true_item < TOP_N:
                    client_hits += 1
                    client_ndcg += 1.0 / math.log2(rank_of_true_item + 2) 
        
        global_hits += client_hits
        global_ndcg += client_ndcg
        
        # Make sure we only add the length of the filtered dataframe to the total!
        total_ranking_tests += len(liked_test_df) 
        
        print(f"[Client {client_id:02d}] Evaluated {len(test_df)} classifications and {len(liked_test_df)} rankings.")

    # =========================================================
    # FINAL MATH & REPORTING
    # =========================================================
    print(f"\n{'='*60}")
    print("🎉 DECENTRALIZED EVALUATION COMPLETE 🎉")
    print(f"{'='*60}")
    
    # 1. Classification Metrics
    total_classifications = global_tp + global_tn + global_fp + global_fn
    accuracy = (global_tp + global_tn) / total_classifications if total_classifications > 0 else 0
    precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
    recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n--- Phase 1: Classification Metrics (Threshold = 0.5) ---")
    print(f"Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"\nConfusion Matrix (Global):")
    print(f"TN: {global_tn} | FP: {global_fp}")
    print(f"FN: {global_fn} | TP: {global_tp}")

    # 2. Ranking Metrics
    if total_ranking_tests > 0:
        final_hr = global_hits / total_ranking_tests
        final_ndcg = global_ndcg / total_ranking_tests
        
        print("\n--- Phase 2: Top-N Recommendation Metrics ---")
        print(f"Total Ranking Tests : {total_ranking_tests}")
        print(f"Global HR@{TOP_N}       : {final_hr:.4f} ({final_hr*100:.2f}%)")
        print(f"Global NDCG@{TOP_N}     : {final_ndcg:.4f}")
    else:
        print("\n--- Phase 2: Top-N Recommendation Metrics ---")
        print("No evaluation data could be processed.")
        
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()