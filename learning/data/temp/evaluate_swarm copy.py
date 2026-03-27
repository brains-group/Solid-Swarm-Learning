import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from config import NUM_ACTIVE_CLIENTS, TOTAL_EPOCHS

# --- Configuration ---
PODS_DIR = "../pods"
GLOBAL_DIR = "../global"

# --- Model Definition (Must exactly match train_node.py) ---
class FoodRecommender(nn.Module):
    def __init__(self, num_users=100000, num_recipes=500000, embedding_dim=32):
        super(FoodRecommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.recipe_embedding = nn.Embedding(num_recipes, embedding_dim)
        self.fc = nn.Linear(embedding_dim * 2, 1)

    def forward(self, user_idx, recipe_idx):
        u = self.user_embedding(user_idx)
        r = self.recipe_embedding(recipe_idx)
        x = torch.cat([u, r], dim=1)
        return torch.sigmoid(self.fc(x))

def main():
    print("📊 Initializing Swarm Evaluation for the Global Model...\n")
    
    # Since all clients train a single model, we look for the model from swarm 0.
    swarm_id = 0 # Assuming a single global swarm for evaluation
    model_path = os.path.join(GLOBAL_DIR, f"global_model_s{swarm_id}_e{TOTAL_EPOCHS - 1}.pt")
    
    if not os.path.exists(model_path):
        print(f"Global model not found at {model_path}. Exiting.")
        return

    print(f"{'='*50}")
    print(f"--- Evaluating Global Model from {model_path} ---")
    
    model = FoodRecommender()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval() 
    
    # Collect test data from ALL active clients
    test_dfs = []
    for client_id in range(1, NUM_ACTIVE_CLIENTS + 1):
        test_path = os.path.join(PODS_DIR, f"client_{client_id}", 'test.csv')
        if os.path.exists(test_path):
            test_dfs.append(pd.read_csv(test_path))
            
    if not test_dfs:
        print("No test data found for any clients.\n")
        return
        
    combined_test_df = pd.concat(test_dfs, ignore_index=True)
    
    if combined_test_df.empty:
        print("Combined test data is empty. Cannot perform evaluation.\n")
        return
        
    # Prepare data for PyTorch
    users = torch.tensor(combined_test_df['user_id'].values % 100000, dtype=torch.long)
    recipes = torch.tensor(combined_test_df['recipe_id'].values % 500000, dtype=torch.long)
    
    # Convert actual ratings to binary labels: 1 if rating is 4 or 5 ("Liked"), else 0 ("Disliked")
    actual_labels_tensor = torch.tensor((combined_test_df['rating'].values >= 4).astype(float), dtype=torch.float32).unsqueeze(1)
    
    # Run inference and calculate metrics
    with torch.no_grad():
        predictions = model(users, recipes)
        
        # Convert sigmoid output (probabilities) to binary predictions (0 or 1)
        predicted_labels = (predictions >= 0.5).float()
        
        # --- Confusion Matrix & Classification Metrics ---
        y_true = actual_labels_tensor.numpy().flatten()
        y_pred = predicted_labels.numpy().flatten()
        
        # Generate the confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Handle case where a class is not predicted, to avoid division by zero
        if cm.shape == (1, 1):
            # This happens if all predictions and all true labels are the same class (e.g., all 0s or all 1s)
            if y_true[0] == 1: # All are positive
                tn, fp, fn, tp = 0, 0, 0, len(y_true)
            else: # All are negative
                tn, fp, fn, tp = len(y_true), 0, 0, 0
        else:
            tn, fp, fn, tp = cm.ravel()
        # Generate the confusion matrix, ensuring it's always 2x2.
        # labels=[0, 1] ensures slots for Disliked (0) and Liked (1), preventing errors
        # on datasets where only one class is present or predicted.
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
    print(f"Tested on {len(combined_test_df)} unseen interactions from {NUM_ACTIVE_CLIENTS} clients.")
    print("\n--- Classification Metrics ---")
    print(f"Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision : {precision:.4f} (Of predicted 'Liked', {precision*100:.2f}% were correct)")
    print(f"Recall    : {recall:.4f} (Of all actual 'Liked', {recall*100:.2f}% were found)")
    print(f"F1-Score  : {f1:.4f}")

    print("\n--- Confusion Matrix ---")
    print(" 'Liked' is the positive class (4-5 stars)")
    print(f"                 Predicted:")
    print(f"               Disliked | Liked")
    print(f"Actual: Disliked | {tn:<8} | {fp:<5} |")
    print(f"        Liked    | {fn:<8} | {tp:<5} |")
    print("------------------------")
    print(f"TN (Correctly Disliked): {tn}")
    print(f"FP (Incorrectly Liked) : {fp}")
    print(f"FN (Incorrectly Disliked): {fn}")
    print(f"TP (Correctly Liked)   : {tp}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()