import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import NUM_ACTIVE_CLIENTS, TOTAL_EPOCHS, TOP_N

PODS_DIR = "../pods"
GLOBAL_DIR = "../global"

def main():
    print("📊 Aggregating Per-Epoch Classification and Final Ranking Stats...\n")
    print(f"{'='*60}")
    print(f"--- Fetching Evaluation Data for {NUM_ACTIVE_CLIENTS} Pods ---")
    
    all_metrics = []
    all_ranking = []
    
    for client_id in range(1, NUM_ACTIVE_CLIENTS + 1):
        pod_dir = os.path.join(PODS_DIR, f"client_{client_id}")
        metrics_path = os.path.join(pod_dir, 'metrics.csv')
        ranking_path = os.path.join(pod_dir, 'final_ranking.csv')
        
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            df['client_id'] = client_id
            all_metrics.append(df)
            
        if os.path.exists(ranking_path):
            rdf = pd.read_csv(ranking_path)
            all_ranking.append(rdf)

    if not all_metrics:
        print("⚠️ No metrics data found. Ensure that training has completed and clients have written 'metrics.csv'.")
        return

    combined_df = pd.concat(all_metrics, ignore_index=True)
    epoch_stats = combined_df.groupby('epoch').sum().reset_index()

    epochs = epoch_stats['epoch'].values
    global_tp = epoch_stats['tp'].values
    global_tn = epoch_stats['tn'].values
    global_fp = epoch_stats['fp'].values
    global_fn = epoch_stats['fn'].values

    total_classifications = global_tp + global_tn + global_fp + global_fn
    
    accuracy = np.divide(global_tp + global_tn, total_classifications, out=np.zeros_like(global_tp, dtype=float), where=total_classifications!=0)
    precision = np.divide(global_tp, global_tp + global_fp, out=np.zeros_like(global_tp, dtype=float), where=(global_tp+global_fp)!=0)
    recall = np.divide(global_tp, global_tp + global_fn, out=np.zeros_like(global_tp, dtype=float), where=(global_tp+global_fn)!=0)
    f1_scores = np.divide(2 * (precision * recall), (precision + recall), out=np.zeros_like(global_tp, dtype=float), where=(precision+recall)!=0)

    print(f"--- Simulating Edge Evaluation Aggregation over {len(epochs)} Epochs ---")

    for i, epoch in enumerate(epochs):
        print(f"[Epoch {int(epoch):02d}] Acc: {accuracy[i]:.4f} | Prec: {precision[i]:.4f} | Rec: {recall[i]:.4f} | F1: {f1_scores[i]:.4f}")

    # =========================================================
    # FINAL MATH & REPORTING
    # =========================================================
    print(f"\n{'='*60}")
    print("🎉 DECENTRALIZED EVALUATION COMPLETE 🎉")
    print(f"{'='*60}")
    
    if len(epochs) > 0:
        print("\n--- Final Epoch Metrics (Threshold = 0.5) ---")
        print(f"Accuracy  : {accuracy[-1]:.4f} ({accuracy[-1]*100:.2f}%)")
        print(f"Precision : {precision[-1]:.4f}")
        print(f"Recall    : {recall[-1]:.4f}")
        print(f"F1-Score  : {f1_scores[-1]:.4f}")
        
        print("\n--- Final Confusion Matrix ---")
        print(f"                Predicted False | Predicted True")
        print(f"Actual False  | TN: {int(global_tn[-1]):<11} | FP: {int(global_fp[-1])}")
        print(f"Actual True   | FN: {int(global_fn[-1]):<11} | TP: {int(global_tp[-1])}")

        # 2. Ranking Metrics
        if all_ranking:
            combined_ranking = pd.concat(all_ranking, ignore_index=True)
            global_hits = combined_ranking['hits'].sum()
            global_ndcg = combined_ranking['ndcg'].sum()
            total_ranking_tests = combined_ranking['total_ranking_tests'].sum()
            
            if total_ranking_tests > 0:
                final_hr = global_hits / total_ranking_tests
                final_ndcg = global_ndcg / total_ranking_tests
                
                print("\n--- Final Phase 2: Top-N Recommendation Metrics ---")
                print(f"Total Ranking Tests : {int(total_ranking_tests)}")
                print(f"Global HR@{TOP_N}       : {final_hr:.4f} ({final_hr*100:.2f}%)")
                print(f"Global NDCG@{TOP_N}     : {final_ndcg:.4f}")
            
        # Generate the plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, accuracy, label='Accuracy', marker='o')
        plt.plot(epochs, precision, label='Precision', marker='s')
        plt.plot(epochs, recall, label='Recall', marker='^')
        plt.title('Global Swarm Classification Metrics over Epochs')
        plt.xlabel('Global Epoch')
        plt.ylabel('Score (0.0 to 1.0)')
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(GLOBAL_DIR, 'classification_metrics_plot.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"\n📈 Line graph saved to: {plot_path}")
    else:
        print("No evaluation data could be processed.")

    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()