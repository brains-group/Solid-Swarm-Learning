import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import NUM_ACTIVE_CLIENTS, TOTAL_EPOCHS, TOP_N

PODS_DIR = "../pods"
GLOBAL_DIR = "../global"
IMAGES_DIR = "../images"

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
            rdf['client_id'] = client_id
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

    # --- Derived ROC Values ---
    tpr = recall # True Positive Rate is mathematically identical to Recall
    fpr = np.divide(global_fp, global_fp + global_tn, out=np.zeros_like(global_fp, dtype=float), where=(global_fp+global_tn)!=0)

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
                
                # --- New Plot: Per-Client Ranking Metrics ---
                os.makedirs(IMAGES_DIR, exist_ok=True)
                combined_ranking['hr'] = np.where(combined_ranking['total_ranking_tests'] > 0, 
                                                  combined_ranking['hits'] / combined_ranking['total_ranking_tests'], 0)
                combined_ranking['ndcg_score'] = np.where(combined_ranking['total_ranking_tests'] > 0, 
                                                          combined_ranking['ndcg'] / combined_ranking['total_ranking_tests'], 0)
                
                plt.figure(figsize=(14, 6))
                bar_width = 0.35
                x = np.arange(len(combined_ranking['client_id']))
                
                plt.bar(x - bar_width/2, combined_ranking['hr'], width=bar_width, label=f'HR@{TOP_N}', color='#66BB6A')
                plt.bar(x + bar_width/2, combined_ranking['ndcg_score'], width=bar_width, label=f'NDCG@{TOP_N}', color='#FFA726')
                
                plt.title(f'Final Top-{TOP_N} Recommendation Metrics per Client')
                plt.xlabel('Client ID')
                plt.ylabel('Score')
                plt.xticks(x, combined_ranking['client_id'], rotation=90 if len(x) > 20 else 0)
                plt.ylim(0, 1.05)
                plt.legend()
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(IMAGES_DIR, 'per_client_ranking_metrics.png'))
                plt.close()
            
        # Generate the classification plots
        os.makedirs(IMAGES_DIR, exist_ok=True)
        
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
        plt.savefig(os.path.join(IMAGES_DIR, 'classification_metrics_plot.png'))
        plt.close()
        
        # --- New Plot: Stacked Area CM Components ---
        plt.figure(figsize=(10, 6))
        plt.stackplot(epochs, global_tp, global_tn, global_fp, global_fn, labels=['TP', 'TN', 'FP', 'FN'], colors=['#4CAF50', '#83BB6A', '#F44336', '#225350'], alpha=0.8)
        plt.title('Global Confusion Matrix Distribution over Epochs')
        plt.xlabel('Global Epoch')
        plt.ylabel('Total Classifications')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, 'global_cm_stacked_area.png'))
        plt.close()
        
        # --- New Plot: Per-Client Final Epoch Accuracy ---
        final_epoch = epochs[-1]
        final_client_metrics = combined_df[combined_df['epoch'] == final_epoch].copy()
        if not final_client_metrics.empty:
            final_client_metrics['total'] = final_client_metrics['tp'] + final_client_metrics['tn'] + final_client_metrics['fp'] + final_client_metrics['fn']
            final_client_metrics['accuracy'] = np.where(final_client_metrics['total'] > 0, 
                                                        (final_client_metrics['tp'] + final_client_metrics['tn']) / final_client_metrics['total'], 0)
            
            plt.figure(figsize=(14, 6))
            plt.bar(final_client_metrics['client_id'], final_client_metrics['accuracy'], color='#29B6F6', edgecolor='black')
            plt.axhline(y=accuracy[-1], color='#D32F2F', linestyle='--', linewidth=2, label=f'Global Avg ({accuracy[-1]:.4f})')
            plt.title(f'Final Epoch ({int(final_epoch)}) Accuracy per Client')
            plt.xlabel('Client ID')
            plt.ylabel('Accuracy')
            plt.xticks(final_client_metrics['client_id'], rotation=90 if len(final_client_metrics['client_id']) > 20 else 0)
            plt.ylim(0, 1.05)
            plt.legend()
            plt.grid(axis='y', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(IMAGES_DIR, 'per_client_final_accuracy.png'))
            plt.close()
            
            # --- New Plot: Client Accuracy Trajectories (Spaghetti Plot) ---
            plt.figure(figsize=(12, 7))
            for c_id in combined_df['client_id'].unique():
                client_data = combined_df[combined_df['client_id'] == c_id].copy()
                client_data = client_data.sort_values('epoch')
                c_total = client_data['tp'] + client_data['tn'] + client_data['fp'] + client_data['fn']
                c_acc = np.where(c_total > 0, (client_data['tp'] + client_data['tn']) / c_total, 0)
                plt.plot(client_data['epoch'], c_acc, alpha=0.3, color='gray')
                
            plt.plot(epochs, accuracy, color='#D32F2F', linewidth=3, label='Global Average')
            plt.title('Client Accuracy Trajectories over Epochs')
            plt.xlabel('Global Epoch')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1.05)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(IMAGES_DIR, 'client_accuracy_trajectories.png'))
            plt.close()

            # --- New Plot: Distribution of Final Client Accuracies ---
            plt.figure(figsize=(8, 6))
            plt.hist(final_client_metrics['accuracy'], bins=np.linspace(0, 1, 21), color='#AB47BC', edgecolor='black', alpha=0.7)
            plt.axvline(x=accuracy[-1], color='#D32F2F', linestyle='--', linewidth=2, label=f'Global Avg ({accuracy[-1]:.4f})')
            plt.title('Distribution of Final Client Accuracies')
            plt.xlabel('Accuracy')
            plt.ylabel('Number of Clients')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(IMAGES_DIR, 'final_accuracy_distribution.png'))
            plt.close()

            # --- New Plot: Evaluation Data Volume per Client ---
            plt.figure(figsize=(14, 6))
            plt.bar(final_client_metrics['client_id'], final_client_metrics['total'], color='#8D6E63', edgecolor='black')
            plt.title('Total Evaluation Samples per Client (Data Diversity/Imbalance)')
            plt.xlabel('Client ID')
            plt.ylabel('Number of Samples')
            plt.xticks(final_client_metrics['client_id'], rotation=90 if len(final_client_metrics['client_id']) > 20 else 0)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(IMAGES_DIR, 'per_client_data_volume.png'))
            plt.close()


            # --- NEW PLOT: GLOBAL ROC TRAJECTORY ---
        plt.figure(figsize=(8, 8))
        # Plot the trajectory line through ROC space
        plt.plot(fpr, tpr, color='#FF8C00', lw=2, marker='o', markersize=6, label='Global Swarm Trajectory')
        
        # Annotate specific epochs along the line to show progress direction
        for i, epoch_num in enumerate(epochs):
            if i == 0 or i == len(epochs)-1 or i % max(1, len(epochs)//5) == 0:
                plt.annotate(f"E{int(epoch_num)}", (fpr[i], tpr[i]), textcoords="offset points", xytext=(8,-5), ha='left', fontsize=10, fontweight='bold')

        plt.plot([0, 1], [0, 1], color='#2F4F4F', lw=2, linestyle='--') # Random guess diagonal
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Recall)', fontsize=12)
        plt.title(f'Global Swarm ROC Trajectory\n(Movement of the global decision boundary across {TOTAL_EPOCHS} epochs)', fontsize=14)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        
        roc_traj_path = os.path.join(IMAGES_DIR, 'global_roc_trajectory.png')
        plt.savefig(roc_traj_path)
        plt.close()
            
        print(f"\n📈 Visualizations successfully saved to the '{IMAGES_DIR}' directory.")
    else:
        print("No evaluation data could be processed.")

    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()