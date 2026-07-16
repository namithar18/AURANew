import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, confusion_matrix

df = pd.read_csv('ch1_robustness_results.csv')

y_true = df['is_byz'].astype(int)
y_score = df['ch1'] # Higher means Byzantine

auc_roc = roc_auc_score(y_true, y_score)
precision, recall, thresholds_pr = precision_recall_curve(y_true, y_score)
auc_pr = auc(recall, precision)
print(f"ROC-AUC: {auc_roc:.4f}, PR-AUC: {auc_pr:.4f}")

thresholds = np.linspace(0, 1.0, 101)
results = []
for t in thresholds:
    # Byzantine if ch1 > t
    y_pred = (df['ch1'] > t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    bal_acc = (rec + (1 - fpr)) / 2
    j = rec - fpr
    results.append({'Threshold': t, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'Balanced_Accuracy': bal_acc, 'FPR': fpr, 'FNR': fnr, 'J': j})

res_df = pd.DataFrame(results)
best_j = res_df.loc[res_df['J'].idxmax()]
best_f1 = res_df.loc[res_df['F1'].idxmax()]

print("\n--- BEST THRESHOLDS ---")
print(f"Max Youden's J: t={best_j['Threshold']:.4f} (J={best_j['J']:.4f}, F1={best_j['F1']:.4f})")
print(f"Max F1: t={best_f1['Threshold']:.4f} (F1={best_f1['F1']:.4f}, J={best_f1['J']:.4f})")

res_df.to_csv('ch1_threshold_analysis.csv', index=False)

N_SEEDS = 30
seed_best_t = []
for s in range(N_SEEDS):
    sdf = df[df['seed'] == s]
    best_j_s = -1
    best_t_s = -1
    for t in thresholds:
        yp = (sdf['ch1'] > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(sdf['is_byz'], yp, labels=[0, 1]).ravel()
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        j = rec - fpr
        if j > best_j_s:
            best_j_s = j
            best_t_s = t
    seed_best_t.append(best_t_s)

seed_best_t = np.array(seed_best_t)
print("\n--- SEED STABILITY ---")
print(f"Optimal Thresholds across seeds: Mean={seed_best_t.mean():.4f}, Std={seed_best_t.std():.4f}")
print(f"95% CI: [{np.percentile(seed_best_t, 2.5):.4f}, {np.percentile(seed_best_t, 97.5):.4f}]")

# Print the specific range [0.25, 0.55] with step 0.05
print("\n--- DETAILED TABLE ---")
print(res_df[(res_df['Threshold'] >= 0.25) & (res_df['Threshold'] <= 0.55)].to_string(index=False))
