import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from data_generation import generate_data
from group_lasso import group_lasso, select_lambda_group_lasso_target_groups
from group_lars import group_lars
from group_garrote import group_non_negative_garrote_full

true_groups = {0, 2, 4}  # Grupy generujące sygnał

def evaluate_selection(beta, group_indices, true_groups):
    selected = set()
    for i, idxs in enumerate(group_indices):
        if np.linalg.norm(beta[idxs]) > 1e-6:
            selected.add(i)
    tp = len(selected & true_groups)
    fp = len(selected - true_groups)
    fn = len(true_groups - selected)
    tn = len(group_indices) - len(true_groups) - fp
    tpr = tp / len(true_groups) if len(true_groups) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return tpr, fpr

# Zmienna do zbierania wyników
results = {
    "Method": ["Group Lasso", "Group LARS", "Group Garrote"],
    "MSE": [0.0, 0.0, 0.0],
    "TPR": [0.0, 0.0, 0.0],
    "FPR": [0.0, 0.0, 0.0]
}

n_repeats = 100
for seed in range(n_repeats):
    X, Y, group_indices = generate_data(seed=seed)

    # Group Lasso
    true_group_count = 3  # bo tylko grupy 0, 2, 4 są istotne
    best_lambda = select_lambda_group_lasso_target_groups(X, Y, group_indices, true_group_count)
    beta_lasso, pred_lasso = group_lasso(X, Y, group_indices, lambda_=best_lambda)
    mse_lasso = mean_squared_error(Y, pred_lasso)
    tpr_lasso, fpr_lasso = evaluate_selection(beta_lasso, group_indices, true_groups)

    # Group LARS
    beta_lars, pred_lars = group_lars(X, Y, group_indices)
    mse_lars = mean_squared_error(Y, pred_lars)
    tpr_lars, fpr_lars = evaluate_selection(beta_lars, group_indices, true_groups)

    # Group Garrote
    beta_garrote, pred_garrote, _, _ = group_non_negative_garrote_full(X, Y, group_indices)
    mse_garrote = mean_squared_error(Y, pred_garrote)
    tpr_garrote, fpr_garrote = evaluate_selection(beta_garrote, group_indices, true_groups)

    # Zbieranie wyników
    results["MSE"][0] += mse_lasso
    results["MSE"][1] += mse_lars
    results["MSE"][2] += mse_garrote

    results["TPR"][0] += tpr_lasso
    results["TPR"][1] += tpr_lars
    results["TPR"][2] += tpr_garrote

    results["FPR"][0] += fpr_lasso
    results["FPR"][1] += fpr_lars
    results["FPR"][2] += fpr_garrote

# Uśrednienie
for metric in ["MSE", "TPR", "FPR"]:
    results[metric] = [v / n_repeats for v in results[metric]]

# Wyniki
results_df = pd.DataFrame(results)
print("\nŚrednie wyniki po 100 powtórzeniach:")
print(results_df)
