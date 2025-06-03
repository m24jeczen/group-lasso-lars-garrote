import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from data_generation import generate_data
from group_lasso import group_lasso, select_lambda_group_lasso_target_groups
from group_lars import group_lars
from group_garrote import group_non_negative_garrote_full
import matplotlib.pyplot as plt

def evaluate_selection(beta, group_indices, true_groups={0, 2, 4}):
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

def perform_experiment(n_repeats=100, true_groups={0, 2, 4}):
    all_results = {
        "Method": [],
        "MSE": [],
        "TPR": [],
        "FPR": []
    }

    for seed in range(n_repeats):
        print(f"Running repeat {seed + 1}/{n_repeats}...")
        X, Y, group_indices = generate_data(seed=seed)

        true_group_count = 3
        best_lambda = select_lambda_group_lasso_target_groups(X, Y, group_indices, true_group_count)
        beta_lasso, pred_lasso = group_lasso(X, Y, group_indices, lambda_=best_lambda)
        mse_lasso = mean_squared_error(Y, pred_lasso)
        tpr_lasso, fpr_lasso = evaluate_selection(beta_lasso, group_indices, true_groups)

        beta_lars, pred_lars = group_lars(X, Y, group_indices)
        mse_lars = mean_squared_error(Y, pred_lars)
        tpr_lars, fpr_lars = evaluate_selection(beta_lars, group_indices, true_groups)

        beta_garrote, pred_garrote, _, _ = group_non_negative_garrote_full(X, Y, group_indices)
        mse_garrote = mean_squared_error(Y, pred_garrote)
        tpr_garrote, fpr_garrote = evaluate_selection(beta_garrote, group_indices, true_groups)

        all_results["Method"] += ["Group Lasso", "Group LARS", "Group Garrote"]
        all_results["MSE"] += [mse_lasso, mse_lars, mse_garrote]
        all_results["TPR"] += [tpr_lasso, tpr_lars, tpr_garrote]
        all_results["FPR"] += [fpr_lasso, fpr_lars, fpr_garrote]

    df_all = pd.DataFrame(all_results)
    df_mean = df_all.groupby("Method").mean().reset_index()

    print("\nAverage Results over {} Repeats:".format(n_repeats))
    print(df_mean)

    return df_mean, df_all



