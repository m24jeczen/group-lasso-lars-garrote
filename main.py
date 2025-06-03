# from data_generation import generate_data
# from group_lasso import group_lasso, select_lambda_group_lasso
# from group_lars import group_lars
# from group_garrote import group_non_negative_garrote_full
# from sklearn.metrics import mean_squared_error
# import pandas as pd

# X, Y, group_indices = generate_data()

# best_lambda, best_cv_mse = select_lambda_group_lasso(X, Y, group_indices)
# print(f"Best lambda for Group Lasso: {best_lambda:.4f} (CV MSE: {best_cv_mse:.4f})")

# beta_lasso, pred_lasso = group_lasso(X, Y, group_indices, lambda_=best_lambda)
# beta_lars, pred_lars = group_lars(X, Y, group_indices)
# beta_garrote, pred_garrote, d_path, residuals = group_non_negative_garrote_full(X, Y, group_indices)

# mse_lasso = mean_squared_error(Y, pred_lasso)
# mse_lars = mean_squared_error(Y, pred_lars)
# mse_garrote = mean_squared_error(Y, pred_garrote)

# results_df = pd.DataFrame({
#     "Method": ["Group Lasso", "Group LARS", "Group Garrote"],
#     "MSE": [mse_lasso, mse_lars, mse_garrote]
# })

# print("\nModel Comparison:")
# print(results_df)

from data_generation import generate_data
from group_lasso import group_lasso, select_lambda_group_lasso_target_groups
from group_lars import group_lars
from group_garrote import group_non_negative_garrote_full
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

X, Y, group_indices = generate_data()
true_groups = {0, 2, 4} 

true_group_count = 3 
best_lambda = select_lambda_group_lasso_target_groups(X, Y, group_indices, true_group_count)

beta_lasso, pred_lasso = group_lasso(X, Y, group_indices, lambda_=best_lambda)
beta_lars, pred_lars = group_lars(X, Y, group_indices)
beta_garrote, pred_garrote, d_path, residuals = group_non_negative_garrote_full(X, Y, group_indices)

def evaluate_selection(beta, group_indices, true_groups):
    selected = set()
    for i, idxs in enumerate(group_indices):
        if np.linalg.norm(beta[idxs]) > 1e-6:
            selected.add(i)
    tp = len(selected & true_groups)
    fp = len(selected - true_groups)
    fn = len(true_groups - selected)
    tn = len(group_indices) - len(true_groups) - fp
    tpr = tp / len(true_groups)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return tpr, fpr, list(selected)

mse_lasso = mean_squared_error(Y, pred_lasso)
mse_lars = mean_squared_error(Y, pred_lars)
mse_garrote = mean_squared_error(Y, pred_garrote)

tpr_lasso, fpr_lasso, selected_lasso = evaluate_selection(beta_lasso, group_indices, true_groups)
tpr_lars, fpr_lars, selected_lars = evaluate_selection(beta_lars, group_indices, true_groups)
tpr_garrote, fpr_garrote, selected_garrote = evaluate_selection(beta_garrote, group_indices, true_groups)

results_df = pd.DataFrame({
    "Method": ["Group Lasso", "Group LARS", "Group Garrote"],
    "MSE": [mse_lasso, mse_lars, mse_garrote],
    "TPR": [tpr_lasso, tpr_lars, tpr_garrote],
    "FPR": [fpr_lasso, fpr_lars, fpr_garrote],
    "Selected Groups": [selected_lasso, selected_lars, selected_garrote]
})

print("\nModel Comparison:")
print(results_df)

