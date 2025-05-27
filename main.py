from data_generation import generate_data
from group_lasso import group_lasso, select_lambda_group_lasso
from group_lars import group_lars
from group_garrote import group_garrote
from sklearn.metrics import mean_squared_error
import pandas as pd

# Wczytaj dane
X, Y, group_indices = generate_data()

# Dobierz lambda dla Group Lasso
best_lambda, best_cv_mse = select_lambda_group_lasso(X, Y, group_indices)
print(f"Best lambda for Group Lasso: {best_lambda:.4f} (CV MSE: {best_cv_mse:.4f})")

# Zastosuj metody
beta_lasso, pred_lasso = group_lasso(X, Y, group_indices, lambda_=best_lambda)
beta_lars, pred_lars = group_lars(X, Y, group_indices, max_groups=5)
beta_garrote, pred_garrote = group_garrote(X, Y, group_indices, lam=0.5)

# Oblicz MSE
mse_lasso = mean_squared_error(Y, pred_lasso)
mse_lars = mean_squared_error(Y, pred_lars)
mse_garrote = mean_squared_error(Y, pred_garrote)

# Tabela wyników
results_df = pd.DataFrame({
    "Method": ["Group Lasso", "Group LARS", "Group Garrote"],
    "MSE": [mse_lasso, mse_lars, mse_garrote]
})

print("\nModel Comparison:")
print(results_df)
