import numpy as np

def group_lasso(X, y, groups, lambda_, max_iter=100, tol=1e-6):
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j, g in enumerate(groups):
            residual = y - X @ beta + X[:, g] @ beta[g]
            S_j = X[:, g].T @ residual
            norm_Sj = np.linalg.norm(S_j)
            threshold = lambda_ * np.sqrt(len(g))
            if norm_Sj <= threshold or np.isnan(norm_Sj):
                beta[g] = 0
            else:
                shrinkage = 1 - threshold / norm_Sj
                beta_g_update = shrinkage * S_j
                if np.linalg.norm(beta_g_update) > 1e3:
                    beta[g] = 0
                else:
                    beta[g] = beta_g_update
        if np.linalg.norm(beta - beta_old) < tol:
            break
    return beta, X @ beta


def select_lambda_group_lasso_target_groups(X, Y, group_indices, true_group_count, lambdas=None):
    if lambdas is None:
        lambdas = np.logspace(-2, 1, 100)

    best_lambda = None
    min_diff = float('inf')

    for lam in lambdas:
        beta, _ = group_lasso(X, Y, group_indices, lambda_=lam)
        selected_groups = 0
        for g in group_indices:
            if np.linalg.norm(beta[g]) > 1e-6:
                selected_groups += 1
        diff = abs(selected_groups - true_group_count)
        if diff < min_diff:
            min_diff = diff
            best_lambda = lam
            if diff == 0:  
                break

    return best_lambda
