import numpy as np
from numpy.linalg import norm

def group_lasso(X, y, groups, lambda_, max_iter=100, tol=1e-6):
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j, g in enumerate(groups):
            residual = y - X @ beta + X[:, g] @ beta[g]
            S_j = X[:, g].T @ residual
            norm_Sj = norm(S_j)
            threshold = lambda_ * np.sqrt(len(g))
            if norm_Sj <= threshold or np.isnan(norm_Sj):
                beta[g] = 0
            else:
                beta[g] = (1 - threshold / norm_Sj) * S_j
        if norm(beta - beta_old) < tol:
            break
    return beta, X @ beta

def compute_cp(y_true, y_pred, df, sigma2):
    rss = np.sum((y_true - y_pred) ** 2)
    return rss + 2 * sigma2 * df

def select_lambda_group_lasso_target_groups(X, y, groups, lambdas=None):
    if lambdas is None:
        lambdas = np.logspace(0.5, 2, 100)


    # Estimate sigma^2 from full OLS
    beta_full = np.linalg.pinv(X.T @ X) @ X.T @ y
    sigma2 = np.mean((y - X @ beta_full) ** 2)

    best_cp = float('inf')
    best_beta = None
    best_lambda = None

    for lam in lambdas:
        beta, y_pred = group_lasso(X, y, groups, lambda_=lam)
        df = sum(1 for g in groups if norm(beta[g]) > 1e-6)
        cp = compute_cp(y, y_pred, df, sigma2)

        if cp < best_cp:
            best_cp = cp
            best_beta = beta
            best_lambda = lam

    return best_beta, best_lambda
