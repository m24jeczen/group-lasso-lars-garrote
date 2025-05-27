import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

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

def select_lambda_group_lasso(X, Y, group_indices, lambdas=None, n_splits=5):
    if lambdas is None:
        lambdas = np.logspace(-2, 0.5, 20)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_lambda = None
    best_mse = np.inf

    for lam in lambdas:
        mses = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = Y[train_idx], Y[test_idx]

            beta, _ = group_lasso(X_train, y_train, group_indices, lambda_=lam)
            y_pred = X_test @ beta
            if np.any(np.isnan(y_pred)):
                mse = np.inf
            else:
                mse = mean_squared_error(y_test, y_pred)
            mses.append(mse)

        mean_mse = np.mean(mses)
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_lambda = lam

    return best_lambda, best_mse