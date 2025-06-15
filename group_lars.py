import numpy as np
from numpy.linalg import pinv, norm

def compute_cp(y_true, y_pred, df, sigma2):
    rss = np.sum((y_true - y_pred) ** 2)
    return rss + 2 * sigma2 * df

def group_lars(X, y, groups, tol=2.5):
    n, p = X.shape
    J = len(groups)
    beta = np.zeros(p)
    residual = y.copy()
    active_groups = []
    beta_path = []
    cp_values = []

    full_beta = pinv(X.T @ X) @ X.T @ y
    sigma2 = np.mean((y - X @ full_beta) ** 2)

    while True:
        corr = np.array([
            norm(X[:, g].T @ residual) / np.sqrt(len(g)) if j not in active_groups else -np.inf
            for j, g in enumerate(groups)
        ])
        j_new = np.argmax(corr)

        if corr[j_new] == -np.inf or corr[j_new] < tol:
            break

        active_groups.append(j_new)
        active_indices = np.concatenate([groups[j] for j in active_groups])
        X_active = X[:, active_indices]

        gamma = pinv(X_active.T @ X_active) @ X_active.T @ y
        beta_new = np.zeros(p)
        beta_new[active_indices] = gamma
        beta_path.append(beta_new.copy())

        y_pred = X @ beta_new
        cp = compute_cp(y, y_pred, df=len(active_groups), sigma2=sigma2)
        cp_values.append(cp)

        beta = beta_new
        residual = y - y_pred

        if norm(residual) < tol:
            break

    # Step 4: Select model with lowest Cp
    min_cp_idx = np.argmin(cp_values)
    best_beta = beta_path[min_cp_idx]
    best_pred = X @ best_beta

    return best_beta, best_pred
