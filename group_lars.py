import numpy as np
from numpy.linalg import pinv, norm
from sklearn.linear_model import LinearRegression

def group_lars(X, y, groups, max_steps=None, tol=1e-6):
    n, p = X.shape
    G = len(groups)
    beta = np.zeros(p)
    residual = y.copy()
    active_groups = []
    step = 0

    while True:
        if max_steps is not None and step >= max_steps:
            break

        # Step 2: Find most correlated group (normalized by group size)
        correlations = [
            norm(X[:, g].T @ residual)**2 / len(g) if j not in active_groups else -np.inf
            for j, g in enumerate(groups)
        ]
        best_group = np.argmax(correlations)

        if correlations[best_group] == -np.inf:
            break

        active_groups.append(best_group)

        # Step 3: Form X_Ak and compute direction gamma
        active_indices = np.concatenate([groups[j] for j in active_groups])
        X_Ak = X[:, active_indices]

        gamma_Ak = pinv(X_Ak.T @ X_Ak) @ (X_Ak.T @ residual)
        gamma = np.zeros(p)
        gamma[active_indices] = gamma_Ak

        # Step 4: Find minimum alpha before a new group reaches same correlation
        alpha = 1.0
        for j, g in enumerate(groups):
            if j in active_groups:
                continue
            Xg = X[:, g]
            S1 = norm(Xg.T @ residual)**2 / len(g)
            S2 = norm(Xg.T @ (residual - X @ gamma))**2 / len(g)

            # Solve: S1 = (1 - alpha)^2 * S1 + alpha^2 * S2  => quadratic in alpha
            a = S2 - 2*S1 + S1
            b = 2*S1 - 2*S2
            c = S1 - S1
            disc = b**2 - 4*a*c

            if disc >= 0 and a != 0:
                alpha_candidate = (-b + np.sqrt(disc)) / (2*a)
                if 0 < alpha_candidate < alpha:
                    alpha = alpha_candidate

        # Step 6: Update beta and residual
        beta += alpha * gamma
        residual = y - X @ beta
        step += 1

        if norm(residual) < tol:
            break

    return beta, X @ beta
