from sklearn.linear_model import LinearRegression
import numpy as np

def group_lars(X, y, groups, max_groups=None):
    residual = y.copy()
    selected_groups = []
    beta = np.zeros(X.shape[1])
    for _ in range(max_groups or len(groups)):
        correlations = [
            np.linalg.norm(X[:, g].T @ residual) / len(g) if j not in selected_groups else -np.inf
            for j, g in enumerate(groups)
        ]
        best_group = np.argmax(correlations)
        if correlations[best_group] == -np.inf:
            break
        selected_groups.append(best_group)
        X_sub = np.hstack([X[:, g] for j, g in enumerate(groups) if j in selected_groups])
        model = LinearRegression(fit_intercept=False).fit(X_sub, y)
        coef = model.coef_
        index = 0
        for j in selected_groups:
            g = groups[j]
            beta[g] = coef[index:index + len(g)]
            index += len(g)
        residual = y - X @ beta
    return beta, X @ beta