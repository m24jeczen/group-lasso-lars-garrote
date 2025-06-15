import numpy as np
from sklearn.linear_model import LinearRegression

def group_non_negative_garrote_full(X, y, groups, tol=1e-12, max_iter=1000):
    n, p = X.shape
    J = len(groups)

    # Step 1: OLS fit
    ols = LinearRegression(fit_intercept=False).fit(X, y)
    beta_ols = ols.coef_

    # Step 2: build matrix Z = [X_1 * beta_OLS_1, ..., X_J * beta_OLS_J]
    Z = np.column_stack([
        X[:, g] @ beta_ols[g] for g in groups
    ])

    # Step 3: initialization
    d = np.zeros(J)
    r = y.copy()
    k = 0
    path = [d.copy()]
    residuals = [r.copy()]
    active_set = []

    while k < max_iter:
        # Step 4: compute correlations
        corr = np.array([
            np.linalg.norm(Z[:, j].T @ r) / len(groups[j]) for j in range(J)
        ])
        inactive = [j for j in range(J) if j not in active_set]
        if not inactive:
            break

        # Step 5: add group with highest correlation to active set
        j_new = inactive[np.argmax(corr[inactive])]
        active_set.append(j_new)

        # Step 6: compute gamma (OLS on active Z columns)
        Z_active = Z[:, active_set]
        gamma_active = np.linalg.pinv(Z_active.T @ Z_active) @ (Z_active.T @ r)
        gamma = np.zeros(J)
        for i, j in enumerate(active_set):
            gamma[j] = gamma_active[i]

        # Step 7: compute step size alpha
        alphas = []
        for j in range(J):
            if j in active_set and gamma[j] < 0:
                alphas.append(-d[j] / gamma[j])
            else:
                alphas.append(1.0)
        alpha = min([a for a in alphas if a > tol] + [1.0])

        # Step 8: update d and residual
        d += alpha * gamma
        r = y - Z @ d
        path.append(d.copy())
        residuals.append(r.copy())

        if alpha == 1.0 or np.linalg.norm(r) < tol:
            break

        k += 1

    # Step 9: Final beta and prediction
    beta_final = np.zeros_like(beta_ols)
    for j, g in enumerate(groups):
        beta_final[g] = beta_ols[g] * d[j]
    y_pred = X @ beta_final

    return beta_final, y_pred, np.array(path), np.array(residuals)

