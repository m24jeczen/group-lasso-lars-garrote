from sklearn.linear_model import LinearRegression
import numpy as np

def group_garrote(X, y, groups, lam=0.5):
    ols = LinearRegression(fit_intercept=False).fit(X, y)
    beta_ols = ols.coef_
    d = np.ones(len(groups))
    for j, g in enumerate(groups):
        group_norm = np.linalg.norm(beta_ols[g])
        if group_norm > 0:
            shrink = max(0, 1 - lam * len(g) / group_norm)
            d[j] = shrink
        else:
            d[j] = 0
    beta_final = np.zeros_like(beta_ols)
    for j, g in enumerate(groups):
        beta_final[g] = beta_ols[g] * d[j]
    return beta_final, X @ beta_final