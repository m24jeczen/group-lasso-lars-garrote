import numpy as np
from sklearn.preprocessing import OneHotEncoder, normalize

def generate_data(seed=0):
    np.random.seed(seed)
    n_samples = 50
    n_factors = 15
    rho = 0.5

    cov_matrix = rho ** np.abs(np.subtract.outer(np.arange(n_factors), np.arange(n_factors)))
    Z = np.random.multivariate_normal(mean=np.zeros(n_factors), cov=cov_matrix, size=n_samples)

    Z_discrete = np.zeros_like(Z, dtype=int)
    for i in range(n_factors):
        q1, q2 = np.percentile(Z[:, i], [33.33, 66.67])
        Z_discrete[:, i] = np.digitize(Z[:, i], bins=[q1, q2])

    def simulate_response(Zd):
        y = (
            1.8 * (Zd[:, 0] == 1) - 1.2 * (Zd[:, 0] == 0) +
            1.0 * (Zd[:, 2] == 1) + 0.5 * (Zd[:, 2] == 0) +
            1.0 * (Zd[:, 4] == 1) + 1.0 * (Zd[:, 4] == 0)
        )
        noise = np.random.normal(0, np.std(y), size=y.shape)
        return y + noise

    Y = simulate_response(Z_discrete)

    encoder = OneHotEncoder(drop='first', sparse_output=False)
    X = encoder.fit_transform(Z_discrete)
    X -= X.mean(axis=0)
    X = normalize(X, axis=0)
    Y -= Y.mean()

    group_indices = []
    for i in range(0, X.shape[1], 2):
        group_indices.append(np.array([i, i+1]))

    return X, Y, group_indices