import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

def locally_weighted_regression(x0, X, y, tau):
    X = np.insert(X, 0, 1, axis=1)
    x0 = np.insert(x0, 0, 1)
    
    m = len(X)
    W = np.zeros((m, m))
    for i in range(m):
        W[i, i] = np.exp(-np.sum((X[i] - x0)**2) / (2 * tau**2))
    
    theta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
    
    y0 = theta @ x0
    
    return y0

np.random.seed(42)
X, y = make_regression(n_samples=100, n_features=1, noise=10)

x_values = np.linspace(np.min(X), np.max(X), 100)
tau = 0.5

predictions = [locally_weighted_regression(np.array([x]), X, y, tau) for x in x_values]

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(x_values, predictions, color='red', label='Locally Weighted Regression')
plt.title('Locally Weighted Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
