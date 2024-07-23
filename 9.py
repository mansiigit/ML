import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import inv

def kernel(point, xmat, k):
    weights = np.exp(np.sum((point - xmat)**2, axis=1) / (-2.0 * k**2))
    return np.diag(weights)

def local_weight_regression(xmat, ymat, k):
    m = xmat.shape[0]
    ypred = np.zeros(m)
    for i in range(m):
        weights = kernel(xmat[i], xmat, k)
        W = inv(xmat.T @ weights @ xmat) @ (xmat.T @ weights @ ymat)
        ypred[i] = xmat[i] @ W
    return ypred

data = sns.load_dataset("tips")
X = np.hstack((np.ones((len(data), 1)), np.array(data.total_bill).reshape(-1, 1)))
ytip = np.array(data.tip).reshape(-1, 1)

k = 0.5
ypred = local_weight_regression(X, ytip, k)

sorted_indices = X[:, 1].argsort()
plt.scatter(data.total_bill, data.tip, color='green')
plt.plot(X[sorted_indices, 1], ypred[sorted_indices], color='red', linewidth=2)
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.show()
