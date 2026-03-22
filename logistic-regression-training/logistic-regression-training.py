import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    x = np.array(X)
    y = np.array(y)
    w, b = np.random.rand(x.shape[1]), 0.1
    for _ in range(500):
        z = x @ w + b
        sigmoid = _sigmoid(z)
        error = sigmoid - y
        gradient_w = x.T @ error  / x.shape[0]
        gradient_b = np.sum(error) / x.shape[0]
        w -= lr*gradient_w
        b -= lr*gradient_b
    return w,b
    pass


x = [[0],[1],[2],[3]]
y = [0,0,1,1]
print(train_logistic_regression(x, y))