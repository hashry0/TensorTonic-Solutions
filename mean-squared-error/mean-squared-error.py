import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    loss = np.sum(1/y_pred.shape[0] * (y_pred - y_true)**2)
    return loss
