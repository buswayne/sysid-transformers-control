import numpy as np
from scipy.signal import lfilter
from statsmodels.tsa.ar_model import AutoReg
from control.matlab import *
import warnings


def u_estimate(u, Tsampling=None):
    """
    Estimates an AR(NN) model for the given output signal of the controller.

    Parameters:
    u (array-like): A column vector of the output signal of the controller.
    Tsampling (float): The time sampling. If None, Tsampling = 1.

    Returns:
    U (TransferFunction): An AR model represented as a transfer function.
    """


    if Tsampling is None:
        Tsampling = 1



    # Convert u to a numpy array
    # u = np.asarray(u)

    # Determine the appropriate order of the AR model
    max_order = 3
    orders = np.arange(1, max_order + 1)

    # Disable warnings temporarily
    warnings.filterwarnings("ignore")

    # Fit AR models and calculate MDL for each order
    mdl_values = []
    n = len(u)
    for order in orders:
        try:
            model = AutoReg(u, lags=order, old_names=False).fit()
            rss = np.sum(model.resid**2)  # Residual sum of squares
            mdl = mdl_criterion(n, rss, order)
            mdl_values.append(mdl)
        except Exception as e:
            mdl_values.append(np.inf)  # If model fitting fails, use infinity for MDL


    # Enable warnings again
    warnings.filterwarnings("default")

    # Select the order with the minimum MDL value
    best_order = orders[np.argmin(mdl_values)]

    # Fit the best AR model
    best_model = AutoReg(u, lags=best_order, old_names=False).fit()

    # Get the denominator coefficients of the AR model
    den = np.concatenate(([1], -best_model.params[1:]))

    # Generate the transfer function in z^-1 form
    U = tf([1, 0], den, Tsampling)

    # Simplify the transfer function using minreal
    U_simplified = minreal(U, verbose=False)

    return U_simplified


def mdl_criterion(n, rss, k):
    """
    Calculate the MDL criterion.

    Parameters:
    n (int): Number of observations.
    rss (float): Residual sum of squares.
    k (int): Number of parameters.

    Returns:
    float: MDL value.
    """
    return n * np.log(rss / n) + k * np.log(n)