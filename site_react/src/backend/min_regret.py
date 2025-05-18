import numpy as np
from scipy.optimize import linprog
from comparison import Perfume, ComparisonMatrix


def maximum_pairwise_regret(X: Perfume, Y: Perfume, P_matrix: np.ndarray):
    """
    Calculate Maximum Pairwise Regret between alternatives X and Y.
    Maximizes w*(Y-X) subject to P*w > 0

    Parameters:
        X, Y: Perfume objects (with .get_vector())
        P_matrix: constraint matrix (numpy ndarray)

    Returns:
        w_opt: optimal weight vector
        max_regret: regret value
        success: whether linprog succeeded
    """
    x_vec = X.get_vector()
    y_vec = Y.get_vector()
    c = y_vec - x_vec
    c_neg = -c

    A_ub = -P_matrix
    b_ub = np.zeros(P_matrix.shape[0])

    n_criteria = len(x_vec)
    A_eq = np.ones((1, n_criteria))
    b_eq = np.ones(1)

    bounds = [(0, None) for _ in range(n_criteria)]

    result = linprog(c_neg, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    if result.success:
        w_opt = result.x
        max_regret = -result.fun
        return w_opt, max_regret, True
    else:
        return None, 0, False


def maximum_regret(X: Perfume, alternatives: list[Perfume], P_matrix: np.ndarray):
    """
    For a given X, find the alternative Y that maximizes regret.

    Returns:
        Y with max regret against X
        w_opt: weight vector
        max_regret_value
    """
    max_regret_value = -float('inf')
    max_regret_alt = None
    max_regret_weights = None

    for Y in alternatives:
        if X == Y:
            continue
        w_opt, regret_value, success = maximum_pairwise_regret(X, Y, P_matrix)

        if success and regret_value > max_regret_value:
            max_regret_value = regret_value
            max_regret_alt = Y
            max_regret_weights = w_opt

    return max_regret_alt, max_regret_weights, max_regret_value


def min_max_regret(alternatives: list[Perfume], P):
    """
    Find the alternative X that minimizes its maximum regret.

    Returns:
        X_min: alternative with minimum max regret
        Y_max: adversarial alternative
        w_opt: weight vector producing max regret
        min_max_regret_value: scalar
    """
    # Assure P is a matrix
    P_matrix = P.get_matrix() if hasattr(P, "get_matrix") else P

    min_regret_value = float('inf')
    min_regret_alt = None
    max_regret_alt = None
    max_regret_weights = None

    for X in alternatives:
        Y, w_opt, regret_value = maximum_regret(X, alternatives, P_matrix)

        if regret_value < min_regret_value:
            min_regret_value = regret_value
            min_regret_alt = X
            max_regret_alt = Y
            max_regret_weights = w_opt

    return min_regret_alt, max_regret_alt, max_regret_weights, min_regret_value
