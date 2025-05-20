from itertools import combinations
from concurrent.futures import ProcessPoolExecutor

import numpy as np


from comparison import ComparisonMatrix, Perfume, create_comparison
from gradient_descent import get_question_d_criterion_1
from d_criterion import get_question
from min_regret import min_max_regret
from optim import get_pref_data, get_pref_data2


def evaluate_pair(args):
    i, j, ellipsis_axis, center = args
    comp = create_comparison(i, j)
    vect = comp.get_vector()
    distance, scal = fitness(ellipsis_axis, center, vect)
    total = distance + scal
    return total, vect, i, j

def get_question_d_criterion_1_parallel(ellipsis_axis, center, W: ComparisonMatrix, b):
    perfumes = {c.p1 for c in W} | {c.p2 for c in W}
    all_pairs = list(combinations(perfumes, 2))
    args = [(i, j, ellipsis_axis, center) for i, j in all_pairs]

    best = None

    with ProcessPoolExecutor() as executor:
        for result in executor.map(evaluate_pair, args):
            total, vect, i, j = result
            if best is None or total < best[0]:
                best = (total, vect, i, j)

    if best:
        return best[1], best[2], best[3]
    return None, None, None



def get_best_question(method: str, W: ComparisonMatrix, b: np.ndarray, P: list[Perfume]):
    """
    Find the best question using the given method to find the best perfume

    Args:
        method (str): the method to use
        W (ComparisonMatrix): the comparison matrix
        b (np.ndarray): the right member
        P (list[Perfume]): the list of perfumes to compare

    Returns:
        tuple[int, int, float]: the perfumes to compare, and the regret
    """

    match method:
        case "d criterion" | "d-criterion":
            center, ellipsis_axis = get_pref_data2(W.get_matrix(), b)
            return get_question(center, W, P, b)

        case "min max regret" | "min-max-regret" | "minmax":
            x, y, _, regret = min_max_regret(P, W)
            return x, y, regret

        case "gradient descent" | "gradient-descent":
            center, ellipsis_axis = get_pref_data(W.get_matrix(), b)
            _, x, y = get_question_d_criterion_1(ellipsis_axis, center, W, b)
            return x, y, -1

        case "newton gradient descent" | "newton descent" | "newton":
            center_newton, ellipsis_axis = get_pref_data2(W.get_matrix(), b)
            _, x, y = get_question_d_criterion_1(P,ellipsis_axis, center_newton, W, b)
            return x, y, -1

        case _:
            raise ValueError(f"Unknown method: {method}")

    return None

