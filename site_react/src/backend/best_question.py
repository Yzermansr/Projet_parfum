from itertools import combinations
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from min_regret import min_regret, min_regret_2
from comparison import ComparisonMatrix, Comparison, Perfume, create_comparison

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

def fitness(ellipsis_axis, center, vector) -> tuple[int, int]:
    # TODO: trouver le vrai nom de ce système
    """
    Calculates two fitness scores between the vector and the axis of the Sonnevend ellipsis.

    Args:
        ellipsis_axis (numpy.ndarray): Main axis of the Sonnevend ellipsis
        vector (numpy.ndarray): the tried vector
        center (numpy.ndarray): the center of the ellipsis

    Returns:
        tuple: a score of distance to the polyhedron centroid and a score of
        colinearity of the vector to the axis
    """

    # Distance au centre du polyèdre (projection de center sur vector)
    # print('ici')
    min_scalar = np.inf
    distance = np.abs(np.dot(vector, center) + vector[-1]) / np.linalg.norm(vector)

    # Colinéarité minimale entre le vecteur testé et les axes principaux
    v_unit = vector / np.linalg.norm(vector)

    scal = np.abs(0.5 - np.dot(v_unit, ellipsis_axis[0] / np.linalg.norm(ellipsis_axis[0])))
    if min_scalar > scal:
        min_scalar = scal

    return distance, min_scalar

def get_question_d_criterion_1(ellipsis_axis, center, W: ComparisonMatrix, b):
    perfumes = {c.p1 for c in W} | {c.p2 for c in W}  # set de tous les parfums comparés
    print(len(perfumes))
    best_score = float("inf")
    best_vector = None
    best_A = None
    best_B = None

    for i, j in combinations(perfumes, 2):
        comp = create_comparison(i, j)
        vect = comp.get_vector()
        distance, scal = fitness(ellipsis_axis, center, vect)
        total = distance + scal
        if total < best_score:
            best_score = total
            best_vector = vect
            best_A = i
            best_B = j

    return best_vector, best_A, best_B

def get_best_question(ellipsis_axis, center, W, b, P):
    """
        Find the best question (with the highest fitness scores) to find the best perfume

        Args:
            ellipsis_axis (numpy.ndarray): Main axis of the Sonnevend ellipsis
            center (numpy.ndarray): the center of the ellipsis
            W (numpy.ndarray): Constraint matrix
            b (numpy.ndarray): Right-hand side vector
            P (numpy.ndarray): Perfumes matrix

        Returns:
            numpy.ndarray : the best question as a vector
        """
    print("Finding the best question...")
    best_vector, dc1p1, dc1p2 = get_question_d_criterion_1(ellipsis_axis, center, W, b)

    # pair, _ = d_criterion_best_question_power(W, b)
    # pa, pb, r = min_regret(P, W)
    #p1, p2, r1 = min_regret_2(P, W)
    print(f"Best question Victor Pez : {dc1p1}, {dc1p2}")
    # print(f"Best question Soupramagoat : {pair[0]}, {pair[1]}")
    # print(f"Best question Yzermans : {pa}, {pb}, regret : {np.linalg.norm(r)}")
    #print(f"Best question Victor Pez Le Retour: {p1}, {p2}, regret : {r1}")
    return "aaa"

