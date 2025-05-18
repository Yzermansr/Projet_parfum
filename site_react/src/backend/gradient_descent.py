import numpy as np

from itertools import combinations

from comparison import ComparisonMatrix, create_comparison


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

    axis_unit = ellipsis_axis[0] / np.linalg.norm(ellipsis_axis[0])

    scal = np.abs(0.5 - np.dot(v_unit, axis_unit))
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