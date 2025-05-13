import numpy as np

from d_criterion import d_criterion_best_question_power
from min_regret import min_regret, min_regret_2
from auto import get_perfumes_from_constraint


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

    # distance to centroid
    num = 0
    for i in range(vector.shape[0] - 1):
        num += vector[i] * center[i]
    num += vector[-1]
    num = np.abs(num)

    denom = np.linalg.norm(vector)
    distance = num / denom

    # colinearity
    # make unit vectors
    v1 = vector / np.linalg.norm(vector)
    min_scal = 100
    for axis in ellipsis_axis:
        unit = axis / np.linalg.norm(axis)
        scal = np.dot(v1, unit)
        if scal < min_scal:
            min_scal = scal

    return distance, scal

def get_question_d_criterion_1(ellipsis_axis, center, W, b):
    # parcourir toutes les questions possibles et trouver celle avec le meilleur score de fitness
    d = center.shape
    best_vector = None
    # best_vector_PCA = None
    best_score = 10000000
    # best_score_PCA = 10000000
    # d-criterion (d-error criterion) pour comparer la manière dont notre polyèdre diminue par rapport aux questions aléatoires

    # TODO: GROS PROBLÈME, ON PARCOURT QUE LES QUESTIONS DÉJÀ POSÉES
    # TODO: JVAIS BOSSER DESSUS
    l, c = W.get_matrix().shape
    for i in range(l):
        dist_score, col_score = fitness(ellipsis_axis, center, W[i].get_vector())
        total_score = dist_score + col_score
        # print(f"score de {i} : {total_score}")
        if total_score < best_score:
            best_score = total_score
            best_comparison = W[i]

    n = 0

    # TODO: ça marche plus
    for j in W.get_matrix()[best_vector, :]:
        n = n + 1
        if (j == 1):
            value1 = n
        if (j == -1):
            value2 = n

    return best_vector, value1, value2

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
    pa, pb, r = min_regret(P, W)
    #p1, p2, r1 = min_regret_2(P, W)
    print(f"Best question Victor Pez : {dc1p1}, {dc1p2} (comparaison : {best_vector}, ids ? : {get_perfumes_from_constraint(best_vector)})")
    # print(f"Best question Soupramagoat : {pair[0]}, {pair[1]}")
    print(f"Best question Yzermans : {pa}, {pb}, regret : {np.linalg.norm(r)}")
    #print(f"Best question Victor Pez Le Retour: {p1}, {p2}, regret : {r1}")
    return "aaa"

