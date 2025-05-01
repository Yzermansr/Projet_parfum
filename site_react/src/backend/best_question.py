import numpy as np
from optim import compute_principal_axes


def fitness(ellipsis_axis, center, vector) -> (int, int):
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
    for i in range(vector.shape[0]-1):
        num += vector[i]*center[i]
    num += vector[-1]
    num = np.abs(num)

    denom = np.linalg.norm(vector)
    distance = num/denom

    # colinearity
    # make unit vectors
    v1 = vector/np.linalg.norm(vector)
    min_scal = 100
    for i in range(ellipsis_axis.shape[0]):
        unit = ellipsis_axis[0]/np.linalg.norm(ellipsis_axis[0])
        scal = np.dot(v1, unit)
        if scal < min_scal:
            min_scal = scal

    return distance, 2*np.abs(scal-0.5)

def get_best_question(ellipsis_axis, center):
    """
        Find the best question (with the highest fitness scores) to find the best perfume

        Args:
            ellipsis_axis (numpy.ndarray): Main axis of the Sonnevend ellipsis
            center (numpy.ndarray): the center of the ellipsis

        Returns:
            numpy.ndarray : the best question as a vector
        """
    # parcourir toutes les questions possibles et trouver celle avec le meilleur score de fitness
    return "au top"