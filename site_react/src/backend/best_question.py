import numpy as np
from optim import compute_principal_axes
from auto import get_data2
from auto import get_preference_matrix2


def fitness(ellipsis_axis, center, vector) -> tuple[int, int]:
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
    for axis in ellipsis_axis:
        unit = axis/np.linalg.norm(axis)
        scal = np.dot(v1, unit) 
        if scal < min_scal:
            min_scal = scal

    return distance, 2*np.abs(min_scal-0.5)

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
    d = center.shape
    best_vector = None
    best_score = 10000000


    A,B =  get_preference_matrix2('yzermans')
    l,c = A.shape
    for i in range(l):
        dist_score, col_score = fitness(ellipsis_axis, center,A[i,:])
        total_score = dist_score + col_score
        print(f"score de {i} : {total_score}")
        if total_score < best_score :
            best_score = total_score
            best_vector = i
    n = 0
    for j in A[best_vector,:]:
        n = n + 1
        if (j == 1):
            value1 = n
        if(j == -1):
            value2 = n
    n = 0
    for j in A[25,:]:
        n = n + 1
        if (j == 1):
            print(n)
        if(j == -1):
            print(n)

    return best_vector, value1, value2 
        
