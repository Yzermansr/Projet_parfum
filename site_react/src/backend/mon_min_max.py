import numpy as np

from comparison import Perfume, ComparisonMatrix


def regret(P: np.ndarray, W: ComparisonMatrix, j: int):
    first = True  
    for w in W:
        w = w.get_vector()
        fw_j = np.dot(w, P[j, :])
        Pbis = np.delete(P, j, axis=0)
        max_i = [np.dot(w, Pbis[i, :]) for i in range(Pbis.shape[0])]
        max_value = max_i[0]
        indice = 0
        for n, value in enumerate(max_i):
            if value > max_value:
                max_value = value
                indice = n
        indice_global = indice if indice < j else indice + 1
        fw_max = P[indice_global, :]  
        max_wj = fw_max - P[j, :]    
        if first:
            max_w = max_wj
            w_max = w
            alt = fw_max
            first = False
        elif np.linalg.norm(max_wj) > np.linalg.norm(max_w):
            max_w = max_wj
            w_max = w
            alt = fw_max
    return w_max, alt, max_w

def min_regret(P: list[Perfume], W: ComparisonMatrix):
    P_mat = np.array([p.get_vector() for p in P])
    for j in range(P_mat.shape[0]):
        w,A,r_j = regret(P_mat,W,j)
        if j==0:
            regret_min = r_j
            Parfum_A = A
            Parfum_B = P_mat[j]
            poids = w
        if np.linalg.norm(regret_min) > np.linalg.norm(r_j):
            regret_min = r_j
            Parfum_A = A
            Parfum_B = P_mat[j]
            poids = w
    return Parfum_A,Parfum_B,regret_min
