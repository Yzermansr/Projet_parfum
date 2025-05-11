import numpy as np


def regret(P, W, j):
    first = True
    for w in W:
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
    return w_max, alt, max_w, indice_global


def min_regret(P, W, vectors=True):
    for j in range(P.shape[0]):
        w, A, r_j, a = regret(P, W, j)

        if (j == 0):
            regret_min = r_j
            Parfum_A = A
            Parfum_B = P[j]
            poids = w
        if (np.linalg.norm(regret_min) > np.linalg.norm(r_j)):
            regret_min = r_j
            Parfum_A = A
            Parfum_B = j
            poids = w
    if vectors:
        return a, Parfum_B, regret_min
    return


def regret_2(P, W, v):
    maxi = -np.inf
    maxi_index = 0
    for w in range(W.shape[0]):
        for j in range(P.shape[0]):
            regret = np.dot(W[w], P[j, :]) - np.dot(W[w], v)
            if regret > maxi:
                maxi = regret
                maxi_index = j
    return maxi, maxi_index


def min_regret_2(P, W):
    min_index = np.inf
    min_regret = 1000
    best_opponent = 0
    # iterate through perfumes
    for v in range(1, P.shape[0]):
        regret, opp = regret_2(P, W, P[v, :])
        if regret < min_regret:
            min_index = v
            best_opponent = opp
            min_regret = regret
    return min_index, best_opponent, min_regret


def condition_darret(P, W, Nmax, eps=10 ** -2):
    n = 0
    while (n < Nmax):
        _, parfum, regret_min = min_regret(P, W)
        if (np.linalg.norm(min_regret) < eps):
            return parfum
        W = nouvelle_question(P, W)
        n = n + 1
    print('Nombre maximum de quesiton posée')
    return parfum