import numpy as np

from comparison import Perfume, ComparisonMatrix


def regret(perfumes: list[Perfume], comparison_matrix: ComparisonMatrix, j: int):
    first = True

    for comparison in comparison_matrix:
        w = comparison.get_vector()
        pj = perfumes[j].get_vector()
        fw_j = np.dot(w, pj)

        # Supprimer le j-ème parfum de la liste
        perfumes_bis = perfumes[:j] + perfumes[j+1:]

        # Calculer dot(w, p) pour chaque parfum ≠ j
        max_value = None
        indice = None

        for i, p in enumerate(perfumes_bis):
            val = np.dot(w, p.get_vector())
            if (max_value is None) or (val > max_value):
                max_value = val
                indice = i

        # Retrouver l’indice global (car on a supprimé j)
        indice_global = indice if indice < j else indice + 1

        # Calcul du vecteur de regret
        alt = perfumes[indice_global]
        max_wj = alt.get_vector() - pj

        if first:
            max_w = max_wj
            w_max = w
            best_alt = alt
            first = False
        elif np.linalg.norm(max_wj) > np.linalg.norm(max_w):
            max_w = max_wj
            w_max = w
            best_alt = alt

    return w_max, best_alt, max_w, perfumes[indice_global]



def min_regret(P: list[Perfume], W: ComparisonMatrix, vectors=True):
    regret_min = np.inf
    Parfum_A = None
    Parfum_B = None
    poids = None
    for j in range(len(P)):
        w, A, r_j, a = regret(P, W, j)

        if (np.linalg.norm(regret_min) > np.linalg.norm(r_j)):
            regret_min = r_j
            Parfum_A = a
            Parfum_B = P[j]
            poids = w
    print(type(Parfum_A), type(Parfum_B))
    if vectors:
        return Parfum_A, Parfum_B, regret_min
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