from itertools import combinations

import numpy as np
import torch

from comparison import ComparisonMatrix, create_comparison, Perfume


def polyhedron_volume(center, W, b):
    shape = W.get_matrix().shape[1]
    x = torch.tensor(center, dtype=torch.float32)
    W = torch.tensor(W.get_matrix(), dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)

    # W = torch.cat([W, torch.eye(shape), -torch.eye(shape)])
    # b = torch.cat([b, torch.ones(shape), torch.ones(shape)])

    r = b - W @ x  # (m)
    denom = r ** 2
    d = x.shape[0]
    H = torch.zeros((d, d), dtype=x.dtype)

    for i in range(W.shape[0]):
        w_i = W[i].unsqueeze(1)  # (d, 1)
        H += (w_i @ w_i.T) / denom[i]

    # Calcul du déterminant
    det = torch.linalg.det(H)
    return H, det

def get_question(center, W: ComparisonMatrix, P: list[Perfume], b: np.ndarray, target_ratio_gap = 0.3):
    assert np.all(b - W.get_matrix() @ center > 0), "x n'est pas strictement à l'intérieur du polyèdre !"

    # volume of the current polyhedron
    d, vol = polyhedron_volume(center, W, b)
    print(f"déterminant : {d.shape}, {torch.allclose(d, torch.diag(torch.diagonal(d)), atol=1e-8)}")
    print(f"Volume of the current polyhedron: {vol:.4f}")

    p1, p2 = None, None
    nb_iter = 0

    b = np.append(b, 1e-7) # add an item to b to match the dimension of W_copy
    for i, j in combinations(P, 2):

        nb_iter += 1
        new_comp = create_comparison(i, j)
        W_copy = W.insert_comparison(new_comp)

        _, new_vol = polyhedron_volume(center, W_copy, b)
        if nb_iter % 100 == 0:
            print(f"nb iterations : {nb_iter}, ratio : {new_vol/vol}")
        if np.abs(new_vol/vol - 1/2) < target_ratio_gap:
            p1, p2 = i, j
            break

    return p1, p2, nb_iter