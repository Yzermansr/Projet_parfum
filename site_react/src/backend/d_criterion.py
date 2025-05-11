import torch
import numpy as np
from optim import barrier_function, find_analytic_center
from itertools import combinations

def hessian_vector_product(f, x, v, A, b, eps=1e-4):
    """
    Approximate the Hessian-vector product H(x) @ v using finite differences:
    H(x) @ v ≈ (∇f(x + εv) - ∇f(x)) / ε
    """
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)

    x = x.detach().clone().requires_grad_(True)
    f_x = f(x, A, b)
    grad_f = torch.autograd.grad(f_x, x, create_graph=False)[0]

    x_eps = (x + eps * v).detach().clone().requires_grad_(True)
    f_x_eps = f(x_eps, A, b)
    grad_f_eps = torch.autograd.grad(f_x_eps, x_eps, create_graph=False)[0]

    return (grad_f_eps - grad_f) / eps


def power_iteration_hvp(f, x, A, b, num_iter=50):
    """
    Approximate the dominant eigenvector of the Hessian using power iteration.
    """
    d = x.shape[0]
    v = torch.randn(d, dtype=torch.float32)
    v = v / v.norm()

    for _ in range(num_iter):
        Hv = hessian_vector_product(f, x, v, A, b)
        v = Hv / Hv.norm()

    return v


def d_criterion_best_question_power(A, b, known_pairs=None):
    center = find_analytic_center(A, b, verbose=False)
    A_tensor = torch.tensor(A, dtype=torch.float32)
    b_tensor = torch.tensor(b, dtype=torch.float32)

    v_max = power_iteration_hvp(barrier_function, center, A_tensor, b_tensor).numpy()

    d = A.shape[1]
    best_score = -1
    best_pair = None
    if known_pairs is None:
        known_pairs = set()

    for i, j in combinations(range(d), 2):
        if (i, j) in known_pairs or (j, i) in known_pairs:
            continue
        direction = np.zeros(d)
        direction[i] = 1
        direction[j] = -1
        direction /= np.linalg.norm(direction)
        alignment = abs(np.dot(direction, v_max))
        if alignment > best_score:
            best_score = alignment
            best_pair = (i, j)

    return best_pair, best_score