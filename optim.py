import torch
import torch.optim as optim
from torch.autograd.functional import hessian
import numpy as np
from scipy.optimize import linprog
from sklearn.decomposition import PCA

def compute_principal_axes(A, b):
    """
    Trouve les directions principales du polyèdre en utilisant l'Analyse en Composantes Principales (PCA).
    """
    # Générer des points à l'intérieur du polyèdre
    points = []
    for _ in range(5000):  # Échantillonnage
        x = np.random.uniform(-2, 2, size=(2,))
        if np.all(A @ x <= b):
            points.append(x)

    points = np.array(points)

    # Appliquer PCA pour obtenir les directions principales
    pca = PCA(n_components=2)
    pca.fit(points)
    return pca.mean_, pca.components_, pca.explained_variance_

def find_interior_point(A, b):
    """
    Find an interior point of a polyhedron Ax <= b using SciPy's linear programming solver.

    Args:
        A (torch.Tensor or numpy.ndarray): Constraint matrix
        b (torch.Tensor or numpy.ndarray): Right-hand side vector

    Returns:
        torch.Tensor: A point strictly inside the polyhedron
    """
    # Convert to numpy arrays if needed
    if isinstance(A, torch.Tensor):
        A_np = A.detach().numpy()
    else:
        A_np = np.array(A)

    if isinstance(b, torch.Tensor):
        b_np = b.detach().numpy()
    else:
        b_np = np.array(b)

    n_dim = A_np.shape[1]  # Number of dimensions

    # Set up the LP: max s subject to Ax + s*1 <= b
    # Reformat as standard form for linprog: min c^T x subject to A_ub x <= b_ub

    # We're minimizing -s, which means maximizing s
    c = np.zeros(n_dim + 1)
    c[-1] = -1.0  # The objective is to maximize s

    # Augment A with column of ones (for s)
    A_aug = np.hstack([A_np, np.ones((A_np.shape[0], 1))])

    # Solve the LP
    result = linprog(c, A_ub=A_aug, b_ub=b_np, method='highs')

    if not result.success:
        raise ValueError(f"Failed to find an interior point: {result.message}")

    # Extract the solution (without the slack variable s)
    x_sol = result.x[:-1]
    s_opt = result.x[-1]

    # Check if we found a valid interior point (s > 0)
    if s_opt <= 0:
        raise ValueError("Could not find an interior point. The polyhedron might be empty or degenerate.")

    # Convert back to torch tensor
    return torch.tensor(x_sol, dtype=torch.float32)

def barrier_function(x, A, b):
    """
    Logarithmic barrier function for the polyhedron Ax <= b.

    Args:
        x (torch.Tensor): Point to evaluate
        A (torch.Tensor): Constraint matrix
        b (torch.Tensor): Right-hand side vector

    Returns:
        torch.Tensor: Value of the barrier function
    """
    return -torch.sum(torch.log(b - A @ x))


def find_analytic_center(A, b, max_iter=1000, lr=0.1, verbose=True):
    """
    Find the analytic center of a polyhedron using barrier method.

    Args:
        A (torch.Tensor): Constraint matrix
        b (torch.Tensor): Right-hand side vector
        max_iter (int): Maximum number of iterations
        lr (float): Learning rate for optimizer
        verbose (bool): Whether to print progress

    Returns:
        torch.Tensor: Approximate analytic center
    """

    # Initialize x inside the polyhedron
    x = torch.tensor(find_interior_point(A, b), requires_grad=True)

    # Verify initial point is feasible
    margin = b - A @ x
    assert torch.all(margin > 0), f"Initial point must be strictly feasible, margins: {margin}"

    # Setup optimizer
    optimizer = optim.SGD([x], lr=lr)

    # Optimization loop
    for i in range(max_iter):
        optimizer.zero_grad() # reset gradient
        loss = barrier_function(x, A, b)
        loss.backward()
        optimizer.step() # step of gradient descent

        # Print progress periodically
        if verbose and i % 100 == 0:
            print(f"Iteration {i}: x = {x.detach().numpy()}, loss = {loss.item():.4f}")

    return x.detach() # detaching the tensor from current graph



