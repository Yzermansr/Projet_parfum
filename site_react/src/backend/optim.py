import torch
import torch.optim as optim
from torch.autograd.functional import hessian
import numpy as np
from scipy.optimize import linprog
from sklearn.decomposition import PCA
from itertools import combinations


def compute_principal_axes(W, b, n_points=5000):
    """
    Trouve toutes les directions principales du polyèdre défini par Ax <= b
    en utilisant une Analyse en Composantes Principales (PCA).

    Args:
        A (np.ndarray): Matrice des contraintes (m x d)
        b (np.ndarray): Vecteur des bornes (m,)
        n_points (int): Nombre de points à échantillonner à l'intérieur du polyèdre

    Returns:
        tuple:
            - mean (np.ndarray): le centroïde des points échantillonnés
            - components (np.ndarray): vecteurs propres (axes principaux) de dimension (d x d)
            - explained_variance (np.ndarray): variances expliquées par chaque axe
    """
    d = W.shape[1]
    points = []

    for _ in range(n_points * 10):  # plus de tentatives pour compenser la faible densité en haute dimension
        x = np.random.uniform(-1, 1, size=d)
        if np.all(W @ x <= b):
            points.append(x)
            if len(points) >= n_points:
                break

    if len(points) < max(10, d):
        raise ValueError(f"Seulement {len(points)} points trouvés dans le polyèdre, PCA impossible.")

    points = np.array(points)

    # PCA sur toutes les composantes (complet)
    pca = PCA(n_components=d)
    pca.fit(points)

    return pca.mean_, pca.components_, pca.explained_variance_

def find_interior_point(W, b, tensor: bool = False):
    """
    Find an interior point of a polyhedron Ax <= b using SciPy's linear programming solver.

    Args:
        A (torch.Tensor or numpy.ndarray): Constraint matrix
        b (torch.Tensor or numpy.ndarray): Right-hand side vector

    Returns:
        torch.Tensor: A point strictly inside the polyhedron
    """
    print("Finding a point inside the polytope...")
    # Convert to numpy arrays if needed
    if isinstance(W, torch.Tensor):
        W_np = W.detach().numpy()
    else:
        W_np = np.array(W)

    if isinstance(b, torch.Tensor):
        b_np = b.detach().numpy()
    else:
        b_np = np.array(b)

    n_dim = W_np.shape[1]
     # Number of dimensions

    # Set up the LP: max s subject to Ax + s*1 <= b
    # Reformat as standard form for linprog: min c^T x subject to A_ub x <= b_ub

    # We're minimizing -s, which means maximizing s
    c = np.zeros(n_dim + 1)
    c[-1] = -1.0  # The objective is to maximize s

    # Augment A with column of ones (for s)
    W_aug = np.hstack([W_np, np.ones((W_np.shape[0], 1))])

    # Solve the LP
    print(f"b : {b_np.shape}, {W.shape}")
    result = linprog(c, A_ub=W_aug, b_ub=b_np, method='highs')

    if not result.success:
        raise ValueError(f"Failed to find an interior point: {result.message}")

    # Extract the solution (without the slack variable s)
    x_sol = result.x[:-1]
    s_opt = result.x[-1]

    # Check if we found a valid interior point (s > 0)
    if s_opt <= 0:
        raise ValueError("Could not find an interior point. The polyhedron might be empty or degenerate.")

    print("Interior point found !")
    # Convert back to torch tensor
    if tensor:
        return torch.tensor(x_sol, dtype=torch.float32)
    return x_sol

def barrier_function(x, W, b):
    """
    Logarithmic barrier function for the polyhedron Ax <= b.

    Args:
        x (torch.Tensor): Point to evaluate
        A (torch.Tensor): Constraint matrix
        b (torch.Tensor): Right-hand side vector

    Returns:
        torch.Tensor: Value of the barrier function
    """
    return -torch.sum(torch.log(b - W @ x))


def find_analytic_center(W, b, max_iter=500, lr=0.0000000000000001, verbose=True):
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
    
    print("shape =", b.shape)    
    W2 = np.eye(W.shape[1])
    W = np.vstack([W,W2,-W2])
    b = np.hstack([b,np.ones(W.shape[1]),+np.ones(W.shape[1])])
    print(b)
    print(W)
    # Initialize x inside the polyhedron
    x = find_interior_point(W, b) 

    print("Finding analytic center of the polytope...")
    # Verify the initial point is feasible
    margin = b - W @ x
    margin = torch.tensor(margin)
    assert torch.all(margin > 0), f"Initial point must be strictly feasible, margins: {margin}"
    
    x =  torch.tensor(x, requires_grad = True)
    W = torch.tensor(W, requires_grad = False)
    b = torch.tensor(b, requires_grad = False)
    # Setup optimizer
    optimizer = optim.SGD([x], lr=lr)
    
    # Optimization loop
    for i in range(200000):
        optimizer.zero_grad() # reset gradient
        loss = barrier_function(x, W, b) # compute loss
        loss.backward()
        optimizer.step() # step of gradient descent

        # Print progress periodically
        if verbose and i % 100 == 0:
            print(f"Iteration {i}: x = {x.shape}, loss = {loss.item():.4f}")

    print("Center found !")
    return x.detach # detaching the tensor from the current graph

def get_ellipsis_data(hessian, center:torch.Tensor):
    """
    Get the ellipsis data from the barrier function's Hessian.

    Args:
        hessian (tuple): the hessian of the barrier function, as returned by torch.autograd.functional.hessian.
        center (torch.Tensor): the analytical center of the polyhedron

    Returns:
         eigenvalues as two float, eigenvectors as two numpy arrays, center as a numpy array
    """
    print("Computing ellipsis data...")
    Hx = hessian
    
    
    # Calcul des valeurs propres
    eigenvalues, eigenvectors = torch.linalg.eig(Hx)
    print("Ellipsis computed !")
    return eigenvalues, eigenvectors.numpy(), center

def get_pref_data(W, b):
    # center
    center = find_analytic_center(W, b)

    # hessian
    print("Computing hessian...")
    x = torch.tensor(center, dtype=torch.float32)
    W_t = torch.tensor(W, dtype=torch.float32)
    b_t = torch.tensor(b, dtype=torch.float32)

    # Calcul de la Hessienne analytique H = Aᵀ D A
    r = 1.0 / (b_t - W_t @ x)
    D = torch.diag(r ** 2)
    H = W_t.T @ D @ W_t

    # Forcer la symétrie
    H = 0.5 * (H + H.T)

    # Régulariser pour éviter les problèmes d'instabilité numérique
    epsilon = 1e-6
    H += epsilon * torch.eye(H.shape[0])

    # Décomposition spectrale
    eigvals, eigvecs = torch.linalg.eigh(H)
    print("Hessian computed !")
    return center, eigvecs

def get_pref_data_2(W, b):
    # center
    center = find_analytic_center(W, b)

    # hessian
    print("Computing hessian...")
    h = hessian(barrier_function, (torch.tensor(center), torch.tensor(W), torch.tensor(b)))
    Hx = h[0][0]
    print("Inverting hessian...")
    h_inv = torch.linalg.inv(Hx)
    _, eigv, _ = get_ellipsis_data(h_inv, center)
    return center, eigv

