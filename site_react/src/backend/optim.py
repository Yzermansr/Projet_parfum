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

def find_interior_point(W: np.ndarray, b: np.ndarray, tensor: bool = False):
    """
    Find an interior point of a polyhedron Ax <= b using SciPy's linear programming solver.

    Args:
        W (numpy.ndarray): Constraint matrix
        b (numpy.ndarray): Right-hand side vector
        tensor (bool): Whether to return a torch.Tensor or a numpy.ndarray. Defaults to False.

    Returns:
        torch.Tensor: A point strictly inside the polyhedron
    """
    print("Finding a point inside the polytope...")

    # Convert to numpy arrays if needed

    n_dim = W.shape[1]
     # Number of dimensions

    # Set up the LP: max s subject to Ax + s*1 <= b
    # Reformat as standard form for linprog: min c^T x subject to A_ub x <= b_ub

    # We're minimizing -s, which means maximizing s
    c = np.zeros(n_dim + 1)
    c[-1] = -1.0  # The objective is to maximize s

    # Augment A with column of ones (for s)
    W_aug = np.hstack([W, np.ones((W.shape[0], 1))])
    print(W.shape)

    # Solve the LP
    print(f"b.shape = {b.shape}, W.shape = {W.shape}")
    result = linprog(c, A_ub=W_aug, b_ub=b, method='highs')

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

def barrier_function(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Logarithmic barrier function for the polyhedron Ax <= b.

    Args:
        x (np.ndarray): Point to evaluate
        W (np.ndarray): Constraint matrix
        b (np.ndarray): Right-hand side vector

    Returns:
        torch.Tensor : Value of the barrier function
    """

    return -torch.sum(torch.log(b - W @ x))


def find_analytic_center(W: np.ndarray, b:np.ndarray, max_iter=1001, lr=0.00000000000001, verbose=True) -> np.ndarray:
    """
    Find the analytic center of a polyhedron using barrier method.

    Args:
        W (torch.Tensor): Constraint matrix
        b (torch.Tensor): Right-hand side vector
        max_iter (int): Maximum number of iterations
        lr (float): Learning rate for optimizer
        verbose (bool): Whether to print progress

    Returns:
        np.ndarray: Approximate analytic center
    """
    print(f"find_analytical_center : {b.shape}, {W.shape}")

    # W2 = np.eye(W.shape[1])
    # W = np.vstack([W,W2,-W2])
    # b = np.hstack([b,np.ones(W.shape[1]),+np.ones(W.shape[1])])

    W = np.vstack([W, -np.ones((1, W.shape[1]))])
    b = np.hstack([b, 1e-7])  # ε > 0

    print(f"find_analytical_center 2: {b.shape}, {W.shape}")
    # Initialize x inside the polyhedron
    x = find_interior_point(W, b)
    # x = torch.tensor(x, dtype = torch.float32, requires_grad = True)

    print("Finding analytic center of the polytope...")
    # Verify the initial point is feasible
    margin = b - W @ x
    margin = torch.tensor(margin)
    assert torch.all(margin > 0), f"Initial point must be strictly feasible, margins: {margin}"
    
    x =  torch.tensor(x, dtype = torch.float32, requires_grad = True)
    W = torch.tensor(W, dtype = torch.float32, requires_grad = False)
    b = torch.tensor(b, dtype = torch.float32, requires_grad = False)
    # Setup optimizer
    optimizer = optim.SGD([x], lr=lr)
    
    # Optimization loop
    for i in range(max_iter):
        optimizer.zero_grad() # reset gradient
        loss = barrier_function(x, W, b) # compute loss
        loss.backward()
        optimizer.step() # step of gradient descent

        # Print progress periodically
        if verbose and i % 100 == 0:
            print(f"Iteration {i}: x = {x.shape}, loss = {loss.item():.4f}")

    print("Center found !")
    return x.detach().numpy() # detaching the tensor from the current graph

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




def find_analytic_center_newton(W, b, max_iter=100, tol=1e-6, verbose=True):
    """
    Trouve le centre analytique du polyèdre Wx <= b en utilisant la méthode de Newton.

    Args:
        W (np.ndarray or torch.Tensor): Matrice des contraintes
        b (np.ndarray or torch.Tensor): Vecteur des bornes
        max_iter (int): Nombre maximal d'itérations
        tol (float): Tolérance de convergence (norme du gradient)
        verbose (bool): Afficher les étapes

    Returns:
        np.ndarray: Centre analytique trouvé
    """
    W = W.detach().numpy() if isinstance(W, torch.Tensor) else np.array(W)
    b = b.detach().numpy() if isinstance(b, torch.Tensor) else np.array(b)

    # Ajout des bornes explicites x_i ∈ [-1,1]
    d = W.shape[1]
    W_ext = np.vstack([W, np.eye(d), -np.eye(d)])
    b_ext = np.hstack([b, np.ones(d), np.ones(d)])

    # Point initial strictement à l'intérieur
    x = find_interior_point(W_ext, b_ext)

    for i in range(max_iter):
        r = 1.0 / (b_ext - W_ext @ x)  # (m,)
        grad = W_ext.T @ r            # (d,)
        H = W_ext.T @ np.diag(r**2) @ W_ext  # Hessienne

        # Newton 
        delta_x = np.linalg.solve(H, grad)
        decrement = grad @ delta_x

        if verbose and i % 5 == 0:
            print(f"Iter {i}, ||grad|| = {np.linalg.norm(grad):.2e}, decrement = {decrement:.2e}")

        # Test de convergence
        if np.linalg.norm(grad) < tol:
            print("Newton converged.")
            break

        
        t = 1.0
        while True:
            x_new = x - t * delta_x
            if np.all(b_ext - W_ext @ x_new > 0):
                break
            t *= 0.5
            if t < 1e-10:
                raise RuntimeError("Line search failed")

        x = x_new

    return x


def get_pref_data(W: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    print(f"Auto : {b.shape}, {W.shape}")

    # center
    center = find_analytic_center(W, b) # center est un np.ndarray
    # hessian
    print("Computing hessian...")
    x = center
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
    return center, eigvecs.detach().numpy()

def get_pref_data2(W: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    print(f"Auto : {b.shape}, {W.shape}")

    # center
    center = find_analytic_center_newton(W, b)  # numpy array
    x = torch.tensor(center, dtype=torch.float32)

    # Convert constraint matrices to torch
    W_t = torch.tensor(W, dtype=torch.float32)
    b_t = torch.tensor(b, dtype=torch.float32)

    # Calcul de la Hessienne analytique H = Aᵀ D A
    print("Computing hessian...")
    r = 1.0 / (b_t - W_t @ x)
    D = torch.diag(r ** 2)
    H = W_t.T @ D @ W_t

    # Forcer la symétrie
    H = 0.5 * (H + H.T)

    # Régulariser
    epsilon = 1e-6
    H += epsilon * torch.eye(H.shape[0])

    # Décomposition spectrale
    eigvals, eigvecs = torch.linalg.eigh(H)
    print("Hessian computed !")
    return center, eigvecs.detach().numpy()
