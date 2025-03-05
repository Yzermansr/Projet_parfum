import torch
import torch.optim as optim
from torch.autograd.functional import hessian

def barrier_function(x):
    # on vérifie si le point est toujours dans le polyèdre
    # margin = b - A @ x
    # assert torch.all(margin >= 0), f"margin must be positive : {margin}"

    return -torch.sum(torch.log(b - A@x))

def find_min(A, b, barrier, nb_iter=1000):
    # Initialisation de x sur la bordure du polyèdre
    x = torch.tensor([0.99, 0.99], requires_grad=True)

    # on vérifie si le point est toujours dans le polyèdre
    margin = b - A @ x
    assert torch.all(margin >= 0), f"margin must be positive : {margin}"

    # Optimiseur (descente de gradient)
    optimizer = optim.SGD([x], lr=0.01)

    for i in range(nb_iter):
        optimizer.zero_grad()
        loss = barrier(x)  # Minimiser la fonction barrière
        loss.backward()  # Calcul des gradients
        optimizer.step()  # Mise à jour de x

        # Affichage périodique
        if i % 100 == 0:
            print(f"Étape {i} : x = {x.tolist()}, perte = {loss.item()}")
    return x

import torch

# Définition du polyèdre : Ax <= b
A = torch.tensor([
    [1.0, 1.0],   # x + y <= 2
    [-1.0, 1.0],  # -x + y <= 1
    [0.0, -1.0]   # y >= 0 (équivalent à -y <= 0)
])
b = torch.tensor([2.0, 1.0, 0.0])

min = find_min(A, b, barrier_function)

print(f"minimum : {min}")

# matrice hessienne
h = hessian(barrier_function, min)
print(h)