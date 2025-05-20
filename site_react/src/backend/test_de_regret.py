import sqlite3
import numpy as np
from comparison import Perfume, ComparisonMatrix, create_comparison
from auto import DATABASE
from min_regret import min_max_regret
from auto import generate_P

def get_comparisons(n):
    """Charge les n premières comparaisons de la base"""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("""SELECT p1.Id, p1.URL, p1.Nom, p1.Tete, p1.Coeur, p1.Fond,
                        p2.Id, p2.URL, p2.Nom, p2.Tete, p2.Coeur, p2.Fond
                 FROM Comparaison c
                 JOIN parfums_numerotes p1 ON c.parfum1 = p1.id
                 JOIN parfums_numerotes p2 ON c.parfum2 = p2.id
                 LIMIT ?""", (n,))
    data = c.fetchall()
    conn.close()

    comparisons = []
    for id1, url1, n1, t1, c1, f1, id2, url2, n2, t2, c2, f2 in data:
        x = Perfume((id1, url1, n1, t1, c1, f1))
        y = Perfume((id2, url2, n2, t2, c2, f2))
        comparisons.append(create_comparison(x, y))
    return ComparisonMatrix(comparisons)

# Récupère une liste de parfums filtrés
P = generate_P()

# Évaluer l'évolution du regret minimax en fonction du nombre de comparaisons utilisées
regrets = []
steps = list(range(5, 55, 5))  # de 5 à 50 comparaisons
for n in steps:
    W = get_comparisons(n)
    _, _, _, regret = min_max_regret(P, W)
    regrets.append(regret)

import matplotlib.pyplot as plt
plt.plot(steps, regrets, marker='o')
plt.xlabel("Nombre de comparaisons")
plt.ylabel("Regret minimax")
plt.title("Évolution du regret minimax en fonction des comparaisons")
plt.grid(True)
plt.show()
