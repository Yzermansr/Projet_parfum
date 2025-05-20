import sqlite3
from collections import Counter

import numpy as np

from comparison import Perfume, create_comparison, ComparisonMatrix

DATABASE = "database"

def generate_P(database: str = DATABASE, k: int = 1, n: int = 2) -> list[Perfume]:
    """
    Builds the matrix of perfumes `P`, with all perfumes containing at least k of the
    n most frequent ingredients.

    Args:
        database (str): the database to use.
        k (int): minimal number of frequent ingredients.
        n (int): number of most frequent ingredients to consider.

    Returns:
        list[Perfume]: list of perfumes containing the frequent ingredients.
    """
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute("SELECT id, URL,Nom , Tete, Coeur, Fond FROM parfums_numerotes")
    rows = cursor.fetchall()
    conn.close()

    perfumes = [Perfume(row) for row in rows]

    # Compter la fréquence des ingrédients
    all_ingredients = []
    for p in perfumes:
        all_ingredients.extend(p.get_ingredients())

    top_ingredients = set(
        ing for ing, _ in Counter(all_ingredients).most_common(n)
    )
    print(top_ingredients)

    # Garder les parfums avec au moins k ingrédients communs avec le top
    filtered_perfumes = [
        p for p in perfumes
        if len(p.get_ingredients() & top_ingredients) >= k
    ]

    return filtered_perfumes

def generate_W():
    """
    Generates the constraint matrix `W`, constraint vector `b`, and perfume matrix `P` used in
    the recommendation algorithm.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing:
        - W: The constraint matrix with shape (l, 327) where `l` is the number of comparisons.
        - b: A vector of constraint values, initialized to 0.01 for each comparison.
        - P: A matrix of vectorized perfumes with shape (m, 327) where `m` is the number of perfumes.
        P is the matrix of p-vectors for every perfume in the comparison database (also see `generate_P`).
    """
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("""SELECT p1.Id    AS parfum1_id,
                        p1.URL   AS parfum1_url,
                        p1.Nom   AS parfum1_nom,
                        p1.Tete  AS parfum1_tete,
                        p1.Coeur AS parfum1_coeur,
                        p1.Fond  AS parfum1_fond,
                        p2.Id    AS parfum2_id,
                        p2.URL   AS parfum2_url,
                        p2.Nom   AS parfum2_nom,
                        p2.Tete  AS parfum2_tete,
                        p2.Coeur AS parfum2_coeur,
                        p2.Fond  AS parfum2_fond
                 FROM Comparaison c
                          JOIN parfums_numerotes p1 ON c.parfum1 = p1.id
                          JOIN parfums_numerotes p2 ON c.parfum2 = p2.id;""")
    data = c.fetchall()
    conn.close()

    #data = convert_data(data, constrains=True)

    # building the constraint matrix W
    # dimensions: l×327
    W = []
    for id1, url1, n1, t1, c1, f1, id2, url2, n2, t2, c2, f2 in data:
        x = Perfume((id1, url1, n1, t1, c1, f1))
        y = Perfume((id2, url2, n2, t2, c2, f2))
        W.append(create_comparison(x, y))
    # print(W[:5])

    # building b
    W = ComparisonMatrix(W)
    b = np.array([1e-7 for _ in range(W.get_matrix().shape[0])])

    return W, b