import sqlite3

import numpy as np

from comparison import Perfume, Comparison, create_comparison, ComparisonMatrix


def check_ingredients_unicity() -> None:
    """
    Prints the perfumes where an ingredient is present in more than one note.

    Returns:
        None
    """
    conn = sqlite3.connect("database")
    c = conn.cursor()
    c.execute("""SELECT Id,
                        Tete,
                        Coeur,
                        Fond
                 FROM parfums_numerotes;""")
    data = c.fetchall()
    conn.close()
    data = convert_data(data)
    for id, t, c, f in data:
        if len(t & c & f) != 0:
            print(id)

def convert_data(data: list, constrains: bool = False):
    res = []
    if constrains:
        for id, t, c, f, id1, t1, c1, f1 in data:
            t_list = set(map(lambda x: int(x), t.split(',')))
            c_list = set(map(lambda x: int(x), c.split(',')))
            f_list = set(map(lambda x: int(x), f.split(',')))
            t1_list = set(map(lambda x: int(x), t1.split(',')))
            c1_list = set(map(lambda x: int(x), c1.split(',')))
            f1_list = set(map(lambda x: int(x), f1.split(',')))
            res.append((id, t_list, c_list, f_list, id1, t1_list, c1_list, f1_list))

    else:
        for id, t, c, f in data:
            t_list = set(map(lambda x: int(x), t.split(',')))
            c_list = set(map(lambda x: int(x), c.split(',')))
            f_list = set(map(lambda x: int(x), f.split(',')))
            res.append((id, t_list, c_list, f_list))

    return res

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
    conn = sqlite3.connect("database")
    c = conn.cursor()
    c.execute("""SELECT p1.Id    AS parfum1_id,
                        p1.Nom   AS parfum1_nom,
                        p1.Tete  AS parfum1_tete,
                        p1.Coeur AS parfum1_coeur,
                        p1.Fond  AS parfum1_fond,
                        p2.Id    AS parfum2_id,
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
    P = []
    for id1, n1, t1, c1, f1, id2, n2, t2, c2, f2 in data:
        x = Perfume((id1, n1, t1, c1, f1))
        y = Perfume((id2, n2, t2, c2, f2))
        if not x in P:
            P.append(x)
        if not y in P:
            P.append(y)
        W.append(create_comparison(x, y))
    # print(W[:5])

    # building b
    W = ComparisonMatrix(W)
    b = np.array([1e-7 for _ in range(W.get_matrix().shape[0])])

    return W, b, P


def generate_P(t: set, c: set, f: set) -> np.ndarray:
    """
    Generates an array representing the presence of ingredients in a perfume. Each index represents
    an ingredient, present if the value is 1.

    Parameters
    ----------
    t : set
        A set containing the top note ingredients of a perfume.
    c : set
        A set containing the heart note ingredients of a perfume.
    f : set
        A set containing the base note ingredients of a perfume.

    Returns
    ----------
    np.ndarray
        A NumPy array of size 328 with counts of ingredient appearances.
    """
    p = np.zeros(328)

    # extract the numbers from the sets
    ingredients = list(t) + list(c) + list(f)

    # building P
    for i in ingredients:
        p[i] += 1

    return p

def get_perfumes_from_constraint(w: np.ndarray) -> tuple[int, int]:
    """
    Finds the perfumes compared to create the constraint vector w.
    Parameters
    ----------
    w (np.ndarray): The constraint vector.

    Returns
    -------
    [int, int]: the IDs of the perfumes compared.
    """
    conn = sqlite3.connect("database")
    c = conn.cursor()
    c.execute("""SELECT Id,
                        Tete,
                        Coeur,
                        Fond
                 FROM parfums_numerotes;""")
    data = c.fetchall()
    conn.close()

    data = convert_data(data)

    for i, (id, t, c, f) in enumerate(data):
        for j, (id2, t2, c2, f2) in enumerate(data, i):
            x = generate_P(t, c, f)
            y = generate_P(t2, c2, f2)
            if np.array_equal(x - y, w) or np.array_equal(x - y, -w):
                return id, id2
    return -1, -1
