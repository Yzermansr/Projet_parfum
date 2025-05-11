import sqlite3
import numpy as np


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

    for id, t, c, f in data:
        t_list = set(map(lambda x: int(x), t.split(',')))
        c_list = set(map(lambda x: int(x), c.split(',')))
        f_list = set(map(lambda x: int(x), f.split(',')))

        if len(t_list & c_list & f_list) != 0:
            print(id)


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
    """
    conn = sqlite3.connect("database")
    c = conn.cursor()
    c.execute("""SELECT p1.Id    AS parfum1_id,
                        p1.Tete  AS parfum1_tete,
                        p1.Coeur AS parfum1_coeur,
                        p1.Fond  AS parfum1_fond,
                        p2.Id    AS parfum2_id,
                        p2.Tete  AS parfum2_tete,
                        p2.Coeur AS parfum2_coeur,
                        p2.Fond  AS parfum2_fond
                 FROM Comparaison c
                          JOIN parfums_numerotes p1 ON c.parfum1 = p1.id
                          JOIN parfums_numerotes p2 ON c.parfum2 = p2.id;""")
    data = c.fetchall()
    conn.close()

    # building the constraint matrix W
    # dimensions: l×327
    W = []
    P = []
    for _, t1, c1, f1, _, t2, c2, f2 in data:
        x = generate_P(t1, c1, f1)
        y = generate_P(t2, c2, f2)
        if not any(np.array_equal(x, p) for p in P):
            P.append(x)
        if not any(np.array_equal(y, p) for p in P):
            P.append(y)
        W.append(x - y)
    # print(W[:5])

    # building b
    W = np.array(W)
    b = np.array([10e-2 for _ in range(W.shape[0])])

    return W, b, np.array(P)


def generate_P(t: str, c: str, f: str) -> np.ndarray:
    """
    Generates an array representing the presence of ingredients in a perfume. Each string contains a series of
    comma-separated integers that are parsed, aggregated, and tallied into a
    frequency count stored in a predefined array `p` of size 327.

    Parameters
    ----------
    t : str
        A string containing comma-separated integers, representing the ingredients
        of the "Tête" note.
    c : str
        A string containing comma-separated integers, representing the ingredients
        of the "Coeur" note.
    f : str
        A string containing comma-separated integers, representing the ingredients
        of the "Fond" note.

    Returns
    -------
    np.ndarray
        A NumPy array of size 327 with counts of ingredient appearances.
        Each index corresponds to a specific ingredient, and the
        value at that index represents its frequency.
    """
    p = np.zeros(327)

    # extract the numbers from the strings
    t_list = list(map(lambda x: int(x), t.split(',')))
    c_list = list(map(lambda x: int(x), c.split(',')))
    f_list = list(map(lambda x: int(x), f.split(',')))
    ingredients = t_list + c_list + f_list

    # building P
    for i in ingredients:
        p[i] += 1

    return p
