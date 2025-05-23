import sqlite3

import numpy as np


class Perfume:
    def __init__(self, data: tuple[int, str, str, str, str, str]):
        self.id = data[0]
        self.nom = data[2]
        self.url = data[1]
        self.tete = set(map(lambda x: int(x), data[3].split(',')))
        self.coeur = set(map(lambda x: int(x), data[4].split(',')))
        self.fond = set(map(lambda x: int(x), data[5].split(',')))

    def __str__(self):
        return f"Perfume {self.id} - {self.nom}"

    def __eq__(self, other) -> bool:
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def get_vector(self):
        p = np.zeros(328)

        # extract the numbers from the sets
        ingredients = list(self.tete) + list(self.coeur) + list(self.fond)

        # building P
        for i in ingredients:
            p[i] += 1

        return p

    def get_ingredients(self):
        return self.tete | self.coeur | self.fond


class Comparison:
    def __init__(self, p1: Perfume, p2: Perfume, vector: np.ndarray):
        self.p1 = p1
        self.p2 = p2
        self.vector = vector

    def __str__(self):
        return f"Perfume {self.p1.id} > Perfume {self.p2.id}"

    def get_vector(self):
        return self.vector


class ComparisonMatrix:
    def __init__(self, comparisons: list[Comparison]):
        self.comparisons = comparisons

    def __iter__(self):
        return iter(self.comparisons)

    def __getitem__(self, index):
        return self.comparisons[index]

    def get_matrix(self):
        m = np.array([c.get_vector() for c in self.comparisons])
        return m

    def insert_comparison(self, comparison: Comparison):
        new = self.comparisons + [comparison]
        return ComparisonMatrix(new)


def create_comparison(p1: tuple[int, str, str, str, str] | Perfume,
                      p2: tuple[int, str, str, str, str] | Perfume) -> Comparison:
    if isinstance(p1, tuple):
        p1 = Perfume(p1)
    if isinstance(p2, tuple):
        p2 = Perfume(p2)

    c = Comparison(p1, p2, p1.get_vector() - p2.get_vector())
    return c

def perfume_distance(p1: Perfume | int, p2: Perfume | int):
    if isinstance(p1, int):
        q = "SELECT Id, URL, Nom, Tete, Coeur, Fond FROM parfums_numerotes WHERE Id = ?;"
        conn = sqlite3.connect("database")
        c = conn.cursor()
        c.execute(q,(p1,))
        data = c.fetchone()
        conn.close()
        p1 = Perfume(data)
    if isinstance(p2, int):
        q = "SELECT Id, URL, Nom, Tete, Coeur, Fond FROM parfums_numerotes WHERE Id = ?;"
        conn = sqlite3.connect("database")
        c = conn.cursor()
        c.execute(q,(p2,))
        data = c.fetchone()
        conn.close()
        p2 = Perfume(data)

    com = p1.tete & p2.tete | p1.coeur & p2.coeur | p1.fond & p2.fond
    dif = p1.tete ^ p2.tete | p1.coeur ^ p2.coeur | p1.fond ^ p2.fond
    mid_len = len(p1.get_ingredients() | p2.get_ingredients())/2
    return (len(com) - len(dif))/mid_len
