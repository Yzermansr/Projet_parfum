import numpy as np
import sqlite3

DB = 'comparaison.db'
DB2 = 'notes.db'
DB1 = 'parfums_numerotes.db'

def get_data():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('SELECT * FROM comparaison')
    data = c.fetchall()
    conn.close()
    comparaison = []
    n = 0
    for row in data:
        n = n + 1
        comparaison.append((
            int(row[1]),
            int(row[2]),
        ))
    print(comparaison)
    return  comparaison,n


def get_data2(pseudo):
    conn = sqlite3.connect(DB2)
    c = conn.cursor()
    c.execute('SELECT * FROM notes where pseudo = (?) order by note desc', (pseudo,))
    data = c.fetchall()
    conn.close()
    comparaison = []
    n = len(data)
    conn = sqlite3.connect(DB1)
    c = conn.cursor()
    for row in data:
        c.execute('SELECT id FROM parfums_numerotes where Nom = (?)', (row[1],))
        data = c.fetchall()
        comparaison.append((
            int(data[0][0]),
            int(row[2]),
        ))
    conn.close()
    comparaison2 = []
    for i in range(len(comparaison)):
        nom = comparaison[i][0]
        note = comparaison[i][1]
        for j in range(len(comparaison) - 1,i,-1):
            if (comparaison[j][1] < note):
                comparaison2.append((
                int(nom),
                int(comparaison[j][0]),
        ))

    print(comparaison2)
    return  comparaison2

comparisons,n = get_data()
comparisons2 = get_data2('yzermans')

def generate_constraints(comparisons, n = 1995, epsilon=1e-2):
    A = []
    b = []
    for i, j in comparisons:
        a = np.zeros(n)
        a[i] = -1
        a[j] = 1
        A.append(a)
        b.append(-epsilon)
    return np.array(A), np.array(b)

A_full, b_full = generate_constraints(comparisons)
print(A_full)
print(b_full)
print("et maintenant avec les notes")
A_full2, b_full2 = generate_constraints(comparisons2)
print(A_full2)
print(b_full2)