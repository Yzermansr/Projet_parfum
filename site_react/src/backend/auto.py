import pandas as pd
import numpy as np
import sqlite3

DB = 'comparaison.db'
DB2 = 'notes.db'

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
        comparaison.append({
            int(row[1]),
            int(row[2]),
        })
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
    for row in data:
        comparaison.append({
            str(row[1]),
            int(row[2]),
        })
    for i in range(n-1):
        print(comparaison[i])

    
    print(comparaison)
    return  comparaison

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