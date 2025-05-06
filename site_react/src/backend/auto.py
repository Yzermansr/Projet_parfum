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
    x = []
    for row in data:
        n = n + 1
        comparaison.append((
            int(row[1]),
            int(row[2]),
        ))
        if row[1] not in x:
            x.append(row[1])
        if row[2] not in x :
            x.append(row[2])
    print(x)
    comparaison2 = []
    for j in range (len(x)):
        for i in range(1,1995):
            if ( i != x[j] ):
                comparaison2.append((i,x[j]))
                comparaison.append((x[j],i))
   
    comparaison3 = []
    for j in range(len(x)):
        for i in range(len(x)):
            if ( x[i] != x[j] ):
                comparaison3.append((x[i],x[j]))   
                comparaison3.append((x[j],x[i]))  
 
    return  comparaison2,n,comparaison3


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
    x = []
    for row in data:
        c.execute('SELECT id FROM parfums_numerotes where Nom = (?)', (row[1],))
        data = c.fetchall()
        comparaison.append((
            int(data[0][0]),
            int(row[2]),
        ))
        if data[0][0] not in x:
            x.append(data[0][0])
        if row[2] not in x :
            x.append(row[2])
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
    print(x)
    return  comparaison2







def generate_constraints2(comparisons, n=1995, epsilon=1e-2, bounds=True):
    A = []
    b = []
    for i, j in comparisons:
        a = np.zeros(n)
        a[i] = -1
        a[j] = 1
        A.append(a)
        b.append(-epsilon)

    if bounds:
        M = 10  
        for i in range(n):
            upper = np.zeros(n)
            upper[i] = 1
            A.append(upper)
            b.append(M)
            
            lower = np.zeros(n)
            lower[i] = -1
            A.append(lower)
            b.append(M)

    return np.array(A), np.array(b)


def get_preference_matrix(username: str):
    comparaisons = [(0,1),(1,2),(3,1),(4,3),(4,1)]
    x = []
    for (i,j) in comparaisons:
        if i not in x :
            x.append(i)
        if j not in x :
            x.append(j)
    test = []
    for j in range(len(x)):
        for i in range(len(x)):
            if ( x[i] != x[j] ):
                test.append((x[i],x[j]))   
                test.append((x[j],x[i])) 
    A2,b2 = generate_constraints2(test,n=5,epsilon=1e-2)
    A, b = generate_constraints2(comparaisons, n = 5, epsilon=1e-2)
    print(A)
    print(b)    
    return A, b


def generate_constraints(comparisons, n=1995, epsilon=1e-2, bounds=True):
    A = []
    b = []
    for i, j in comparisons:
        a = np.zeros(n)
        a[i] = -1
        a[j] = 1
        A.append(a)
        b.append(-epsilon)
    return np.array(A),np.array(b)

def get_preference_matrix2(username: str):
    comparaisons = [(0,1),(1,2),(3,1),(4,3),(1,4)]
    x = []
    for (i,j) in comparaisons:
        if i not in x :
            x.append(i)
        if j not in x :
            x.append(j)
    test = []
    for j in range(len(x)):
        for i in range(len(x)):
            if ( x[i] != x[j] ):
                test.append((x[i],x[j]))   
                test.append((x[j],x[i])) 

    A2,b2 = generate_constraints(test,n=5,epsilon=1e-2)
    A, b = generate_constraints2(comparaisons, n = 5, epsilon=1e-2)
    print(A2)
    print(b2)    
    return A2,b2

