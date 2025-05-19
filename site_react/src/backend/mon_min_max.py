


def regret(P, W, j):
    first = True  
    for w in W:
        fw_j = np.dot(w, P[j, :])  
        Pbis = np.delete(P, j, axis=0)
        max_i = [np.dot(w, Pbis[i, :]) for i in range(Pbis.shape[0])]
        max_value = max_i[0]
        indice = 0
        for n, value in enumerate(max_i):
            if value > max_value:
                max_value = value
                indice = n
        indice_global = indice if indice < j else indice + 1
        fw_max = P[indice_global, :]  
        max_wj = fw_max - P[j, :]    
        if first:
            max_w = max_wj
            w_max = w
            alt = fw_max
            first = False
        elif np.linalg.norm(max_wj) > np.linalg.norm(max_w):
            max_w = max_wj
            w_max = w
            alt = fw_max
    return w_max, alt, max_w

def min_regret(P,W):
    for j in range(P.shape[0]):
        w,A,r_j = regret(P,W,j) 
        if (j==0):
            regret_min = r_j
            Parfum_A = A
            Parfum_B = P[j]
            poids = w
        if(np.linalg.norm(regret_min) > np.linalg.norm(r_j)):
            regret_min = r_j
            Parfum_A = A
            Parfum_B = P[j]
            poids = w
    return Parfum_A,Parfum_B,regret_min
