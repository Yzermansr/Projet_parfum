from auto import generate_W, generate_P
from best_question import get_best_question
from mon_min_max import min_regret

import numpy as np

import scipy

def get_new_question():
    # preferences matrix
    W, b = generate_W()
    P = generate_P()
    #res_gd = get_best_question("gradient descent", W, b, P)
    #res_nd = get_best_question("newton", W, b, P)
    # res_mmr = get_best_question("min-max-regret", W, b, P)
    #_,_,result = min_regret(P,W)
    res_dc, nb_iter = get_best_question("d-criterion", W, b, P)

    print("Best Questions:")
    #print(f"Gradient Descent: {res_gd[0]}, {res_gd[1]}")
    #print(f"Newton Gradient Descent: {res_nd[0]}, {res_nd[1]}")
    print(f"D-criterion: {res_dc[0]}, {res_dc[1]}, number of iterations: {nb_iter} ({nb_iter/scipy.comb(len(P), 2) * 100}% of maximum)")
    # print(f"MinMax Regret: {res_mmr[0]}, {res_mmr[1]}")
    #print(f"Minimum maximum regret value: {np.linalg.norm(result)}")

    

if __name__ == "__main__":
    get_new_question()
