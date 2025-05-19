from auto import generate_W
from best_question import get_best_question
from mon_min_max import min_regret

def get_new_question():
    # preferences matrix
    W, b, P = generate_W()
    res_gd = get_best_question("gradient descent", W, b, P)
    res_nd = get_best_question("newton", W, b, P)
    res_mmr = get_best_question("min-max-regret", W, b, P)
    _,_,result = min_regret(P,W)
    
    print(f'Minimum maximum regret value: {result}')
    print("Best Questions:")
    print(f"Gradient Descent: {res_gd[0]}, {res_gd[1]}")
    print(f"Newton Gradient Descent: {res_nd[0]}, {res_nd[1]}")
    print(f"MinMax Regret: {res_mmr[0]}, {res_mmr[1]}")
    print(f"Minimum maximum regret value: {res_mmr[2]}")

    

if __name__ == "__main__":
    get_new_question()
