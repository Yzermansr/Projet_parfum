from optim import get_pref_data, get_pref_data2, compute_principal_axes
from best_question import get_best_question
from auto import generate_W
from min_regret import min_max_regret

def get_new_question(username: str):
    # preferences matrix
    W, b, P = generate_W()

    # getting center of the polyhedron and ellipsis axis
    center, axis = get_pref_data(W.get_matrix(), b)
    center2, axis2 = get_pref_data2(W.get_matrix(), b)
    #_, axis_PCA, _ = compute_principal_axes(A, b)

    # best question
    best_question_vect = get_best_question(axis, center, W, b, P)
    best_question_vect2 = get_best_question(axis2, center2, W, b, P)
    print(best_question_vect)
    print(best_question_vect2)
    X_min, Y_max, w_opt, min_max_value = min_max_regret(P, W)
    
    print("Minimax Regret Solution:")
    print(f"Alternative with minimal maximum regret: {X_min}")
    print(f"Worst-case alternative: {Y_max}")
    print(f"Weight vector producing maximum regret: {w_opt}")
    print(f"Minimum maximum regret value: {min_max_value}")

if __name__ == "__main__":
    get_new_question('yzermans')
