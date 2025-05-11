from optim import get_pref_data, get_pref_data_2
from best_question import get_best_question
from auto import generate_W

def get_new_question(username: str):
    # preferences matrix
    A, b, P = generate_W()

    # getting center of the polyhedron and ellipsis axis
    center, axis = get_pref_data(A, b)

    # best question
    best_question_vect = get_best_question(axis, center, A, b, P)
    print(best_question_vect)

if __name__ == "__main__":
    get_new_question('yzermans')
