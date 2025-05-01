from auto import get_preference_matrix
from optim import get_pref_data
from best_question import get_best_question

def get_new_question(username: str):
    # preferences matrix
    A, b = get_preference_matrix(username)

    # getting center of the polyhedron and ellipsis axis
    center, axis = get_pref_data(A, b)

    # best question
    best_question_vect = get_best_question(axis, center)
    print(best_question_vect)

if __name__ == "__main__":
    get_new_question('yzermans')
