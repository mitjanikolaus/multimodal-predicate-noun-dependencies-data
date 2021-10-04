import itertools


def get_tuples_no_duplicates(names):
    all_tuples = [(a1, a2) for a1, a2 in list(itertools.product(names, names)) if a1 != a2]
    tuples = []
    for (a1, a2) in all_tuples:
        if not (a2, a1) in tuples:
            tuples.append((a1, a2))
    return tuples

