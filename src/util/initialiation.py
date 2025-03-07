import random

def pseudo_random_init(flights, pairings):
    S = [0] * len(pairings)
    U = set(flights)
    while U:
        j = random.choice(list(U))
        I = [i for i, pairing in enumerate(pairings) if j in pairing]
        if I:
            k = random.choice(I)
            S[k] = 1
            U -= set(pairings[k])
    return S
