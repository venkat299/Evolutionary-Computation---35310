import random

def two_opt_swap(tour):
    a, b = sorted(random.sample(range(len(tour)), 2))
    return tour[:a] + tour[a:b+1][::-1] + tour[b+1:]