# ─────────────────────────────────────────────
# 2. UTILITAIRES
# ─────────────────────────────────────────────

import numpy as np
from data import DEPOT

def euclidean(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def route_distance(route, customers):
    if not route:
        return 0.0
    total = euclidean(DEPOT, customers[route[0]])
    for i in range(len(route)-1):
        total += euclidean(customers[route[i]], customers[route[i+1]])
    total += euclidean(customers[route[-1]], DEPOT)
    return total


def k_min(customers, capacity):
    return int(np.ceil(len(customers) / capacity))