# ─────────────────────────────────────────────
# 3. BASELINE OPTIMALE GLOBALE (indépendante de Fisher)
# ─────────────────────────────────────────────

import itertools
import numpy as np
from utilities import route_distance, k_min

def _generate_partitions(items, n_groups, max_per_group):
    """Génère toutes les partitions valides de items en n_groups."""
    if n_groups == 1:
        if len(items) <= max_per_group:
            yield [items[:]]
        return
    for size in range(
        max(0, len(items) - max_per_group * (n_groups - 1)),
        min(len(items), max_per_group) + 1,
    ):
        for combo in itertools.combinations(items, size):
            remaining = [x for x in items if x not in combo]
            for rest in _generate_partitions(remaining, n_groups - 1, max_per_group):
                yield [list(combo)] + rest


def _tsp_exact(cluster, customers):
    """TSP exact par énumération pour un cluster."""
    if len(cluster) == 0:
        return [], 0.0
    if len(cluster) == 1:
        return cluster, route_distance(cluster, customers)
    best_d, best_r = np.inf, None
    for perm in itertools.permutations(cluster):
        d = route_distance(list(perm), customers)
        if d < best_d:
            best_d, best_r = d, list(perm)
    return best_r, best_d


def compute_true_optimal(customers, capacity, nv):
    """
    Calcule la vraie distance optimale CVRP en énumérant toutes les
    partitions valides (k ∈ [k_min, nv]) et en résolvant le TSP exact
    pour chaque groupe.

    Retourne (optimal_distance, optimal_routes, optimal_k).
    Cette valeur est FIXE pour une instance donnée — elle ne dépend
    pas de la méthode de clustering utilisée.

    Limité aux instances où capacity ≤ 6 (énumération raisonnable).
    """
    cids      = list(customers.keys())
    k_min_val = k_min(customers, capacity)
    best_dist = np.inf
    best_routes = None
    best_k      = None

    for num_v in range(k_min_val, nv + 1):
        for partition in _generate_partitions(cids, num_v, capacity):
            # Vérifier que tous les clients sont couverts
            if set(c for g in partition for c in g) != set(cids):
                continue
            total = 0.0
            routes = []
            for group in partition:
                r, d = _tsp_exact(group, customers)
                routes.append(r)
                total += d
            if total < best_dist:
                best_dist   = total
                best_routes = routes
                best_k      = num_v

    return best_dist, best_routes, best_k