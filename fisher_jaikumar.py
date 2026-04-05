# ─────────────────────────────────────────────
# 4. FISHER-JAIKUMAR
# ─────────────────────────────────────────────

import numpy as np
from data import DEPOT
from utilities import euclidean

def select_seeds(customers, K):
    """K semences maximin : bien espacées autour du dépôt."""
    cids  = list(customers.keys())
    seeds = [max(cids, key=lambda c: euclidean(DEPOT, customers[c]))]
    for _ in range(K - 1):
        best_cid, best_d = None, -np.inf
        for cid in cids:
            if cid in seeds:
                continue
            d = min(euclidean(customers[cid], customers[s]) for s in seeds)
            if d > best_d:
                best_d, best_cid = d, cid
        seeds.append(best_cid)
    return seeds


def fisher_jaikumar(customers, K, capacity, n_restarts=5):
    """
    Clustering par coût d'insertion c_{ik} = d(0,i) + d(i,s_k) - d(0,s_k).
    GAP résolu par greedy avec règle du regret.
    Retourne les clusters non-vides uniquement.
    """
    cids = list(customers.keys())
    best_clusters, best_cost = None, np.inf

    for restart in range(n_restarts):
        seeds = (select_seeds(customers, K) if restart == 0
                 else list(np.random.choice(cids, K, replace=False)))

        C = np.array([
            [euclidean(DEPOT, customers[cid])
             + euclidean(customers[cid], customers[s])
             - euclidean(DEPOT, customers[s])
             for s in seeds]
            for cid in cids
        ])

        regrets = sorted(
            [(np.sort(C[i])[1] - np.sort(C[i])[0] if len(C[i]) >= 2 else 0.0, i, cid)
             for i, cid in enumerate(cids)],
            reverse=True
        )

        clusters = [[] for _ in range(K)]
        caps     = [capacity] * K
        assigned = {c: False for c in cids}

        for _, i, cid in regrets:
            for k in np.argsort(C[i]):
                if caps[k] > 0:
                    clusters[k].append(cid)
                    caps[k] -= 1
                    assigned[cid] = True
                    break

        for cid in cids:
            if not assigned[cid]:
                clusters[np.argsort(C[cids.index(cid)])[0]].append(cid)

        cost = sum(C[cids.index(cid)][k]
                   for k, cl in enumerate(clusters) for cid in cl)
        if cost < best_cost:
            best_cost     = cost
            best_clusters = [cl[:] for cl in clusters]

    return [cl for cl in best_clusters if cl]