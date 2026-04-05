# ─────────────────────────────────────────────
# 5. QUBO + ISING
# ─────────────────────────────────────────────

import numpy as np
from data import DEPOT
from utilities import euclidean

def build_tsp_qubo(cluster_ids, customers, penalty=10.0):
    n     = len(cluster_ids)
    all_c = [DEPOT] + [customers[cid] for cid in cluster_ids]
    N     = len(all_c)
    D     = np.array([[euclidean(all_c[i], all_c[j]) for j in range(N)]
                      for i in range(N)])

    Q   = np.zeros((n*n, n*n))
    idx = lambda i, p: (i-1)*n + p

    for i in range(1, N):
        Q[idx(i,0)][idx(i,0)]     += D[0][i]
        Q[idx(i,n-1)][idx(i,n-1)] += D[i][0]

    for p in range(n-1):
        for i in range(1, N):
            for j in range(1, N):
                if i != j:
                    Q[idx(i,p)][idx(j,p+1)] += D[i][j]

    for p in range(n):
        for i in range(1, N):
            Q[idx(i,p)][idx(i,p)] += penalty * (1-2)
            for j in range(i+1, N):
                Q[idx(i,p)][idx(j,p)] += penalty * 2

    for i in range(1, N):
        for p in range(n):
            Q[idx(i,p)][idx(i,p)] += penalty * (1-2)
            for q in range(p+1, n):
                Q[idx(i,p)][idx(i,q)] += penalty * 2

    return (Q + Q.T) / 2, n


def qubo_to_ising(Q):
    n = Q.shape[0]
    h, J, offset = np.zeros(n), np.zeros((n,n)), 0.0
    for i in range(n):
        h[i]   += Q[i][i] / 2
        offset += Q[i][i] / 4
    for i in range(n):
        for j in range(i+1, n):
            J[i][j] += Q[i][j] / 4
            h[i]    += Q[i][j] / 4
            h[j]    += Q[i][j] / 4
            offset  += Q[i][j] / 4
    return h, J, offset