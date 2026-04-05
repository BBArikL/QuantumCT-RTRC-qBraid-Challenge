# ─────────────────────────────────────────────
# 7. DÉCODAGE
# ─────────────────────────────────────────────

import numpy as np

def decode_tsp(bitstring, cluster_ids, n):
    bits   = [int(b) for b in reversed(bitstring)]
    matrix = np.array(bits).reshape(n, n)
    route, used, valid = [None]*n, set(), True
    for p in range(n):
        assigned = np.where(matrix[:, p] == 1)[0]
        if len(assigned) != 1 or assigned[0] in used:
            valid = False; break
        used.add(assigned[0])
        route[p] = cluster_ids[assigned[0]]
    return route if (valid and len(used) == n) else list(cluster_ids)