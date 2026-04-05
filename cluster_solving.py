# ─────────────────────────────────────────────
# 8. RÉSOLUTION PAR CLUSTER
# ─────────────────────────────────────────────

import numpy as np
from utilities import route_distance
from qubo_ising import build_tsp_qubo
from qaoaCircuit_xyMixer import run_qaoa
from decoding import decode_tsp
from global_optimal_baseline import _tsp_exact

def solve_clusters(clusters, customers, p_layers=1, eps=1e-6,
                   compute_cluster_ratios=False):
    """
    TSP QAOA sur chaque cluster.

    Si compute_cluster_ratios=True, calcule pour chaque cluster
    l'optimal TSP exact (brute force sur les clients du cluster uniquement)
    et retourne le ratio QAOA/optimal par cluster ainsi que le ratio moyen.
    Applicable dès que capacity ≤ ~6 clients par cluster.
    """
    routes        = []
    total_dist    = 0.0
    max_qubits    = 0
    max_gates     = 0
    best_history  = []
    cluster_ratios = []   # ratio QAOA/optimal par cluster (si demandé)

    for i, cluster in enumerate(clusters):
        if len(cluster) == 1:
            routes.append(cluster)
            d = route_distance(cluster, customers)
            total_dist += d
            print(f"    Véhicule {i+1} : trivial {cluster}")
            if compute_cluster_ratios:
                cluster_ratios.append({
                    "cluster"   : cluster,
                    "qaoa_dist" : d,
                    "opt_dist"  : d,
                    "ratio"     : 1.0,
                })
            continue

        n_cl = len(cluster)
        print(f"    Véhicule {i+1} : {cluster} → {n_cl**2} qubits")

        Q, n = build_tsp_qubo(cluster, customers)
        bs, _, n_q, n_g, hist = run_qaoa(Q, n_cl, p_layers=p_layers, eps=eps)

        if n_q > max_qubits or (n_q == max_qubits and n_g > max_gates):
            max_qubits, max_gates, best_history = n_q, n_g, hist

        route     = decode_tsp(bs, cluster, n)
        qaoa_dist = route_distance(route, customers)
        total_dist += qaoa_dist
        routes.append(route)

        if compute_cluster_ratios:
            _, opt_dist = _tsp_exact(cluster, customers)
            ratio = qaoa_dist / opt_dist if opt_dist > 0 else 1.0
            cluster_ratios.append({
                "cluster"   : cluster,
                "qaoa_dist" : qaoa_dist,
                "opt_dist"  : opt_dist,
                "ratio"     : ratio,
            })
            print(f"      QAOA={qaoa_dist:.3f} | Opt TSP={opt_dist:.3f} "
                  f"| Ratio={ratio:.4f}"
                  + ("  ✓" if ratio <= 1.05 else "  ⚠"))

    mean_ratio = (np.mean([r["ratio"] for r in cluster_ratios])
                  if cluster_ratios else None)

    return routes, total_dist, max_qubits, max_gates, best_history, \
           cluster_ratios, mean_ratio
