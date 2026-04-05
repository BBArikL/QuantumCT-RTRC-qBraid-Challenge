# ─────────────────────────────────────────────
# 9. PIPELINE PAR INSTANCE
# ─────────────────────────────────────────────

import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
from data import INSTANCES
from utilities import k_min
from cluster_solving import solve_clusters
from global_optimal_baseline import compute_true_optimal
from fisher_jaikumar import fisher_jaikumar
from route_map import plot_routes_map
from convergence_graph import plot_convergence

def solve_instance(instance_id, p_values=(1, 2, 3), eps=1e-6):
    """
    Pour chaque p ∈ p_values :
      - Cherche le meilleur k ∈ [k_min, nv]
      - Résout avec Fisher-Jaikumar + QAOA
      - Instances 1-4 : ratio global vs optimal brute force
      - Instances 5 et 7 : ratio moyen par cluster (TSP exact sur chaque cluster)
    """
    inst      = INSTANCES[instance_id]
    customers = inst["customers"]
    capacity  = inst["capacity"]
    nv        = inst["nv"]
    k_min_val = k_min(customers, capacity)
    can_brute_force      = (capacity <= 6 and len(customers) <= 12)
    # Pour instances 5 et 7 : les clusters ont au plus capacity clients,
    # donc le TSP exact par cluster est toujours faisable (≤ 4! = 24 permutations)
    compute_cluster_ratios = instance_id in (7,)

    print(f"\n{'='*60}")
    print(f"  INSTANCE {instance_id}  "
          f"(nv={nv}, C={capacity}, clients={len(customers)}, k_min={k_min_val})")
    print(f"{'='*60}")

    # ── Calcul de l'optimal GLOBAL (une seule fois) ────────────────
    if can_brute_force:
        print("  Calcul de l'optimal global (brute force)...")
        t_opt = time.time()
        true_optimal, optimal_routes, optimal_k = compute_true_optimal(
            customers, capacity, nv
        )
        print(f"  Optimal global : {true_optimal:.4f} "
              f"(k={optimal_k}, {time.time()-t_opt:.1f}s)")
    else:
        true_optimal   = None
        optimal_routes = None
        optimal_k      = None
        print("  Optimal global : N/A (capacity > 6, énumération trop coûteuse)")

    convergence_data = {}
    best_overall     = {"distance": np.inf, "p": None, "routes": None}

    for p in p_values:
        print(f"\n{'─'*40}  p={p}")
        t0     = time.time()
        best_p = {"distance": np.inf}

        for k in range(k_min_val, nv + 1):
            print(f"  k={k}")
            clusters = fisher_jaikumar(customers, k, capacity)
            routes, dist, n_q, n_g, hist, cluster_ratios, mean_ratio = solve_clusters(
                clusters, customers, p_layers=p, eps=eps,
                compute_cluster_ratios=compute_cluster_ratios
            )

            # Ratio par rapport à l'optimal GLOBAL (fixe pour l'instance)
            if true_optimal is not None:
                ratio = dist / true_optimal
                print(f"    Distance QAOA  : {dist:.4f}")
                print(f"    Optimal global : {true_optimal:.4f}  (fixe)")
                print(f"    Ratio          : {ratio:.4f}  "
                      + ("✓" if ratio <= 1.05 else "⚠  >5% de l'optimal"))
            else:
                ratio = None
                print(f"    Distance QAOA  : {dist:.4f}")
                if mean_ratio is not None:
                    print(f"    Ratio moyen / cluster : {mean_ratio:.4f}  "
                          + ("✓" if mean_ratio <= 1.05 else "⚠  >5% de l'optimal TSP local"))

            if dist < best_p["distance"]:
                best_p = {
                    "distance"      : dist,
                    "ratio"         : ratio,
                    "mean_ratio"    : mean_ratio,
                    "cluster_ratios": cluster_ratios,
                    "k"             : k,
                    "routes"        : routes,
                    "n_qubits"      : n_q,
                    "n_gates"       : n_g,
                    "history"       : hist,
                }

        convergence_data[p] = {
            **best_p,
            "optimal"   : true_optimal,   # identique pour tous les p
            "exec_time" : time.time() - t0,
        }

        ratio_str      = f"{best_p['ratio']:.4f}"      if best_p.get('ratio')      is not None else "N/A"
        mean_ratio_str = f"{best_p['mean_ratio']:.4f}" if best_p.get('mean_ratio') is not None else "N/A"
        print(f"  → p={p} : k={best_p['k']}, dist={best_p['distance']:.4f}, "
              f"ratio={ratio_str}, ratio_moy_cluster={mean_ratio_str}, "
              f"qubits={best_p['n_qubits']}, portes={best_p['n_gates']}, "
              f"temps={convergence_data[p]['exec_time']:.1f}s")

        if best_p["distance"] < best_overall["distance"]:
            best_overall = {
                "distance" : best_p["distance"],
                "p"        : p,
                "routes"   : best_p["routes"],
            }

    # ── Tableau résumé ─────────────────────────────────────────────
    opt_str = f"{true_optimal:.4f}" if true_optimal is not None else "N/A"
    print(f"\n{'─'*70}")
    print(f"  Résumé instance {instance_id}  |  Optimal global : {opt_str}")

    if compute_cluster_ratios:
        print(f"  {'p':<5}{'k':<5}{'Dist QAOA':<13}{'Ratio global':<14}"
              f"{'Ratio moy/cluster':<20}{'Qubits':<9}{'Portes':<9}{'Temps(s)'}")
        print(f"  {'─'*70}")
        for p in p_values:
            d      = convergence_data[p]
            rat    = f"{d['ratio']:.4f}"      if d.get('ratio')      is not None else "N/A"
            mrat   = f"{d['mean_ratio']:.4f}" if d.get('mean_ratio') is not None else "N/A"
            tag    = " ← meilleur" if p == best_overall["p"] else ""
            print(f"  {p:<5}{d['k']:<5}{d['distance']:<13.4f}{rat:<14}{mrat:<20}"
                  f"{d['n_qubits']:<9}{d['n_gates']:<9}{d['exec_time']:.1f}s{tag}")

        # Détail par cluster pour le meilleur p
        best_p_val = best_overall["p"]
        best_cr    = convergence_data[best_p_val].get("cluster_ratios", [])
        if best_cr:
            print(f"\n  Détail par cluster (meilleur p={best_p_val}) :")
            print(f"  {'Cluster':<30}{'QAOA dist':<13}{'Opt TSP':<13}{'Ratio'}")
            print(f"  {'─'*60}")
            for cr in best_cr:
                flag = "  ✓" if cr["ratio"] <= 1.05 else "  ⚠"
                print(f"  {str(cr['cluster']):<30}{cr['qaoa_dist']:<13.4f}"
                      f"{cr['opt_dist']:<13.4f}{cr['ratio']:.4f}{flag}")
    else:
        print(f"  {'p':<5}{'k':<5}{'Dist QAOA':<13}{'Ratio':<10}"
              f"{'Qubits':<9}{'Portes':<9}{'Temps(s)'}")
        print(f"  {'─'*58}")
        for p in p_values:
            d   = convergence_data[p]
            rat = f"{d['ratio']:.4f}" if d.get('ratio') is not None else "N/A"
            tag = " ← meilleur" if p == best_overall["p"] else ""
            print(f"  {p:<5}{d['k']:<5}{d['distance']:<13.4f}{rat:<10}"
                  f"{d['n_qubits']:<9}{d['n_gates']:<9}{d['exec_time']:.1f}s{tag}")

    # ── Sauvegarde solution ────────────────────────────────────────
    lines = []
    for i, route in enumerate(best_overall["routes"]):
        if route:
            lines.append(f"r{i+1}: 0, {', '.join(str(c) for c in route)}, 0")
        else:
            lines.append(f"r{i+1}: 0, 0")

    with open(f"Instance{instance_id}.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  ✅ Solution (p={best_overall['p']}) → Instance{instance_id}.txt")

    # ── Graphiques ─────────────────────────────────────────────────
    plot_convergence(instance_id, convergence_data, p_values, true_optimal)
    plot_routes_map(instance_id, best_overall["routes"], customers, best_overall["p"])

    return convergence_data