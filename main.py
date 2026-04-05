"""
CVRP Solver — Fisher-Jaikumar + QAOA XY-Mixeur
===============================================
Pipeline :
  1. Fisher-Jaikumar  → clusters par coût d'insertion minimal
  2. QAOA + XY-mixeur → TSP sur chaque cluster
  3. Reconstruction   → routes complètes dépôt → clients → dépôt

Analyse :
  - Sweep sur p ∈ {1, 2, 3} couches QAOA par instance
  - Recherche du meilleur nombre de véhicules k ∈ [k_min, nv]
  - Warm start sur le premier restart QAOA
  - Graphique de convergence par instance (PNG)
  - Ratio d'approximation GLOBAL (distance totale QAOA / distance totale optimale)

NOTE sur le ratio d'approximation :
  L'optimal est calculé UNE SEULE FOIS par instance (indépendamment de Fisher)
  en énumérant toutes les partitions valides et en résolvant le TSP exact
  pour chacune. Ce ratio est donc une vraie mesure d'optimalité globale,
  identique pour tous les p et tous les k testés.
"""

import numpy as np
from pipeline_instances import solve_instance

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. DONNÉES
# ─────────────────────────────────────────────

'''INSTANCES = {
    1: {"nv": 2, "capacity": 5,
        "customers": {1: (-2,2), 2: (-5,8), 3: (2,3)}},
    2: {"nv": 2, "capacity": 2,
        "customers": {1: (-2,2), 2: (-5,8), 3: (2,3)}},
    3: {"nv": 3, "capacity": 2,
        "customers": {1:(-2,2), 2:(-5,8), 3:(2,3),
                      4:(5,7),  5:(2,4),  6:(2,-3)}},
    4: {"nv": 4, "capacity": 3,
        "customers": {1:(-2,2),  2:(-5,8),  3:(6,3),
                      4:(4,4),   5:(3,2),   6:(0,2),
                      7:(-2,3),  8:(-4,3),  9:(2,3),
                      10:(2,7), 11:(-2,5), 12:(-1,4)}},
    7: {"nv": 5, "capacity": 4,
        "customers": {
            1:  (-11,  3),  2:  ( -7,  9),  3:  ( -3,  6),
            4:  ( -9, -2),  5:  (  4, 11),  6:  ( 10,  5),
            7:  (  7, -1),  8:  (  2,  7),  9:  (  9, -6),
            10: (  5, -11), 16: ( -5, -8),  17: (-10, -5),
            18: ( -8,  6),  19: (  1, -5),  20: ( -4,  1),
            11: (  0,  0),  12: ( -1, 10),  13: (  6, -7),
            14: ( -6,  4),  15: (  3, -2),
        }},
}

DEPOT = (0, 0)'''


# ─────────────────────────────────────────────
# 2. UTILITAIRES
# ─────────────────────────────────────────────

'''def euclidean(p1, p2):
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
    return int(np.ceil(len(customers) / capacity))'''


# ─────────────────────────────────────────────
# 3. BASELINE OPTIMALE GLOBALE (indépendante de Fisher)
# ─────────────────────────────────────────────

'''def _generate_partitions(items, n_groups, max_per_group):
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

    return best_dist, best_routes, best_k'''


# ─────────────────────────────────────────────
# 4. FISHER-JAIKUMAR
# ─────────────────────────────────────────────

'''def select_seeds(customers, K):
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

    return [cl for cl in best_clusters if cl]'''


# ─────────────────────────────────────────────
# 5. QUBO + ISING
# ─────────────────────────────────────────────

'''def build_tsp_qubo(cluster_ids, customers, penalty=10.0):
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
    return h, J, offset'''


# ─────────────────────────────────────────────
# 6. CIRCUIT QAOA + XY-MIXEUR
# ─────────────────────────────────────────────

'''def build_qaoa_xy_circuit(n_clients, gamma, beta, h, J, eps=1e-6):
    n_q = n_clients * n_clients
    qc  = QuantumCircuit(n_q)
    qc.h(range(n_q))

    def q(i, p): return i * n_clients + p

    for layer in range(len(gamma)):
        for i in range(n_q):
            for j in range(i+1, n_q):
                if abs(J[i][j]) > eps:
                    qc.cx(i, j)
                    qc.rz(2 * gamma[layer] * J[i][j], j)
                    qc.cx(i, j)
        for i in range(n_q):
            if abs(h[i]) > eps:
                qc.rz(2 * gamma[layer] * h[i], i)

        for p in range(n_clients):
            for i in range(n_clients):
                for j in range(i+1, n_clients):
                    qc.rxx(2 * beta[layer], q(i,p), q(j,p))
                    qc.ryy(2 * beta[layer], q(i,p), q(j,p))

    qc.measure_all()
    return qc


def energy(params, n_clients, h, J, offset, p_layers,
           shots=1024, eps=1e-6, history=None):
    gamma, beta = params[:p_layers], params[p_layers:]
    n_q = n_clients * n_clients
    qc  = build_qaoa_xy_circuit(n_clients, gamma, beta, h, J, eps)
    counts = (StatevectorSampler()
              .run([qc], shots=shots)
              .result()[0].data.meas.get_counts())

    E, total = 0.0, sum(counts.values())
    for bs, cnt in counts.items():
        spins = [2*int(b)-1 for b in reversed(bs)]
        e = (offset
             + sum(h[i]*spins[i] for i in range(n_q))
             + sum(J[i][j]*spins[i]*spins[j]
                   for i in range(n_q) for j in range(i+1, n_q)))
        E += cnt * e
    E /= total

    if history is not None:
        history.append(E)
    return E


def warm_start(Q, n_clients, p_layers):
    x = np.zeros(Q.shape[0])
    for i in range(n_clients):
        x[i * n_clients + i] = 1.0
    e = float(x @ Q @ x)
    b = np.clip(np.pi / (4 * abs(e)), 1e-3, np.pi/4) if abs(e) > 1e-10 else np.pi/8
    g0 = np.full(p_layers, np.pi/4) + np.random.uniform(-0.05, 0.05, p_layers)
    b0 = np.full(p_layers, b)       + np.random.uniform(-0.05, 0.05, p_layers)
    return np.concatenate([g0, b0])


def run_qaoa(Q, n_clients, p_layers=1, n_restarts=3, eps=1e-6):
    n_q          = Q.shape[0]
    h, J, offset = qubo_to_ising(Q)
    best_e, best_res, best_hist = np.inf, None, []

    for i in range(n_restarts):
        history = []
        p0 = (warm_start(Q, n_clients, p_layers) if i == 0
              else np.concatenate([np.random.uniform(0, 2*np.pi, p_layers),
                                   np.random.uniform(0, np.pi,   p_layers)]))

        res = minimize(energy, p0,
                       args=(n_clients, h, J, offset, p_layers, 1024, eps, history),
                       method='COBYLA',
                       options={'maxiter': 300, 'rhobeg': 0.5})
        if res.fun < best_e:
            best_e, best_res, best_hist = res.fun, res, history

    g_opt, b_opt = best_res.x[:p_layers], best_res.x[p_layers:]
    qc = build_qaoa_xy_circuit(n_clients, g_opt, b_opt, h, J, eps)
    counts = (StatevectorSampler()
              .run([qc], shots=2048)
              .result()[0].data.meas.get_counts())

    best_bs, best_qubo = None, np.inf
    for bs in counts:
        x   = np.array([int(b) for b in reversed(bs)])
        val = x @ Q @ x
        if val < best_qubo:
            best_qubo, best_bs = val, bs

    qc_tmp = build_qaoa_xy_circuit(n_clients, g_opt, b_opt, h, J, eps)
    qc_tmp.remove_final_measurements()
    n_gates = sum(qc_tmp.count_ops().values())

    return best_bs, best_qubo, n_q, n_gates, best_hist'''


# ─────────────────────────────────────────────
# 7. DÉCODAGE
# ─────────────────────────────────────────────

'''def decode_tsp(bitstring, cluster_ids, n):
    bits   = [int(b) for b in reversed(bitstring)]
    matrix = np.array(bits).reshape(n, n)
    route, used, valid = [None]*n, set(), True
    for p in range(n):
        assigned = np.where(matrix[:, p] == 1)[0]
        if len(assigned) != 1 or assigned[0] in used:
            valid = False; break
        used.add(assigned[0])
        route[p] = cluster_ids[assigned[0]]
    return route if (valid and len(used) == n) else list(cluster_ids)'''


# ─────────────────────────────────────────────
# 8. RÉSOLUTION PAR CLUSTER
# ─────────────────────────────────────────────

'''def solve_clusters(clusters, customers, p_layers=1, eps=1e-6,
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
           cluster_ratios, mean_ratio'''


# ─────────────────────────────────────────────
# 9. PIPELINE PAR INSTANCE
# ─────────────────────────────────────────────

'''def solve_instance(instance_id, p_values=(1, 2, 3), eps=1e-6):
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

    return convergence_data'''


# ─────────────────────────────────────────────
# 10. CARTE DES ROUTES
# ─────────────────────────────────────────────

'''VEHICLE_COLORS = [
    "#534AB7", "#1D9E75", "#D85A30", "#BA7517",
    "#E24B4A", "#185FA5", "#639922", "#D4537E",
]


def plot_routes_map(instance_id, routes, customers, p_best):
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_aspect("equal")
    ax.set_title(f"Instance {instance_id} — Routes QAOA (p={p_best})",
                 fontsize=13, pad=12)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    all_x  = [xy[0] for xy in customers.values()] + [0]
    all_y  = [xy[1] for xy in customers.values()] + [0]
    margin = 2.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.grid(True, color="#e0e0e0", linewidth=0.5, zorder=0)
    ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=1)
    ax.axvline(0, color="#cccccc", linewidth=0.8, zorder=1)

    import matplotlib.patches as mpatches
    legend_handles = []

    for v_idx, route in enumerate(routes):
        if not route:
            continue
        color  = VEHICLE_COLORS[v_idx % len(VEHICLE_COLORS)]
        dist   = route_distance(route, customers)
        full   = [None] + route + [None]

        def get_xy(cid):
            return customers[cid] if cid is not None else DEPOT

        for seg in range(len(full) - 1):
            x1, y1 = get_xy(full[seg])
            x2, y2 = get_xy(full[seg + 1])
            rad = 0.08 * (v_idx % 3 - 1)
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6,
                                        connectionstyle=f"arc3,rad={rad}",
                                        mutation_scale=14),
                        zorder=3)

        legend_handles.append(mpatches.Patch(
            color=color,
            label=f"V{v_idx+1} : {route}  (dist={dist:.2f})"
        ))

    client_color = {}
    for v_idx, route in enumerate(routes):
        for cid in route:
            client_color[cid] = VEHICLE_COLORS[v_idx % len(VEHICLE_COLORS)]

    for cid, (cx, cy) in customers.items():
        col = client_color.get(cid, "#888888")
        ax.plot(cx, cy, "o", color=col, markersize=18,
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        ax.text(cx, cy, str(cid), ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=6)

    ax.plot(0, 0, "*", color="black", markersize=20,
            markeredgecolor="white", markeredgewidth=1, zorder=7)
    ax.text(0.3, 0.3, "Dépôt", fontsize=8, color="#333333", zorder=7)

    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(0.0, -0.02), ncol=2, fontsize=8,
              frameon=True, framealpha=0.9, edgecolor="#cccccc")

    plt.tight_layout()
    path = f"map_instance{instance_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Carte des routes : {path}")'''


# ─────────────────────────────────────────────
# 11. GRAPHIQUE DE CONVERGENCE
# ─────────────────────────────────────────────

'''def plot_convergence(instance_id, convergence_data, p_values, true_optimal=None):
    COLORS = {1: "#534AB7", 2: "#1D9E75", 3: "#D85A30"}

    has_ratio = true_optimal is not None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax1, ax2  = axes

    fig.suptitle(f"Instance {instance_id} — QAOA Convergence per Depth p",
                 fontsize=13, y=1.01)

    # Panneau gauche : convergence énergie Ising
    for p in p_values:
        hist = convergence_data[p].get("history", [])
        if not hist:
            continue
        best_so_far = np.minimum.accumulate(hist)
        ax1.plot(np.arange(1, len(best_so_far)+1), best_so_far,
                 color=COLORS.get(p, "gray"), linewidth=2,
                 label=f"p={p}  (dist={convergence_data[p]['distance']:.3f})")
        ax1.axhline(best_so_far[-1], color=COLORS.get(p, "gray"),
                    linestyle="--", linewidth=0.7, alpha=0.4)

    ax1.set_xlabel("Iterations COBYLA")
    ax1.set_ylabel("Best Ising Energy")
    ax1.set_title("Convergence")
    ax1.legend(fontsize=9)
    ax1.spines[["top","right"]].set_visible(False)
    ax1.grid(axis="y", linewidth=0.4, alpha=0.5)

    # Panneau droit : ratio d'approximation global par p
    if has_ratio:
        ps_r   = [p for p in p_values if convergence_data[p].get("ratio") is not None]
        ratios = [convergence_data[p]["ratio"] for p in ps_r]

        bars = ax2.bar([str(p) for p in ps_r], ratios,
                       color=[COLORS.get(p, "gray") for p in ps_r],
                       width=0.4, edgecolor="none")
        for bar, r in zip(bars, ratios):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.002,
                     f"{r:.4f}",
                     ha="center", va="bottom", fontsize=9, color="#2C2C2A")

        ax2.axhline(1.0, color="#E24B4A", linestyle="--",
                    linewidth=1.2, label=f"Optimal = {true_optimal:.4f}")
        ax2.axhline(1.05, color="#BA7517", linestyle=":",
                    linewidth=1.0, label="Seuil 5%")
        y_max = max(ratios) + 0.05
        y_min = max(0.90, min(ratios) - 0.05)
        ax2.set_ylim(y_min, y_max)
        ax2.set_xlabel("Depth p")
        ax2.set_ylabel("Ratio global (QAOA / optimal)")
        ax2.set_title("Global Approximation Ratio")
        ax2.legend(fontsize=9)
    else:
        # Pas d'optimal disponible : afficher les distances absolues
        ps_v   = list(p_values)
        dists  = [convergence_data[p]["distance"] for p in ps_v]
        bars   = ax2.bar([str(p) for p in ps_v], dists,
                         color=[COLORS.get(p, "gray") for p in ps_v],
                         width=0.4, edgecolor="none")
        for bar, d in zip(bars, dists):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.01 * max(dists),
                     f"{d:.3f}",
                     ha="center", va="bottom", fontsize=9, color="#2C2C2A")
        ax2.set_xlabel("Depth p")
        ax2.set_ylabel("Distance totale QAOA")
        ax2.set_title("Solution Quality (no exact baseline)")

    ax2.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    path = f"convergence_instance{instance_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Graphique : {path}")'''


# ─────────────────────────────────────────────
# 12. EXÉCUTION
# ─────────────────────────────────────────────

if __name__ == "__main__":

    all_results = {}
    for inst_id in [1, 2, 3, 4, 7]:
        all_results[inst_id] = solve_instance(inst_id, p_values=(1, 2, 3))

    print(f"\n{'='*75}")
    print("  TABLEAU RÉCAPITULATIF GLOBAL")
    print(f"{'='*75}")
    print(f"  {'Inst':<6}{'p':<5}{'k':<6}{'Qubits':<10}"
          f"{'Portes':<10}{'Temps(s)':<12}{'Distance':<14}{'Optimal':<14}{'Ratio'}")
    print(f"  {'─'*78}")
    for inst_id, res in all_results.items():
        for p, d in sorted(res.items()):
            opt_str   = f"{d['optimal']:.4f}" if d.get('optimal') is not None else "N/A"
            ratio_str = f"{d['ratio']:.4f}"   if d.get('ratio')   is not None else "N/A"
            print(f"  {inst_id:<6}{p:<5}{d['k']:<6}{d['n_qubits']:<10}"
                  f"{d['n_gates']:<10}{d['exec_time']:<12.1f}"
                  f"{d['distance']:<14.4f}{opt_str:<14}{ratio_str}")
        print()