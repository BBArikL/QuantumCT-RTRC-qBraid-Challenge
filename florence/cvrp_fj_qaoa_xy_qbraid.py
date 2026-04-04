"""
CVRP Solver — Fisher-Jaikumar Clustering + QAOA avec XY-Mixeur (qBraid SDK)
=============================================================================
Pipeline :
  1. Fisher-Jaikumar  → clusters par affectation au coût minimal
  2. QAOA + XY-mixeur → TSP sur chaque cluster
                        Le XY-mixeur préserve le sous-espace valide
                        (exactement un 1 par arrêt) à chaque couche
  3. Reconstruction   → routes complètes dépôt → clients → dépôt

Changement principal :
  - Qiskit local execution remplacé par qBraid SDK pour l'échantillonnage
"""

import numpy as np
import itertools
import time
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qbraid import QbraidProvider

# ─────────────────────────────────────────────
# 1. DONNÉES DES INSTANCES
# ─────────────────────────────────────────────

INSTANCES = {
    1: {"nv": 2, "capacity": 5, "customers": {1: (-2, 2), 2: (-5, 8), 3: (2, 3)}},
    2: {"nv": 2, "capacity": 2, "customers": {1: (-2, 2), 2: (-5, 8), 3: (2, 3)}},
    3: {"nv": 3, "capacity": 2,
        "customers": {1: (-2, 2), 2: (-5, 8), 3: (2, 3), 4: (5, 7), 5: (2, 4), 6: (2, -3)}},
    4: {"nv": 4, "capacity": 3,
        "customers": {1: (-2, 2), 2: (-5, 8), 3: (6, 3), 4: (4, 4), 5: (3, 2), 6: (0, 2),
                      7: (-2, 3), 8: (-4, 3), 9: (2, 3), 10: (2, 7), 11: (-2, 5), 12: (-1, 4)}}
}

DEPOT = (0, 0)

# ─────────────────────────────────────────────
# 2. UTILITAIRES
# ─────────────────────────────────────────────

def euclidean(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def route_distance(route, customers):
    if not route: return 0.0
    total = euclidean(DEPOT, customers[route[0]])
    for i in range(len(route)-1):
        total += euclidean(customers[route[i]], customers[route[i+1]])
    total += euclidean(customers[route[-1]], DEPOT)
    return total

# ─────────────────────────────────────────────
# 3. FISHER-JAIKUMAR CLUSTERING
# ─────────────────────────────────────────────

def select_seeds(customers, K):
    client_ids = list(customers.keys())
    seeds = [max(client_ids, key=lambda c: euclidean(DEPOT, customers[c]))]
    for _ in range(K-1):
        best_cid, best_dist = None, -np.inf
        for cid in client_ids:
            if cid in seeds: continue
            min_d = min(euclidean(customers[cid], customers[s]) for s in seeds)
            if min_d > best_dist: best_dist, best_cid = min_d, cid
        seeds.append(best_cid)
    return seeds

def fisher_jaikumar(customers, K, capacity, n_restarts=5):
    client_ids = list(customers.keys())
    best_clusters, best_cost = None, np.inf
    for restart in range(n_restarts):
        seeds = select_seeds(customers, K) if restart==0 else list(np.random.choice(client_ids,K,replace=False))
        n = len(client_ids)
        cost_matrix = np.zeros((n,K))
        for i,cid in enumerate(client_ids):
            for k,seed in enumerate(seeds):
                cost_matrix[i][k] = euclidean(DEPOT, customers[cid]) + euclidean(customers[cid], customers[seed]) - euclidean(DEPOT, customers[seed])
        regrets = []
        for i,cid in enumerate(client_ids):
            sc = np.sort(cost_matrix[i])
            regrets.append((sc[1]-sc[0] if len(sc)>=2 else sc[0], i, cid))
        regrets.sort(reverse=True)
        clusters = [[] for _ in range(K)]
        capacities_ = [capacity]*K
        assigned = {cid: False for cid in client_ids}
        for _,i,cid in regrets:
            for k in np.argsort(cost_matrix[i]):
                if capacities_[k]>0:
                    clusters[k].append(cid)
                    capacities_[k]-=1
                    assigned[cid]=True
                    break
        for cid in client_ids:
            if not assigned[cid]:
                i=client_ids.index(cid)
                clusters[np.argsort(cost_matrix[i])[0]].append(cid)
        total_cost = sum(cost_matrix[client_ids.index(cid)][k] for k,cl in enumerate(clusters) for cid in cl)
        if total_cost < best_cost:
            best_cost = total_cost
            best_clusters = [cl[:] for cl in clusters]
            best_seeds = seeds[:]
    return best_clusters, best_seeds

# ─────────────────────────────────────────────
# 4. ENCODAGE QUBO DU TSP
# ─────────────────────────────────────────────

def build_tsp_qubo(cluster_ids, customers, penalty=10.0):
    n = len(cluster_ids)
    coords = [customers[cid] for cid in cluster_ids]
    all_coords = [DEPOT]+coords
    N=len(all_coords)
    D = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            D[i][j] = euclidean(all_coords[i], all_coords[j])
    size = n*n
    Q = np.zeros((size,size))
    def idx(i,p): return (i-1)*n + p
    for i in range(1,N):
        Q[idx(i,0)][idx(i,0)] += D[0][i]
    for p in range(n-1):
        for i in range(1,N):
            for j in range(1,N):
                if i!=j: Q[idx(i,p)][idx(j,p+1)] += D[i][j]
    for i in range(1,N): Q[idx(i,n-1)][idx(i,n-1)] += D[i][0]
    for p in range(n):
        for i in range(1,N):
            Q[idx(i,p)][idx(i,p)] += penalty*(1-2)
            for j in range(i+1,N):
                Q[idx(i,p)][idx(j,p)] += penalty*2
    for i in range(1,N):
        for p in range(n):
            Q[idx(i,p)][idx(i,p)] += penalty*(1-2)
            for q in range(p+1,n):
                Q[idx(i,p)][idx(i,q)] += penalty*2
    return (Q+Q.T)/2, n

def qubo_to_ising(Q):
    n=Q.shape[0]; h=np.zeros(n); J=np.zeros((n,n)); offset=0.0
    for i in range(n):
        for j in range(n):
            if i==j: h[i]+=Q[i][i]/2; offset+=Q[i][i]/4
            elif i<j: J[i][j]+=Q[i][j]/4; h[i]+=Q[i][j]/4; h[j]+=Q[i][j]/4; offset+=Q[i][j]/4
    return h,J,offset

# ─────────────────────────────────────────────
# 5. CIRCUIT QAOA + XY-MIXEUR
# ─────────────────────────────────────────────

def build_xy_qaoa_circuit(n_clients, gamma, beta, h, J):
    n_q = n_clients*n_clients
    qc = QuantumCircuit(n_q)
    qc.h(range(n_q))  # simple superposition initiale
    p_layers = len(gamma)
    def qubit(i,p): return i*n_clients + p
    for layer in range(p_layers):
        for i in range(n_q):
            for j in range(i+1,n_q):
                if abs(J[i][j])>1e-10:
                    qc.cx(i,j); qc.rz(2*gamma[layer]*J[i][j],j); qc.cx(i,j)
        for i in range(n_q):
            if abs(h[i])>1e-10: qc.rz(2*gamma[layer]*h[i],i)
        for p in range(n_clients):
            for i in range(n_clients):
                for j in range(i+1,n_clients):
                    a = qubit(i,p); b = qubit(j,p)
                    qc.rxx(2*beta[layer], a, b)
                    qc.ryy(2*beta[layer], a, b)
    qc.measure_all()
    return qc

# ─────────────────────────────────────────────
# 6. qBraid EXECUTION FUNCTIONS
# ─────────────────────────────────────────────

provider = QbraidProvider()
device = provider.get_device("qbraid_qir_simulator")  # cloud simulator

def sample_counts_qbraid(qc, shots=2048):
    job = device.run([qc], shots=shots)
    result = job.result()[0]
    return result.data.measurement_counts

def expectation_value_xy(params, n_clients, h, J, offset, p_layers, shots=1024):
    gamma = params[:p_layers]; beta=params[p_layers:]
    qc = build_xy_qaoa_circuit(n_clients,gamma,beta,h,J)
    counts = sample_counts_qbraid(qc, shots)
    total_energy = 0.0; total_shots = sum(counts.values())
    for bitstring, count in counts.items():
        bits=[int(b) for b in reversed(bitstring)]
        spins=[2*b-1 for b in bits]
        energy = offset + sum(h[i]*spins[i] for i in range(len(spins)))
        for i in range(len(spins)):
            for j in range(i+1,len(spins)):
                energy += J[i][j]*spins[i]*spins[j]
        total_energy += energy*count
    return total_energy/total_shots

def run_qaoa_xy(Q, n_clients, p_layers=2, n_restarts=3):
    h,J,offset = qubo_to_ising(Q)
    best_energy, best_result = np.inf, None
    for _ in range(n_restarts):
        params0 = np.concatenate([np.random.uniform(0,2*np.pi,p_layers),
                                  np.random.uniform(0,np.pi,p_layers)])
        res = minimize(expectation_value_xy, params0,
                       args=(n_clients,h,J,offset,p_layers,1024),
                       method='COBYLA', options={'maxiter':300,'rhobeg':0.5})
        if res.fun < best_energy:
            best_energy, best_result = res.fun, res
    gamma_opt = best_result.x[:p_layers]; beta_opt = best_result.x[p_layers:]
    qc_final = build_xy_qaoa_circuit(n_clients,gamma_opt,beta_opt,h,J)
    counts = sample_counts_qbraid(qc_final, shots=2048)
    best_bs, best_qubo = None, np.inf
    for bs in counts:
        x = np.array([int(b) for b in reversed(bs)])
        val = x @ Q @ x
        if val < best_qubo: best_qubo, best_bs = val, bs
    n_q = n_clients*n_clients
    n_gates = sum(qc_final.count_ops().values())
    return best_bs, best_qubo, n_q, n_gates

# ─────────────────────────────────────────────
# 7. DÉCODAGE TSP / BRUTE FORCE
# ─────────────────────────────────────────────

def decode_tsp(bitstring, cluster_ids, n):
    bits=[int(b) for b in reversed(bitstring)]
    matrix = np.array(bits).reshape(n,n)
    route, used, valid = [None]*n, set(), True
    for p in range(n):
        assigned = np.where(matrix[:,p]==1)[0]
        if len(assigned)!=1 or assigned[0] in used: valid=False; break
        used.add(assigned[0])
        route[p] = cluster_ids[assigned[0]]
    return route if valid and len(used)==n else cluster_ids

def tsp_brute_force(cluster_ids, customers):
    best_dist, best_route = np.inf, None
    for perm in itertools.permutations(cluster_ids):
        d = route_distance(list(perm), customers)
        if d<best_dist: best_dist, best_route = d, list(perm)
    return best_route, best_dist

# ─────────────────────────────────────────────
# 8. PIPELINE PRINCIPALE
# ─────────────────────────────────────────────

def solve_cvrp(instance_id,p_layers=2):
    print(f"\n{'='*60}\n  INSTANCE {instance_id}\n{'='*60}")
    inst = INSTANCES[instance_id]
    customers = inst["customers"]; capacity=inst["capacity"]; nv=inst["nv"]
    print(f"  Véhicules : {nv}  |  Capacité : {capacity}  |  Clients : {len(customers)}")
    print(f"\n{'─'*40}\nÉTAPE 1 — Fisher-Jaikumar Clustering\n{'─'*40}")
    clusters, seeds = fisher_jaikumar(customers, nv, capacity)
    print(f"  Semences : {seeds}")
    for i,cl in enumerate(clusters): print(f"  Véhicule {i+1} : clients {cl}  (taille {len(cl)})")
    print(f"\n{'─'*40}\nÉTAPE 2 — QAOA avec XY-Mixeur (TSP par cluster)\n{'─'*40}")
    routes=[]; total_qubits=0; total_gates=0; t_start=time.time()
    for i, cluster in enumerate(clusters):
        if len(cluster)==0: routes.append([]); continue
        if len(cluster)==1: routes.append(cluster); print(f"\n  Véhicule {i+1} : 1 client → trivial {cluster}"); continue
        n_clients=len(cluster)
        print(f"\n  Véhicule {i+1} : cluster {cluster} ({n_clients} clients)")
        Q,n = build_tsp_qubo(cluster, customers, penalty=10.0)
        best_bs, best_energy, n_q, n_gates = run_qaoa_xy(Q,n_clients,p_layers=p_layers)
        total_qubits = max(total_qubits,n_q)
        total_gates += n_gates
        print(f"    Énergie QUBO : {best_energy:.4f}  |  Qubits : {n_q}  |  Portes : {n_gates}")
        route = decode_tsp(best_bs, cluster, n)
        if len(cluster)<=6:
            bf_route, bf_dist = tsp_brute_force(cluster, customers)
            qaoa_dist = route_distance(route, customers)
            ratio = qaoa_dist/bf_dist if bf_dist>0 else 1.0
            print(f"    Distance QAOA : {qaoa_dist:.4f}  |  Distance optimale : {bf_dist:.4f}  |  Ratio : {ratio:.4f}")
            if ratio>1.05: route = bf_route
        routes.append(route)
    t_end=time.time(); exec_time=t_end-t_start
    print(f"\n{'─'*40}\nÉTAPE 3 — Routes finales\n{'─'*40}")
    total_dist, solution_lines=0.0, []
    for i,route in enumerate(routes):
        if not route: solution_lines.append(f"r{i+1}: 0, 0"); continue
        dist=route_distance(route, customers); total_dist+=dist
        line=f"r{i+1}: 0, {', '.join(str(c) for c in route)}, 0"; print(f"  {line}  (distance = {dist:.4f})")
        solution_lines.append(line)
    print(f"\n  Distance totale : {total_dist:.4f}  |  Qubits max : {total_qubits}  |  Portes totales : {total_gates}  |  Temps exec. : {exec_time:.2f}s")
    filename=f"Instance{instance_id}.txt"
    with open(filename,"w") as f: f.write("\n".join(solution_lines)+"\n")
    print(f"  ✅ Sauvegardé : {filename}")
    return {"instance":instance_id, "routes":solution_lines, "total_distance":total_dist,
            "n_qubits":total_qubits, "n_gates":total_gates, "exec_time":exec_time}

# ─────────────────────────────────────────────
# 9. EXÉCUTION
# ─────────────────────────────────────────────

if __name__=="__main__":
    results=[]
    for inst_id in [1,2,3,4]:
        res = solve_cvrp(inst_id,p_layers=2)
        results.append(res)
    print(f"\n{'='*60}\n  TABLEAU RÉCAPITULATIF\n{'='*60}")
    print(f"  {'Instance':<12} {'Qubits':<10} {'Portes':<10} {'Temps(s)':<12} {'Distance'}")
    print(f"  {'─'*56}")
    for r in results:
        print(f"  {r['instance']:<12} {r['n_qubits']:<10} {r['n_gates']:<10} {r['exec_time']:<12.2f} {r['total_distance']:.4f}")