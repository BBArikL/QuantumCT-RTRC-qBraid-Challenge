from concurrent.futures import ThreadPoolExecutor

from hybrid_solver import HybridClusterer
from qaoa import solve_qubo_qaoa
from tsp_qubo import TSP_QUBO


def solve_cluster(cluster, distance_matrix, depot=0):
    tsp = TSP_QUBO(distance_matrix, cluster, depot=depot)
    Q = tsp.build_qubo()
    solution, energy, history, n_qubits, n_ops = solve_qubo_qaoa(Q)
    route = tsp.decode(solution)  # decode automatically wraps depot
    
    return {
        "cluster": cluster,
        "route": route,
        "energy": energy,
        "history": history,
        "n_qubits": n_qubits,
        "n_ops": n_ops,
    }

def solve_cvrp_parallel(distance_matrix, demands, capacity, coords, depot=0):
    clusterer = HybridClusterer(distance_matrix, demands, capacity, coords, depot=depot)
    clusters = clusterer.cluster()
    
    results = []
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(solve_cluster, c, distance_matrix, depot) for c in clusters]
        for f in futures:
            results.append(f.result())
    
    total_cost = sum(r["energy"] for r in results)
    max_ops = max(r["n_ops"] for r in results)
    max_qubits = max(r["n_qubits"] for r in results)
    return results, total_cost, max_ops, max_qubits, len(clusters)