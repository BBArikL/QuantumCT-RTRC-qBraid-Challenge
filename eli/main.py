import numpy as np
import matplotlib.pyplot as plt
from qiskit_optimization.applications import Tsp
from qbraid.runtime.native import QAOA
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# --- 1. CLUSTERING DE FISHER-JAIKUMAR (APPROCHÉ) ---
def fisher_jaikumar_clustering(nodes, n_vehicles, capacity):
    """
    Approche par graines et coût d'insertion.
    Chaque client a une demande de 1.
    """
    depot = np.array(nodes[0])
    n_total = len(nodes)
    
    # Sélection des graines : les points les plus éloignés du dépôt
    dists_to_depot = [np.linalg.norm(np.array(n) - depot) for n in nodes[1:]]
    seed_indices = np.argsort(dists_to_depot)[-n_vehicles:]
    # Correction d'index car on a ignoré le dépôt (index 0)
    seed_indices = [idx + 1 for idx in seed_indices]
    
    clusters = [[idx] for idx in seed_indices]
    cluster_loads = [1] * n_vehicles
    
    # Liste des clients restants (ni dépôt, ni graines)
    remaining_clients = [i for i in range(1, n_total) if i not in seed_indices]
    
    for client_idx in remaining_clients:
        client_pos = np.array(nodes[client_idx])
        insertion_costs = []
        
        for j in range(n_vehicles):
            seed_pos = np.array(nodes[seed_indices[j]])
            # Coût d'insertion simplifié (distance à la graine)
            cost = np.linalg.norm(client_pos - seed_pos)
            insertion_costs.append((cost, j))
        
        # On trie pour trouver la graine la plus proche
        insertion_costs.sort()
        
        assigned = False
        for _, cluster_idx in insertion_costs:
            if cluster_loads[cluster_idx] < capacity:
                clusters[cluster_idx].append(client_idx)
                cluster_loads[cluster_idx] += 1
                assigned = True
                break
        
        if not assigned:
            print(f"Warning: Client {client_idx} n'a pas pu être assigné (Capacité atteinte).")
            
    return clusters

# --- 2. RÉSOLVEUR TSP VIA qBRAID QAOA ---
def solve_tsp_qbraid(cluster_indices, all_nodes):
    # Préparation des coordonnées locales (Dépôt + Clients du cluster)
    local_nodes = [all_nodes[0]] + [all_nodes[i] for i in cluster_indices]
    n = len(local_nodes)
    
    # Matrice de distance
    adj_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            adj_matrix[i, j] = np.linalg.norm(np.array(local_nodes[i]) - np.array(local_nodes[j]))

    # Création du problème TSP via Qiskit Optimization
    tsp_app = Tsp(adj_matrix)
    qp = tsp_app.to_quadratic_program()
    
    # Conversion en Hamiltonien Ising pour qBraid
    observable, offset = qp.to_ising()
    
    # Utilisation du QAOA natif de qBraid
    # On utilise un simulateur par défaut pour le hackathon
    qaoa_solver = QAOA(observable=observable, reps=2)
    
    print(f"   Exécution QAOA pour un cluster de {n-1} clients...")
    res = qaoa_solver.minimize()
    
    # Interprétation de la solution
    # On récupère la route locale [0, 2, 1, 0] etc.
    local_order = tsp_app.interpret(res)
    
    # Conversion en indices globaux pour la visualisation
    global_route = [0]
    for idx in local_order:
        if idx != 0:
            global_route.append(cluster_indices[idx-1])
    global_route.append(0)
    
    return global_route

# --- 3. PIPELINE PRINCIPAL ---
def main():
    # Simulation des données (Instances du CVRP)
    # Format: (x, y). Le premier est le dépôt (0,0)
    nodes = [
        (0, 0),    # Dépôt (0)
        (8, 2), (2, 9), (-5, 7), (-9, 2), 
        (-4, -6), (2, -8), (9, -3), (5, 5),
        (-2, -2), (3, 3)
    ]
    
    n_vehicles = 3
    capacity = 4  # Max 4 clients par véhicule (Demande = 1)

    print("=== Étape 1 : Clustering Fisher-Jaikumar ===")
    clusters = fisher_jaikumar_clustering(nodes, n_vehicles, capacity)
    for i, c in enumerate(clusters):
        print(f"Véhicule {i+1} : {len(c)} clients assignés -> {c}")

    print("\n=== Étape 2 : Optimisation Quantique (qBraid QAOA) ===")
    plt.figure(figsize=(10, 7))
    colors = ['blue', 'green', 'purple', 'orange']

    for i, cluster in enumerate(clusters):
        route = solve_tsp_qbraid(cluster, nodes)
        print(f"Route finale véhicule {i+1} : {route}")
        
        # Données pour le graphique
        rx = [nodes[node][0] for node in route]
        ry = [nodes[node][1] for node in route]
        plt.plot(rx, ry, marker='o', color=colors[i % len(colors)], label=f'Route {i+1}')

    # Affichage du dépôt
    plt.scatter(nodes[0][0], nodes[0][1], color='red', s=200, marker='X', label='Dépôt', zorder=10)
    
    # Affichage des clients restants non-visités s'il y en a
    all_clients = set(range(1, len(nodes)))
    visited_clients = set([node for cluster in clusters for node in cluster])
    missing = all_clients - visited_clients
    if missing:
        mx = [nodes[m][0] for m in missing]
        my = [nodes[m][1] for m in missing]
        plt.scatter(mx, my, color='gray', label='Non-assignés')

    plt.title(f"Solution CVRP : Fisher Clustering + qBraid QAOA\n(Capacité={capacity}, Véhicules={n_vehicles})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()