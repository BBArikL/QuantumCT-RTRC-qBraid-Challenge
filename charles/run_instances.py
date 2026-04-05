import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from cvrp_solver import solve_cvrp_parallel
from dotenv import load_dotenv
from visualization import create_route_map, save_map, calculate_total_distance

# Import your existing hybrid solver and visualization modules

load_dotenv()

# -------------------------
# CVRP Instances Definition
# -------------------------

instances = {
    1: {
        "coords": {
            0: (0,0),
            1: (-2,2),
            2: (-5,8),
            3: (2,3)
        },
        "demands": [0,1,1,1],  # assume each customer has demand 1
        "vehicles": 2,
        "capacity": 5
    },
    2: {
        "coords": {
            0: (0,0),
            1: (-2,2),
            2: (-5,8),
            3: (2,3)
        },
        "demands": [0,1,1,1],
        "vehicles": 2,
        "capacity": 2
    },
    3: {
        "coords": {
            0: (0,0),
            1: (-2,2),
            2: (-5,8),
            3: (2,3),
            4: (5,7),
            5: (2,4),
            6: (2,-3)
        },
        "demands": [0,1,1,1,1,1,1],
        "vehicles": 3,
        "capacity": 2
    },
    4: {
        "coords": {
            0: (0,0),
            1: (-2,2), 2: (-5,8), 3: (6,3), 4: (4,4),
            5: (3,2), 6: (0,2), 7: (-2,3), 8: (-4,3),
            9: (2,3), 10: (2,7), 11: (-2,5), 12: (-1,4)
        },
        "demands": [0] + [1]*12,
        "vehicles": 4,
        "capacity": 3
    }
}

# -------------------------
# Resource Usage Table
# -------------------------
resource_usage = []

# -------------------------
# Helper Function: Write solution
# -------------------------
def write_solution_file(instance_number, results):
    filename = f"Instance{instance_number}.txt"
    with open(filename, "w") as f:
        for idx, r in enumerate(results):
            route = r["route"]
            # exclude depot duplicates in text file representation
            route_str = ", ".join(str(n) for n in route)
            f.write(f"r{idx+1}: {route_str}\n")
    print(f"Solution written to {filename}")


# -------------------------
# Run All Instances
# -------------------------
for inst_num, inst in instances.items():
    print(f"\n=== Solving Instance {inst_num} ===")
    coords = inst["coords"]
    demands = inst["demands"]
    capacity = inst["capacity"]
    
    n_customers = len(coords)-1
    start_time = time.time()
    
    # Solve CVRP with parallel clusters
    results, total_cost, max_ops, max_qubits, n_clusters = solve_cvrp_parallel(
        distance_matrix=np.array([
            [np.linalg.norm(np.array(coords[i])-np.array(coords[j])) 
             for j in range(len(coords))] 
            for i in range(len(coords))
        ]),
        demands=demands,
        capacity=capacity,
        coords=coords,
        depot=0
    )
    
    exec_time = time.time() - start_time
    
    # Write solution file
    write_solution_file(inst_num, results)
    
    # Optional: generate interactive map
    m = create_route_map(results, coords, depot=0)
    distance = calculate_total_distance(results, coords, depot=0)
    save_map(m, filename=f"Instance{inst_num}_map.html")
    
    resource_usage.append({
        "Instance": inst_num,
        "# of clusters": n_clusters,
        "Qubits": max_qubits,
        "Gate Ops": max_ops,
        "Distance": distance,
        "Exec Time": round(exec_time,2)
    })

# -------------------------
# Print Resource Usage Table
# -------------------------
import pandas as pd

df = pd.DataFrame(resource_usage)
print("\n=== Resource Usage Summary ===")
print(df.to_markdown(index=False))