# CVRP-QAOA Hybrid Solver

## Overview

This repository contains a **hybrid quantum-classical solver** for the **Capacitated Vehicle Routing Problem (CVRP)** using **QAOA via qBraid SDK**. The solver is depot-aware (depot at `(0,0)`), ensures each vehicle starts and ends at the depot, and optimizes routes for multiple vehicles with capacity constraints.

**Key Features:**

- Initial clustering using the **Sweep algorithm**  
- Refinement of clusters via **Capacity K-Means**  
- Optional **Hierarchical Clustering** (experimental)  
- Solving each cluster using **QAOA** with qBraid SDK  
- Parallel execution across clusters for speed  
- Early stopping and convergence tracking  
- Interactive maps and benchmarking visualizations  

---

## Repository Structure

```text
CVRP-QAOA-Hybrid/
│
├─ README.md
├─ requirements.txt
├─ hybrid_solver.py       # Core clustering + QAOA pipeline
├─ map_viz.py             # Interactive map visualization
├─ analysis.py            # Benchmarking and performance plotting
├─ run_instances.py       # Runs latest CVRP instances and exports solutions
│
├─ solutions/             # Solution text files (Instance1.txt, ...)
├─ maps/                  # Interactive HTML route maps
└─ figures/               # Benchmark and convergence plots
````

---

## CVRP Problem

**Capacitated Vehicle Routing Problem (CVRP):**

* **Given:** A depot, customer locations, number of vehicles, vehicle capacity
* **Objective:** Minimize total distance traveled by all vehicles
* **Constraints:**

  1. Each route starts and ends at the depot `(0,0)`
  2. Each customer is visited exactly once
  3. Vehicle capacity is not exceeded

**Mathematical Formulation:**

[
\text{Minimize } \sum_{v \in Vehicles} \sum_{i,j \in Route_v} d_{ij}
]

subject to the above constraints.

---

## Hybrid Solver Pipeline

1. **Initial Clustering:**

   * Sweep Algorithm to form initial clusters based on angles from depot
2. **Refine Clusters:**

   * Capacity K-Means ensures vehicle capacity constraints are satisfied
3. **Solve Clusters with QAOA:**

   * Each cluster is solved as a small TSP using QAOA
   * Parallel execution for speed
   * Early stopping monitors convergence

---

## CVRP Instances Included

| Instance | # Vehicles | Capacity | # Customers |
|----------|------------|----------|-------------|
| 1        | 2          | 5        | 3           |
| 2        | 2          | 2        | 3           |
| 3        | 3          | 2        | 6           |
| 4        | 4          | 3        | 12          |

---

## How to Run

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Run all instances:

```bash
python run_instances.py
```

* Generates:

  * Solution text files in `solutions/`
  * Interactive maps in `maps/`
  * Optional convergence/performance plots in `figures/`

---

## Solution Format

Each solution file (`InstanceX.txt`) contains routes in the format:

```
r1: 0, customer1, customer2, ..., 0
r2: 0, customer3, customer4, ..., 0
```

* `0` denotes the depot
* Each line corresponds to one vehicle route

---

## Benchmark and Performance

Resource usage and performance can be tracked via `analysis.py`. Example metrics include:

| CVRP Instance | # Qubits | # Gate Operations | Execution Time |
|---------------|----------|-------------------|----------------|
| 1             |          |                   |                |
| 2             |          |                   |                |
| 3             |          |                   |                |
| 4             |          |                   |                |

**Convergence and route efficiency plots:**
*(Add your generated images here)*

* Convergence: `figures/convergence.png`
* Route performance: `figures/routes.png`
* Cluster efficiency: `figures/cluster_performance.png`
* Scaling vs. runtime: `figures/scaling.png`

---

## Interactive Route Maps

Each instance has an interactive HTML map:

| Instance | Map File                  |
|----------|---------------------------|
| 1        | `maps/Instance1_map.html` |
| 2        | `maps/Instance2_map.html` |
| 3        | `maps/Instance3_map.html` |
| 4        | `maps/Instance4_map.html` |

* Depot is marked in **red**
* Vehicle routes plotted in different colors

---

## Notes

* The QAOA solver uses **qBraid SDK** (simulator backend)
* The depot `(0,0)` is enforced for all vehicle routes
* Parallelized execution reduces total computation time

---

## Authors

* Your Name
* Hackathon / Research Team

---

## Future Work

* Add hierarchical clustering option fully integrated
* Optimize QAOA parameters using parameter tuning and adaptive schedule
* Run on real quantum backend via qBraid
