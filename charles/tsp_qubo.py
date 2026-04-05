import numpy as np


class TSP_QUBO:
    def __init__(self, distance_matrix, nodes, depot=0, A=50, pruning_percentile=0.9):
        self.nodes = [depot] + nodes
        self.n = len(self.nodes)
        self.D = distance_matrix
        self.A = A

        # Calculate a threshold: e.g., ignore the top 10% of longest distances
        all_distances = distance_matrix.flatten()
        self.threshold = np.quantile(all_distances[all_distances > 0], pruning_percentile)

        self.var_index = {}
        self.index_var = {}
        self._create_variables()

    def _create_variables(self):
        idx = 0
        # Optimization: Fix node 0 at time 0 to reduce qubits to (n-1)^2
        for i in range(1, self.n):
            for t in range(1, self.n):
                self.var_index[(i, t)] = idx
                self.index_var[idx] = (i, t)
                idx += 1

    def build_qubo(self):
        num_vars = len(self.var_index)
        Q = np.zeros((num_vars, num_vars))

        # 1. Constraint: Each node (excluding depot) appears exactly once
        for i in range(1, self.n):
            idxs = [self.var_index[(i, t)] for t in range(1, self.n)]
            for a in idxs:
                Q[a, a] -= 2 * self.A
                for b in idxs:
                    if a != b: Q[a, b] += 2 * self.A

        # 2. Constraint: Each position (excluding time 0) is occupied once
        for t in range(1, self.n):
            idxs = [self.var_index[(i, t)] for i in range(1, self.n)]
            for a in idxs:
                Q[a, a] -= 2 * self.A
                for b in idxs:
                    if a != b: Q[a, b] += 2 * self.A

        # 3. Objective: Distance Cost with Pruning
        for t in range(1, self.n - 1):
            for i in range(1, self.n):
                for j in range(1, self.n):
                    if i != j:
                        dist = self.D[self.nodes[i]][self.nodes[j]]
                        # Only add interaction if it's below the threshold
                        if dist < self.threshold:
                            u = self.var_index[(i, t)]
                            v = self.var_index[(j, t + 1)]
                            Q[u, v] += dist

        # 4. Handle Depot connections (Node 0 at t=0 and t=n)
        for i in range(1, self.n):
            # Distance from depot to first stop (t=1)
            dist_start = self.D[self.nodes[0]][self.nodes[i]]
            if dist_start < self.threshold:
                idx_start = self.var_index[(i, 1)]
                Q[idx_start, idx_start] += dist_start

            # Distance from last stop (t=n-1) back to depot
            dist_end = self.D[self.nodes[i]][self.nodes[0]]
            if dist_end < self.threshold:
                idx_end = self.var_index[(i, self.n - 1)]
                Q[idx_end, idx_end] += dist_end

        return Q
    
    def decode(self, bitstring):
        route = [None]*self.n
        for idx, val in enumerate(bitstring):
            if val == 1:
                i,t = self.index_var[idx]
                route[t] = self.nodes[i]
        # ensure start/end at depot
        return [self.nodes[0]] + [r for r in route[1:] if r is not None] + [self.nodes[0]]