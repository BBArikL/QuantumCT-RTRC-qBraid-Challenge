import numpy as np

class TSP_QUBO:
    def __init__(self, distance_matrix, nodes, depot=0, A=50):
        # prepend depot to node list
        self.nodes = [depot] + nodes
        self.n = len(self.nodes)
        self.D = distance_matrix
        self.A = A
        
        self.var_index = {}
        self.index_var = {}
        self._create_variables()
    
    def _create_variables(self):
        idx = 0
        for i in range(self.n):
            for t in range(self.n):
                self.var_index[(i, t)] = idx
                self.index_var[idx] = (i, t)
                idx += 1
    
    def build_qubo(self):
        Q = np.zeros((len(self.var_index), len(self.var_index)))
        
        # constraints: each node appears once
        for i in range(self.n):
            idxs = [self.var_index[(i,t)] for t in range(self.n)]
            for a in idxs:
                Q[a,a] -= 2*self.A
                for b in idxs:
                    if a != b:
                        Q[a,b] += 2*self.A
        
        # constraints: each position occupied once
        for t in range(self.n):
            idxs = [self.var_index[(i,t)] for i in range(self.n)]
            for a in idxs:
                Q[a,a] -= 2*self.A
                for b in idxs:
                    if a != b:
                        Q[a,b] += 2*self.A
        
        # distance cost
        for t in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        Q[
                            self.var_index[(i,t)],
                            self.var_index[(j,(t+1)%self.n)]
                        ] += self.D[self.nodes[i]][self.nodes[j]]
        
        return Q
    
    def decode(self, bitstring):
        route = [None]*self.n
        for idx, val in enumerate(bitstring):
            if val == 1:
                i,t = self.index_var[idx]
                route[t] = self.nodes[i]
        # ensure start/end at depot
        return [self.nodes[0]] + [r for r in route[1:] if r is not None] + [self.nodes[0]]