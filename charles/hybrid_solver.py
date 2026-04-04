class HybridClusterer:
    def __init__(self, distance_matrix, demands, capacity, coords, depot=0):
        self.D = distance_matrix
        self.demands = demands
        self.capacity = capacity
        self.coords = coords
        self.depot = depot
        self.n = len(distance_matrix)
    
    def _angle(self, i):
        import math
        x0, y0 = self.coords[self.depot]
        x, y = self.coords[i]
        return math.atan2(y - y0, x - x0)
    
    def sweep(self):
        # exclude depot
        customers = [i for i in range(self.n) if i != self.depot]
        customers.sort(key=lambda i: self._angle(i))
        
        clusters, current, load = [], [], 0
        
        for i in customers:
            if load + self.demands[i] > self.capacity:
                clusters.append(current)
                current, load = [], 0
            
            current.append(i)
            load += self.demands[i]
        
        if current:
            clusters.append(current)
        
        return clusters
    
    def refine(self, clusters, max_iter=20):
        import numpy as np
        
        k = len(clusters)
        centroids = [
            np.mean([self.coords[i] for i in c], axis=0)
            for c in clusters
        ]
        
        for _ in range(max_iter):
            new_clusters = [[] for _ in range(k)]
            loads = [0]*k
            
            for i in range(self.n):
                if i == self.depot:
                    continue  # skip depot
                
                best_c, best_dist = None, float("inf")
                
                for c in range(k):
                    if loads[c] + self.demands[i] > self.capacity:
                        continue
                    
                    cx, cy = centroids[c]
                    x, y = self.coords[i]
                    dist = (x-cx)**2 + (y-cy)**2
                    
                    if dist < best_dist:
                        best_dist, best_c = dist, c
                
                if best_c is None:
                    best_c = loads.index(min(loads))
                
                new_clusters[best_c].append(i)
                loads[best_c] += self.demands[i]
            
            # update centroids
            for c in range(k):
                if new_clusters[c]:
                    pts = np.array([self.coords[i] for i in new_clusters[c]])
                    centroids[c] = np.mean(pts, axis=0)
            
            clusters = new_clusters
        
        return clusters
    
    def cluster(self):
        return self.refine(self.sweep())