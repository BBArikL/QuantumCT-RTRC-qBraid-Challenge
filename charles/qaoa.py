import os

from qbraid_core.services import QuantumRuntimeClient
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qbraid import QbraidProvider
import numpy as np
import random

def solve_qubo_qaoa_qbraid(Q, p=2, max_iter=60, patience=10, shots=1024):
    n = Q.shape[0]
    
    qc = QuantumCircuit(n)
    gammas = [Parameter(f"g{i}") for i in range(p)]
    betas = [Parameter(f"b{i}") for i in range(p)]
    
    for i in range(n):
        qc.h(i)
    
    for layer in range(p):
        for i in range(n):
            if Q[i,i] != 0:
                qc.rz(2*gammas[layer]*Q[i,i], i)
            
            for j in range(i+1, n):
                if Q[i,j] != 0:
                    qc.cx(i,j)
                    qc.rz(2*gammas[layer]*Q[i,j], j)
                    qc.cx(i,j)
        
        for i in range(n):
            qc.rx(2*betas[layer], i)
    
    qc.measure_all()

    client = QuantumRuntimeClient(os.environ['QBRAID_API_KEY'])
    provider = QbraidProvider(client=client)
    device = provider.get_device("qbraid:qbraid:sim:qir-sv")
    
    best_energy = float("inf")
    best_solution = None
    history = []
    no_improve = 0
    
    for it in range(max_iter):
        params = {g: random.uniform(0,np.pi) for g in gammas}
        params.update({b: random.uniform(0,np.pi) for b in betas})
        
        bound = qc.assign_parameters(params)
        job = device.run(bound, shots=shots)
        counts = job.result().data.get_counts()
        
        improved = False
        
        for bitstring in counts:
            x = np.array([int(b) for b in bitstring[::-1]])
            energy = x @ Q @ x
            if energy < best_energy:
                best_energy = energy
                best_solution = x
                improved = True
        
        history.append(best_energy)
        no_improve = 0 if improved else no_improve+1
        
        if no_improve >= patience:
            break
    
    return best_solution, best_energy, history