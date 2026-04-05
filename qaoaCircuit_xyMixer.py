# ─────────────────────────────────────────────
# 6. CIRCUIT QAOA + XY-MIXEUR
# ─────────────────────────────────────────────

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qubo_ising import qubo_to_ising

def build_qaoa_xy_circuit(n_clients, gamma, beta, h, J, eps=1e-6):
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

    return best_bs, best_qubo, n_q, n_gates, best_hist