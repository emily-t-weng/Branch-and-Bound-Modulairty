import numpy as np
import networkx as nx
import scipy.sparse as sp
from networkx.algorithms import community
import time
import matplotlib.pyplot as plt
import metis

from graph_generation import make_sbm_sparse, convert_to_v, assigned_modularity
from modularity_list import computeQ, modularity_sa

# TTS
def tts(t, p, R=0.99):
    if p == 0:
        return np.inf
    if p == 1:
        return t
    return t * np.log(1 - R) / np.log(1 - p)

def estimate_tts_sa(A, partition, r, maxsteps, beta,
                    n_trials=50, R=0.99, tol=0.001, seed=0, Q_target=None):
    n = A.shape[0]
    if Q_target is None:  # default: planted partition
        Q_target = assigned_modularity(A, partition)

    times, Qs, successes = [], [], 0

    for t in range(n_trials):
        np.random.seed(seed + t)
        start = time.time()
        Q = modularity_sa(A, r, maxsteps, beta)
        times.append(time.time() - start)
        Qs.append(Q)
        if Q >= Q_target - tol:
            successes += 1

    p = (successes+1) / (n_trials+1)
    t_run = np.mean(times)
    return {
        "n": n,
        "Q_target": Q_target,
        "Q_mean": np.mean(Qs),
        "Q_std": np.std(Qs),
        "p": p,
        "t_run": t_run,
        "tts": tts(t_run, p, R),
        "successes": successes,
        "n_trials": n_trials,
    }


def estimate_tts_louvain(A, partition, n_trials=50, R=0.99, tol=0.001, seed=0,
                         Q_target=None):
    n = A.shape[0]
    if Q_target is None:  # default: planted partition
        Q_target = assigned_modularity(A, partition)
    G = nx.from_scipy_sparse_array(sp.csr_matrix(A))

    times, Qs, successes = [], [], 0
    for t in range(n_trials):
        start = time.time()
        comms = community.louvain_communities(G, seed=seed + t)
        Q = community.modularity(G, comms)
        times.append(time.time() - start)
        Qs.append(Q)
        if Q >= Q_target - tol:
            successes += 1

    p = (successes+1) / (n_trials+1)
    t_run = np.mean(times)
    return {
        "n": n,
        "Q_target": Q_target,
        "Q_mean": np.mean(Qs),
        "Q_std": np.std(Qs),
        "p": p,
        "t_run": t_run,
        "tts": tts(t_run, p, R),
        "successes": successes,
        "n_trials": n_trials,
    }
