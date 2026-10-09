import numpy as np
import time
from manual import modularity_max
import scipy.sparse as sp
import networkx as nx

# A is in csr format
def computeQ(A, v, k, m):
    v = np.asarray(v)
    Q = 0.0
    for c in np.unique(v):
        nodes = np.where(v == c)[0]
        Q += A[nodes, :][:, nodes].sum() - (k[nodes].sum() ** 2)/(2.0 * m)
    Q /= (2.0 * m)
    return Q

def modularity_sa(A, r, maxsteps, beta):
    n = A.shape[0]
    k = np.asarray(A.sum(axis=1)).ravel()
    m = k.sum() / 2.0

    x = np.random.randint(0, r, size=n)
    Qx = computeQ(A, x, k, m)   # compute once outside loop only change if proposal is accepted
    for i in range(1, maxsteps + 1):
        T = beta / np.log(1 + (i + 1))
        proposal = x.copy()
        j = np.random.randint(0, n)
        proposal[j] = np.random.randint(0, r)
        Qp = computeQ(A, proposal, k, m)
        alpha = np.exp((Qp - Qx) / T)

        if np.random.uniform() < alpha:
            x = proposal
            Qx = Qp  

    return Qx

def modularity_sa_part(A, r, maxsteps, beta):
    n = A.shape[0]
    k = np.asarray(A.sum(axis=1)).ravel()
    m = k.sum() / 2.0

    x = np.random.randint(0, r, size=n)
    Qx = computeQ(A, x, k, m)   # compute once outside loop only change if proposal is accepted
    for i in range(1, maxsteps + 1):
        T = beta / np.log(1 + (i + 1))
        proposal = x.copy()
        j = np.random.randint(0, n)
        proposal[j] = np.random.randint(0, r)
        Qp = computeQ(A, proposal, k, m)
        alpha = np.exp((Qp - Qx) / T)

        if np.random.uniform() < alpha:
            x = proposal
            Qx = Qp  

    return Qx, x

G = nx.read_edgelist("testing data/ca-HepTh.txt", comments='#', nodetype=int)

if __name__ == "__main__":
    A = np.array([
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    ])
    csr = sp.csr_matrix(A)
    r = 3
    start = time.time()
    print(modularity_max(A, r)[0:2])
    print(f"Brute force time: {time.time() - start}")

    start = time.time()
    np.random.seed(100)
    print("SA Q:", modularity_sa(csr, r, 1000, 0.1))
    print(f"SA time: {time.time() - start}")

    hepth = nx.to_scipy_sparse_array(G, nodelist=sorted(G.nodes()), format='csr')
    start = time.time()
    np.random.seed(100)
    print("SA Large Q:", modularity_sa(hepth, r, 10000000, 0.00001))
    print(f"SA Large time: {time.time() - start}")

   
