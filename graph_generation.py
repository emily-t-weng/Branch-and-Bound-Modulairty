import networkx as nx
import numpy as np
import time
import scipy.sparse as sp
from modularity_list import computeQ, modularity_sa

# def make_sbm_sparse(sizes, p_in, p_out, seed=42):
#     probs = np.full((len(sizes), len(sizes)), p_out)
#     np.fill_diagonal(probs, p_in)
#     G = nx.stochastic_block_model(sizes, probs, seed=seed, sparse=True)
#     A = nx.to_scipy_sparse_array(G, format="csr", dtype=np.int32)
#     return A, G.graph["partition"]  

def make_sbm_sparse(sizes, p_in, p_out, seed=42):
    G = nx.random_partition_graph(sizes, p_in, p_out, seed=seed, directed=False)
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=np.int32)
    return A, G.graph["partition"]  

# communities to v (for graph and louvain)
def convert_to_v(communities, n):
    v = np.zeros(n, dtype=int)
    for c, nodes in enumerate(communities):
        for i in nodes:
            v[i] = c
    return v

def assigned_modularity(A, partition):
    n = A.shape[0]
    k = np.asarray(A.sum(axis=1)).ravel()
    m = k.sum() / 2.0
    v_true = convert_to_v(partition, n)
    return computeQ(A, v_true, k, m)

if __name__ == "__main__":
    A, truth = make_sbm_sparse([10]*10, p_in=0.2, p_out=0.02, seed = 0)
    print(A.nnz // 2)
    print(A.shape[0])   
    print(assigned_modularity(A, truth))
    
    start = time.time()
    Q = modularity_sa(A, 10, 1000, 0.005)
    t_sa = time.time() - start
    print(Q)
    print(t_sa)

