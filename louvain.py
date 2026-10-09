import time
import numpy as np
import networkx as nx
from networkx.algorithms import community
from modularity_list import computeQ


G1 = nx.read_edgelist("testing data/ca-HepTh.txt", comments='#', nodetype=int)
G1 = nx.convert_node_labels_to_integers(G1, first_label=0, ordering='sorted')
G2 = nx.read_edgelist("testing data/web-NotreDame.txt", comments='#', nodetype=int)
G2 = nx.convert_node_labels_to_integers(G2, first_label=0, ordering='sorted')

def convert_to_v(communities, n):
    v = np.zeros(n, dtype=int)
    for c, nodes in enumerate(communities):
        for i in nodes:
            v[i] = c
    return v

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
    G = nx.from_numpy_array(A)
    start = time.time()
    louvain_communities = community.louvain_communities(G, seed=123)
    louvain_time = time.time() - start
    louvain_modularity = community.modularity(G, louvain_communities)
    r_louvain = len(louvain_communities)
    print("Louvain Q:", louvain_modularity)
    print("Louvain time:", louvain_time)
    print("Number of Communities:", r_louvain)
    print("Communities:", convert_to_v(louvain_communities, 10))

    G3 = nx.karate_club_graph()
    start = time.time()
    louvain_communities = community.louvain_communities(G3, seed=123)
    louvain_time = time.time() - start
    louvain_modularity = community.modularity(G3, louvain_communities)
    r_louvain = len(louvain_communities)
    print("Louvain Q Karate:", louvain_modularity)
    print("Louvain time Karate:", louvain_time)
    print("Number of Communities Karate:", r_louvain)

    start = time.time()
    louvain_communities = community.louvain_communities(G1, seed=123)
    louvain_time = time.time() - start
    louvain_modularity = community.modularity(G1, louvain_communities)
    r_louvain = len(louvain_communities)
    print("Louvain Q hepth:", louvain_modularity)
    print("Louvain time hepth:", louvain_time)
    print("Number of Communities hepth:", r_louvain)

    # test louvain modularity
    sorted_nodes = sorted(G1.nodes())
    test_v = convert_to_v(louvain_communities, len(G1))
    hepth = nx.to_scipy_sparse_array(G1, nodelist=sorted(G1.nodes()), format='csr')
    k = np.asarray(hepth.sum(axis=1)).ravel()
    m = k.sum() / 2.0
    print("Manual Louvain Q hepth:", computeQ(hepth, test_v, k=k, m=m))


    start = time.time()
    louvain_communities = community.louvain_communities(G2, seed=123)
    louvain_time = time.time() - start
    louvain_modularity = community.modularity(G2, louvain_communities)
    r_louvain = len(louvain_communities)
    print("Louvain Q Notre Dame:", louvain_modularity)
    print("Louvain time Notre Dame:", louvain_time)
    print("Number of Communities Notre Dame:", r_louvain)
  
    #  test louvain modularity
    test_v = convert_to_v(louvain_communities, len(G2))
    notredame = nx.to_scipy_sparse_array(G2, nodelist=sorted(G2.nodes()), format='csr')
    k = np.asarray(notredame.sum(axis=1)).ravel()
    m = k.sum() / 2.0
    print("Manual Louvain Q Notre Dame:", computeQ(notredame, test_v, k=k, m=m))