import numpy as np
import networkx as nx
from new_bnb_modifiedv3 import computeQ_bounds
from sa_modularity import *
import time


# code to enumerate all vectors of length n with entries 0..(r-1)
def enum(A, r, save_count = False):
	n = A.shape[0]
	k = A.sum(axis=1)
	m = k.sum()/2.0
	best_Q = -np.inf 
	visited = 0
	pruned = 0
	leaves = 0
	# partially filled vector v, pointer p to current position
	def explore(v,p, verbose = False):
		nonlocal best_Q, visited, pruned, leaves
		visited += 1

		Q, lower, upper = computeQ_bounds(A, v, k, m)
		if verbose:
			print(f"{v}, Bounds: [{lower:.4f}, {upper:.4f}]")

		# prune if upper bound is lower than the current best Q
		if upper <= best_Q:
			if verbose:
				print("pruned")
			pruned += 1
			return None

		# if all vector entries are assigned then recursion stops and compute full Q of that branch 
		if p==len(v)-1:
			leaves += 1
			if Q > best_Q:
				best_Q = Q
			return Q
		
		# issue r recursive calls with next digit/entry set to 0..(r-1)
		p = p+1
		for i in range(r):
			v[p] = i
			explore(v,p)
			v[p] = -1
			
	# first call of recursive function with all vector entries set to -1
	v = np.zeros(n,dtype=int)-1
	explore(v,-1)
	if save_count:
		return best_Q, visited, pruned, leaves
	return best_Q

if __name__=="__main__":
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
	r = 3
	start = time.time()
	print(enum(A, r))
	print(f"Branch & Bound Runtime: {time.time() - start}")

	start = time.time()
	print(modularity_max(A, r))
	print(f"Brute Force Runtime: {time.time() - start}")

	start = time.time()
	np.random.seed(100)
	print(modularity_sa(A, r, 1000, 0.1))
	print(f"SA Runtime: {time.time() - start}")

	G = nx.karate_club_graph()
	# Adjacency matrix for SA
	adj_matrix = nx.to_numpy_array(G, dtype=int)
	print("Karate club graph")
	print(adj_matrix)

	r = 2
	start = time.time()
	print(enum(adj_matrix, r))
	print(f"Branch & Bound Runtime Karate: {time.time() - start}")

