import time
import numpy as np
import networkx as nx
from enumerate_new_v2 import enum
from graph_generation import make_sbm_sparse
from modularity_list import computeQ, modularity_sa
from enumerate_new_v2 import enum
from manual_sum import modularity_max
import numpy as np
import matplotlib.pyplot as plt
from graph_generation import make_sbm_sparse   
from matplotlib.ticker import ScalarFormatter, NullFormatter


def tree_depth(n, r): # every partial and complete v assignments -> 1 + r + r^2 + ... + r^n
	if r == 1:
		return n + 1
	return (r**(n + 1) - 1) // (r - 1)


def run_sizes(sizes, r, seed = 0):

	results = {"n": [], "visited": [], "pruned": [], "tree_depth": [], "leaves": [], "brute_force_nodes":[]}

	for n in sizes:
		bs = n // r                     
		block_sizes = [bs] * r
		p_in = 0.7
		p_out = 0.1

		A, partition = make_sbm_sparse(block_sizes, p_in, p_out, seed=seed)
		best_Q, visited, pruned, leaves = enum(A, r, save_count=True)

		results["n"].append(n)
		results["visited"].append(visited)
		results["pruned"].append(pruned)
		results["tree_depth"].append(tree_depth(n, r))
		results["leaves"].append(leaves)
		results["brute_force_nodes"].append(r**n)

	return results

LINE_STYLES = {
    "Branch and Bound": {"color": "black", "linestyle": "-",  "marker": "o"},
    "Brute Force":      {"color": "black", "linestyle": "--", "marker": "s"},
    "SA":               {"color": "black", "linestyle": ":",  "marker": "^"},
    "Louvain":          {"color": "black", "linestyle": "-.", "marker": "D"},
}

def plot_pruning_1(results):
	fig, ax = plt.subplots(figsize=(7, 5))

	all_n = results["n"]
	ax.plot(all_n, results["tree_depth"], "o--", label="full tree (no pruning)", color= 'black')
	ax.plot(all_n, results["visited"], "s-", label="nodes visited (branch & bound)", color= 'black')

	ax.set_yscale("log")
	ax.set_xscale("log")
	ax.xaxis.set_major_formatter(ScalarFormatter())
	ax.xaxis.set_minor_formatter(NullFormatter())
	ax.set_xlabel("graph size $n$")
	ax.set_ylabel("nodes in search tree")
	ax.set_xticks(all_n)        
	ax.set_title(f"Branch-and-bound pruning ($r={2}$)")
	ax.grid(True, which="both", alpha=0.3)
	ax.legend()
	
	return fig, ax

def plot_pruning_2(results):
	fig, ax = plt.subplots(figsize=(7, 5))

	all_n = results["n"]
	ax.plot(all_n, results["brute_force_nodes"], "o--", color= 'black', label="all assignments evaluated (brute force)")
	ax.plot(all_n, results["leaves"], "s-", color= 'black', label="full assignments reached (branch & bound)")

	ax.set_yscale("log")
	ax.set_xscale("log")
	ax.xaxis.set_major_formatter(ScalarFormatter())
	ax.xaxis.set_minor_formatter(NullFormatter())
	ax.set_xlabel("graph size $n$")
	ax.set_ylabel("Number of full assignments evaluated (log)")
	ax.set_xticks(all_n)        
	# ax.set_title(f"Branch-and-bound pruning ($r={2}$)")
	# ax.grid(True, which="both", alpha=0.3)
	ax.legend()
	
	return fig, ax

def fit_pruning_2_scaling(results):
	# log(y) = slope * log(n) + intercept

	n = np.array(results["n"], dtype=float)
	series = {
		"brute force": np.array(results["brute_force_nodes"], dtype=float),
		"Branch and Bound": np.array(results["leaves"], dtype=float),
	}
	fits = {}
	for name, y in series.items():
		slope, intercept = np.polyfit(np.log10(n), np.log10(y), 1)
		fits[name] = (slope, intercept)
	return fits


# sizes = [6, 10, 14, 18, 22, 26]
sizes = [6, 9, 12, 15]
results = run_sizes(sizes, r=3)
fig2, ax2 = plot_pruning_2(results)
fig2.savefig("bnb/time_plots/bnb_plots/enum_pruning_3.pdf", dpi=150, bbox_inches="tight")

for name, (slope, intercept) in fit_pruning_2_scaling(results).items():
	print(f"{name}: slope={slope:.4f}, intercept={intercept:.4f}")