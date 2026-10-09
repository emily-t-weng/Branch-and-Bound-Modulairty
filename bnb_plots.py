import os
import numpy as np
import networkx as nx
import scipy.sparse as sp
from networkx.algorithms import community
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter

from graph_generation import *
from modularity_list import computeQ, modularity_sa
from enumerate_new_v2 import enum
from manual_sum import modularity_max 
from tts import (tts, assigned_modularity, estimate_tts_sa,
                 estimate_tts_louvain)

# Louvain communities to v
def convert_to_v(communities, n):
    v = np.zeros(n, dtype=int)
    for c, nodes in enumerate(communities):
        for i in nodes:
            v[i] = c
    return v

def estimate_deterministic(solver, A, partition, r, R=0.99, tol=0.001, Q_target=None):
    n = A.shape[0]
    if Q_target is None:  # default: planted partition
        Q_target = assigned_modularity(A, partition)

    start = time.time()
    out = solver(A, r)
    t_run = time.time() - start

    Q = out[0] if isinstance(out, tuple) else out  # unpack if tuple

    success = Q >= Q_target - tol
    return {
        "n": n, 
        "Q_target": Q_target, 
        "Q_mean": Q, 
        "Q_std": 0.0,
        "p": 1.0 if success else 0.0, 
        "t_run": t_run,
        "tts": t_run if success else np.inf,
        "successes": int(success), 
        "n_trials": 1,
    }

def run_simulation(baseline="Brute Force"):
    # sizes = [6, 10, 14, 18, 20, 22, 24]
    sizes = [6, 9, 12]
    block_count = 3
    p_in, p_out = 0.7, 0.08
    r = block_count
    seed = 0
    maxsteps, beta = 1000, 0.1
    n_trials = 50

    methods = {
        "Branch and Bound":  {"kind": "det", "solver": enum,        "max_n": 24},
        "Brute Force": {"kind": "det", "solver": modularity_max, "max_n": 24},
        "SA":          {"kind": "sa"},
        "Louvain":     {"kind": "louvain"},
    }

    results = {name: [] for name in methods}

    # run the baseline first so its Q can be the success target for the rest
    order = [baseline] + [name for name in methods if name != baseline]

    for n in sizes:
        bs = n // block_count
        A, partition = make_sbm_sparse([bs] * block_count, p_in, p_out, seed=seed)

        Q_ref = None  # falls back to planted Q if baseline skipped at this n
        for name in order:
            cfg = methods[name]
            if cfg.get("max_n", np.inf) < n:
                continue
            if cfg["kind"] == "det":
                res = estimate_deterministic(cfg["solver"], A, partition, r,
                                             Q_target=Q_ref)
            elif cfg["kind"] == "sa":
                res = estimate_tts_sa(A, partition, r, maxsteps, beta,
                                      n_trials=n_trials, seed=seed, Q_target=Q_ref)
            elif cfg["kind"] == "louvain":
                res = estimate_tts_louvain(A, partition, n_trials=n_trials,
                                           seed=seed, Q_target=Q_ref)
            if name == baseline:
                Q_ref = res["Q_mean"]
            results[name].append(res)

    return results


LINE_STYLES = {
    "Branch and Bound": {"color": "black", "linestyle": "-",  "marker": "o"},
    "Brute Force":      {"color": "black", "linestyle": "--", "marker": "s"},
    "SA":               {"color": "black", "linestyle": ":",  "marker": "^"},
    "Louvain":          {"color": "black", "linestyle": "-.", "marker": "D"},
}


def plot_time(results):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, rs in results.items():
        if rs:
            x = [r["n"] for r in rs]
            y = [r["t_run"] for r in rs]
            style = LINE_STYLES.get(name, {"color": "black", "linestyle": "-", "marker": "o"})
            ax.plot(x, y, label=name, **style)
    ax.set_xlabel("n"); ax.set_ylabel("runtime (s)")
    ax.set_yscale("log")
    # ax.set_title("Mean runtime")
    all_n = sorted({r["n"] for rs in results.values() for r in rs if rs})
    ax.set_xticks(all_n)
    ax.legend()
    return fig, ax


def plot_accuracy(results, baseline="Brute Force"):
    # Q_opt from the exact baseline at each n
    Q_opt = {r["n"]: r["Q_mean"] for r in results[baseline]}

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, rs in results.items():
        if name == baseline:
            continue
        rs = [r for r in rs if r["n"] in Q_opt]  # only n where baseline ran
        if rs:
            x = [r["n"] for r in rs]
            gap = [Q_opt[r["n"]] - r["Q_mean"] for r in rs]
            style = LINE_STYLES.get(name, {"color": "black", "linestyle": "-", "marker": "o"})
            ax.plot(x, gap, label=name, **style)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("n"); ax.set_ylabel(r"$Q_{opt} - \bar{Q}$")
    # ax.set_title("Accuracy gap (mean ± std)")
    all_n = sorted(Q_opt)
    ax.set_xticks(all_n)
    ax.legend()
    return fig, ax


def fit_tts_scaling(results):
    # log(TTS) = slope * log(n) + intercept
    fits = {}
    for name, rs in results.items():
        x = np.array([r["n"] for r in rs], dtype=float)
        y = np.array([r["tts"] for r in rs], dtype=float)
        slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
        fits[name] = (slope, intercept)
    return fits


def plot_tts(results, baseline="Brute Force"):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, rs in results.items():
        if name == baseline:
            continue
        if rs:
            x = [r["n"] for r in rs]
            y = [r["tts"] for r in rs]
            style = LINE_STYLES.get(name, {"color": "black", "linestyle": "-", "marker": "o"})
            ax.plot(x, y, label=name, **style)
    ax.set_xlabel("n"); ax.set_ylabel("TTS (s)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    # ax.set_title("Time to solution")
    all_n = sorted({r["n"] for name, rs in results.items() if name != baseline for r in rs})
    ax.set_xticks(all_n)
    ax.legend()
    return fig, ax


if __name__ == "__main__":
    results = run_simulation()

    fig, ax = plot_time(results)
    fig.savefig("bnb/time_plots/bnb_plots/mean_time_3_test.pdf", dpi=150, bbox_inches="tight")

    fig2, ax2 = plot_accuracy(results)
    fig2.savefig("bnb/time_plots/bnb_plots/accuracy_3_test.pdf", dpi=150, bbox_inches="tight")

    fig3, ax3 = plot_tts(results)
    fig3.savefig("bnb/time_plots/bnb_plots/tts_3_test.pdf", dpi=150, bbox_inches="tight")

    # for name, (slope, intercept) in fit_tts_scaling(results).items():
    #     print(f"{name}: slope={slope:.4f}, intercept={intercept:.4f}")