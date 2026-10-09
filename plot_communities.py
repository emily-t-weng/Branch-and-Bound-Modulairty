import numpy as np
import matplotlib.pyplot as plt
from manual import *

def plot_graph(A, pos, save_path, community = None):
    n = A.shape[0]
    fig, ax = plt.subplots(figsize=(10, 10))
    # edges
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 0:
                xi, yi = pos[i]
                xj, yj = pos[j]
                ax.plot([xi, xj], [yi, yj], color="black", linewidth=1.5, zorder=1)
    #color palette
    palette = ["red", "green", "cornflowerblue", "blue", "purple", "yellow", "pink", "orange"]

    # nodes
    if community is not None:
        for i in range(n):
            x, y = pos[i]
            color = palette[community[i]]
            ax.scatter(x, y, s=400, c=color, edgecolors="black", linewidths=1.5, zorder=2)
            ax.text(x, y, str(i), ha="center", va="center", fontsize=12, color="white", zorder=3)
    else:
        for i in range(n):
            x, y = pos[i]
            ax.scatter(x, y, s=400, c="white", edgecolors="black", linewidths=1.5, zorder=2)
            ax.text(x, y, str(i), ha="center", va="center", fontsize=12, color="black", zorder=3)

    ax.axis("off")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    # Just plotting the adjacency matrix
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

    # Position for each node (visually assessed from wikipedia image)
    pos = {
        0: (1, 7),
        1: (1, 10),
        2: (3, 6),
        3: (8, 6),
        4: (10, 10),
        5: (10, 6),
        6: (5, 3),
        7: (0, 0),
        8: (10, 0),
        9: (6, 7),
    }

    plot_graph(A, pos, save_path="network_plot.png")

    # Plot after finding communities
    r = 3
    best_Q, best_S = modularity_max(A, r)
    community = np.argmax(best_S, axis=1)
    print(best_Q)
    print(best_S)
    print(community)
    plot_graph(A, pos, save_path="network_plot_communities.png", community=community)