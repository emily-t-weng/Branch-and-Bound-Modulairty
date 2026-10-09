import numpy as np
import time
from manual import modularity_max

def modularity_matrix(A):
    k = A.sum(axis=1)
    m = k.sum()/2.0
    B = A - np.outer(k, k)/(2.0 * m)
    return B

def modularity_sa(A, r, maxsteps, beta):
    n = A.shape[0]
    k = A.sum(axis=1)
    m = k.sum()/2.0
    B = modularity_matrix(A)
    def get_S(a):
        S = np.zeros((n, r), dtype=int)
        for j in range(n):
            S[j, a[j]] = 1
        return S

    def Q(a):
        S = get_S(a)
        return np.trace(S.T@B@S)/(m*2.0)
    
    x = np.random.randint(0, r, size=n)
    
    for i in range(1, maxsteps + 1):
        T = beta/np.log(1 + (i + 1))
        # change one random node to a random community
        proposal = x.copy()
        j = np.random.randint(0, n)
        proposal[j] = np.random.randint(0, r)
        alpha = np.exp((Q(proposal) - Q(x))/T)

        if np.random.uniform() < alpha:
            x = proposal

    return Q(x), get_S(x)

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
    r = 3
    
    start = time.time()
    print(modularity_max(A, r))
    print(f"Brute force: {time.time() - start}")

    start = time.time()
    np.random.seed(100)
    print(modularity_sa(A, r, 1000, 0.1))
    print(f"SA: {time.time() - start}")