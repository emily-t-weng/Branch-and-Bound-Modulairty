import numpy as np

def converter(x,b,l):
    res = np.zeros(l,dtype=int)
    for i in range(l):
        res[l-i-1] = x % b
        x = x // b
    return res

def modularity_matrix(A):
    k = A.sum(axis=1)
    m = k.sum()/2.0
    B = A - np.outer(k, k)/(2.0 * m)
    return B

def modularity_max(A, r):
    n = A.shape[0]
    k = A.sum(axis=1)
    m = k.sum()/2.0
    B = modularity_matrix(A)
    best_Q = -np.inf
    best_S = None
    # get all S and compute Q directly
    for i in range(r**n):
        v = converter(i, r, n)
        S = np.zeros((n, r), dtype=int)
        for j in range(n):
            S[j, v[j]] = 1
        Q = np.trace(S.T@B@S)/(m*2.0)
        if Q > best_Q:
            best_Q = Q
            best_S = S
            best_v = v
    return best_Q, best_S, best_v

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
    best_Q, best_S, best_v = modularity_max(A, r)
    print(best_Q)
    print(best_v)

