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
    m = k.sum() / 2.0
    B = modularity_matrix(A)
    best_Q = -np.inf

    for i in range(r**n):
        base_r_vector = converter(i, r, n)
        Q = 0.0
        for v in range(n):
            for w in range(n):
                if base_r_vector[v] == base_r_vector[w]:
                    Q += B[v, w]
        Q /= (2.0 * m)
        if Q > best_Q:
            best_Q = Q
            best_S_v = base_r_vector

    return best_Q, best_S_v

def modularity_bounds(A, s_partial):
    s = np.asarray(s_partial)
    n = A.shape[0]
    m = A.sum() / 2.0
    B = modularity_matrix(A)

    upper_sum = 0.0
    lower_sum = 0.0

    for v in range(n):
        for w in range(n):
            if v == w:
                upper_sum += B[v, w]
                lower_sum += B[v, w]
            elif s[v] != -1 and s[w] != -1:
                if s[v] == s[w]:
                    upper_sum += B[v, w]
                    lower_sum += B[v, w]
            else:
                if B[v, w] > 0:
                    upper_sum += B[v, w]
                else:
                    lower_sum += B[v, w]

    upper = upper_sum / (2.0 * m)
    lower = lower_sum / (2.0 * m)
    return lower, upper

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
    best_Q, best_S = modularity_max(A, r)
    s_partial = np.array([-1, 0, 0, 1, 1, 1, 2, 2, 2, 1])
    #s_partial = best_S
    lower, upper = modularity_bounds(A, s_partial)
    print(best_Q)
    print(best_S)
    print(lower)
    print(upper)
