import numpy as np
import scipy.sparse as sp
# from manual_sum import modularity_bounds

# double summation as on wikipedia page, A is in csr format
def computeQ_sum(A, v, k, m):
	B = A - np.outer(k, k)/(2.0 * m)
	Q = 0.0
	lowerQ = 0.0
	upperQ = 0.0
	for i in range(len(v)):
		for w in range(len(v)):
			if (v[i]!=-1 and v[w]!=-1) or (i==w):
				if v[i] == v[w]:
					Q += B[i,w]
					lowerQ += B[i,w]
					upperQ += B[i,w]
			else:
				if B[i,w]<0:
					lowerQ += B[i,w]
				else:
					upperQ += B[i,w]
	Q /= (2.0 * m)
	lowerQ /= (2.0 * m)
	upperQ /= (2.0 * m)
	return Q,lowerQ,upperQ

# simplification, A is in csr format
def computeQ_bounds(A, v, k, m):
	Q = 0.0
	lowerQ = 0.0
	upperQ = 0.0
	clusters = np.unique(v)
	clusters = np.setdiff1d(clusters,-1)
	for c in clusters:
		# add contribution to Q for all nodes that are assigned to clusters
		nodes = np.where(v == c)[0]
		val = A[nodes,:][:,nodes].sum() - (k[nodes].sum()**2)/(2.0 * m)
		Q += val
		lowerQ += val
		upperQ += val
	# for all nodes that are unassigned, add whole row/column from matrix B = A - outer(k,k)/(2.0*m)
	nodes = np.where(v == -1)[0]
	for i in range(len(nodes)):
		index = nodes[i]
		ignore = nodes[range(i+1)]
		# row (excluding previous nodes and diagonal element)
		temp = A[index,:] - k[index]*k/(2.0 * m)
		temp = np.delete(temp,ignore)
		lowerQ += temp[temp<0].sum()
		upperQ += temp[temp>0].sum()
		# column (excluding previous nodes and diagonal element)
		temp = A[:,index] - k[index]*k/(2.0 * m)
		temp = np.delete(temp,ignore)
		lowerQ += temp[temp<0].sum()
		upperQ += temp[temp>0].sum()
		# add diagonal element to Q, lowerQ, upperQ regardless of its sign
		temp = A[index,index] - k[index]*k[index]/(2.0 * m)
		Q += temp
		lowerQ += temp
		upperQ += temp
	Q /= (2.0 * m)
	lowerQ /= (2.0 * m)
	upperQ /= (2.0 * m)
	return Q,lowerQ,upperQ

# simulated annealing
def modularity_sa(A, r, maxsteps, beta):
	n = A.shape[0]
	k = np.asarray(A.sum(axis=1)).ravel()
	m = k.sum()/2.0
	# track best solution
	x = np.random.randint(0, r, size=n)
	# compute once outside loop and only change if proposal is accepted
	Qx = computeQ_sum(A, x, k, m)[0]
	for i in range(1, maxsteps + 1):
		T = beta / np.log(1 + (i + 1))
		proposal = x.copy()
		j = np.random.randint(0, n)
		proposal[j] = np.random.randint(0, r)
		Qp = computeQ_sum(A, proposal, k, m)[0]
		alpha = np.exp((Qp - Qx) / T)
		if np.random.uniform() < alpha:
			x = proposal
			Qx = Qp
	return x

if __name__ == "__main__":
	# generate random adjacency matrix of dimension (n,n) with edge probability p
	n = 100
	p = 0.5
	A = np.zeros((n,n),dtype=int)
	for i in range(n):
		for j in range(i):
			A[i,j] = np.random.randint(2,size=1)[0]
			A[j,i] = A[i,j]
	print(A)

	# approximate maximal modularity assignment
	csr = sp.csr_matrix(A)
	r = 3
	v = modularity_sa(A,r,100,1)

	# test computeQ
	k = np.asarray(A.sum(axis=1)).ravel()
	m = k.sum()/2.0
	# print(computeQ_sum(A,v,k,m))
	# print(computeQ_trace(A,v,k,m,r))
	# print(computeQ(A,v,k,m))

	unassigned = 20
	indices = np.random.choice(np.arange(n),size=unassigned,replace=False)
	v[indices] = -1
	print(v)
	print(computeQ_sum(A,v,k,m))
	print(computeQ_bounds(A,v,k,m))
