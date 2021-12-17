import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix

def metropolis_weights_matrix(adjacency):
    adj = np.array(adjacency)
    degree = np.sum(adj, axis=1)
    consensus = np.zeros_like(adjacency, dtype=float) 
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[0]):
            if adjacency[i, j] > 0:
                consensus[i, j] = 1 / (1 + max(degree[i], degree[j]))
                consensus[j, i] = consensus[i, j] # symmetrical
        consensus[i, i] = 1 - (consensus[i, :].sum())
                
    return consensus

def normalized_laplacian_weights_matrix(adjacency):
   """ 
   References:
   ----------
   https://mathworld.wolfram.com/LaplacianMatrix.html
   """
   adj = np.array(adjacency)
   degree = np.sum(adj, axis=1)
   laplacian = np.diag(degree) - adjacency
   for i in range(adj.shape[0]):
       for j in range(i + 1, adj.shape[0]):
           if adjacency[i, j] > 0:
               laplacian[i, j] = - (1 / np.sqrt(degree[i] * degree[j]))
               laplacian[j, i] = laplacian[i, j] # symmetrical
       laplacian[i, i] = 1
               
   return laplacian


def laplacian_weights_matrix(adjacency, fast=True):
    """Converts adjacency matrix into a consensus matrix
    References:
    ----------
    `Fast Linear Iterations For Distributed Averaging.` --
    Lin Xiao and Stephen Boyd, 2004

    """
    eye = np.eye(*adjacency.shape) 
    degree = np.sum(adjacency, axis=1)
    laplacian = np.diag(degree) - adjacency

    # fast computation -- two largest
    if fast:
        alpha = 1 / sum(sorted(degree, reverse=True)[:2])
    else:
        eig, _ = np.linalg.eig(laplacian)
        alpha = 2 /(eig[0] + eig[-2])

    consensus = eye - alpha * laplacian
    return np.array(consensus)


def adjacency_matrix(n_nodes, n_edges):

    full_edge_list = \
        [(i, j) for i in range(n_nodes - 1) for j in range(i + 1, n_nodes)] 

    edge_ids = \
        np.random.choice(len(full_edge_list), replace=False, size=int(n_edges))

    edge_list = [full_edge_list[i] for i in sorted(edge_ids)]

    data = (np.ones(len(edge_list), dtype=int), zip(*edge_list))
    adjacency = csr_matrix(data, dtype=int, shape=(n_nodes, n_nodes)).toarray()
    adjacency = adjacency + adjacency.T
    return adjacency

# performs distributed averaging on a simple graph.
def main(n_nodes=5, target=3):
    n_edges = 2 * (n_nodes -1) 

    adjacency = adjacency_matrix(n_nodes, n_edges)
    # full_edge_list = \
    #     [(i, j) for i in range(n_nodes - 1) for j in range(i + 1, n_nodes)] 

    # edge_ids = \
    #     np.random.choice(len(full_edge_list), replace=False, size=int(n_edges))

    # edge_list = [full_edge_list[i] for i in sorted(edge_ids)]

    # data = (np.ones(len(edge_list), dtype=int), zip(*edge_list))
    # adjacency = csr_matrix(data, dtype=int, shape=(n_nodes, n_nodes)).toarray()
    # adjacency = adjacency + adjacency.T
    print('ADJACENCY:')
    print(adjacency)

    # generate an array with average == target
    x = np.random.randint(low=0, high=2 * target, size=n_nodes)
    res = target - np.mean(x)
    x = x.astype(np.float32) + res

    print('DATA:')
    print(dict(enumerate(x.tolist())))

    print('Laplacian:')
    C = laplacian_weights_matrix(adjacency, fast=True)
    print(C)
    # C = metropolis_weights_matrix(adjacency)
    # C = normalized_laplacian_weights_matrix(adjacency)
    log = [x]
    n_steps = 99
    for _ in range(n_steps):
        x = C @ x
        log.append(x)

    X = np.linspace(1, n_steps + 1, n_steps + 1)
    Y = np.stack(log)
    T = np.ones(n_steps + 1) * target 
    # specifying horizontal line type
    # Beware that the graph must be fully connected
    plt.axhline(y = target, color = (0.2, 1.0, 0.2), linestyle = '-')
    plt.suptitle('Consensus Iterations')
    plt.ylabel('Data')
    plt.xlabel('Time')
    plt.plot(X, Y)
    plt.show()


if __name__ == '__main__': 
    main()
