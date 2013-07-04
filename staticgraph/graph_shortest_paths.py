"""
Module implementing the standard shortest path algorithms for unweighted graphs
"""

import staticgraph as sg
from numpy import arange, int32, empty, uint32
from itertools import imap
from staticgraph.exceptions import StaticGraphNodeAbsentException

def single_source_shortest_path(G, s):
    """
    Returns a sequence of vertices alongwith their predecessor in their 
    shortest paths from source node s for an undirected staticgraph G.
    
    Parameters
    ----------
    G : An undirected staticgraph.
    s : Source node.
    
    Returns
    -------
    path : A 2-dimensional numpy int32 array.
          1st row: The set of vertices.
          2nd row: Corresponding predecessors in shortest path from the source
          
    Notes
    ------

    It is mandatory that G be undirected.
    predecessor = 2**32 - 1 for source node & unreachable nodes.
    """
    
    if s >= G.order():
        raise StaticGraphNodeAbsentException("source node absent in graph!!")
    
    order = G.order()
    path = arange((order * 2), dtype = uint32)
    path[order : order * 2] = (2 ** 32) - 1
    path = path.reshape(2, order)
    
    queue = empty(order, dtype = uint32)
    
    front = rear = 0
    queue[rear] = s
    rear = 1

    while (front != rear):
        u = queue[front]
        front = front + 1
        start = G.n_indptr[u]
        stop = G.n_indptr[u + 1]
        for v in imap(int, G.n_indices[start:stop]):
            if path[1, v] == (2 ** 32) - 1 and v != s:
                path[1, v] = u
                queue[rear] = v
                rear = rear + 1
    return path

def single_source_shortest_path_dist(G, s):
    """
    Returns a sequence of vertices alongwith the length of their 
    shortest paths from source node s for an undirected staticgraph G.
    
    Parameters
    ----------
    G : An undirected staticgraph.
    s : Source node.
    
    Returns
    -------
    dist : A 2-dimensional numpy int32 array.
          1st row: The set of vertices.
          2nd row: Corresponding distances in shortest path from the source
          
    Notes
    ------

    It is mandatory that G be undirected.
    """

    order = G.order()
    dist = empty((order * 2), dtype = uint32)
    dist = dist.reshape(2, order)
    index = 0
    indptr, indices = sg.graph_traversal.bfs_all(G, s)
    for i in xrange(indptr.size - 1):
        start, stop = indptr[i], indptr[i+1]
        for j in xrange(start, stop):
            dist[0, index] = indices[j]
            dist[1, index] = i
            index += 1

    return dist[:, :index]

def floyd_warshall(G):
    """
    Find all-pairs shortest path lengths using Floyd-Warshall's algorithm.

    Parameters
    ----------
    G : An undirected unweighted simple staticgraph

    Returns
    -------
    Matrix : A 3-Dimensional NumPy Array.
             
             Matrix(:, :, 0) represents the distances of shortest paths 
             among all the pairs of vertices in G.

             Matrix(:, :, 1) represents the immediate predecessors
             in the corresponding shortest paths. 
             
    Notes
    ------
    Floyd's algorithm is appropriate for finding shortest paths in
    dense graphs or graphs with negative weights when Dijkstra's
    algorithm fails.  This algorithm can still fail if there are
    negative cycles.  It has running time O(n^3) with running space of O(n^2).
    
    G must be undirected, simple and unweighted.

    Distance > G.order() for node pair (i, j) if j is unreachable from i.
    Predecessor is -1 for node pair (i, j) if j is not reachable from i.
    """
    
    order = G.order()
    matrix = empty((order, order, 2), dtype = int32)
    matrix[:, :, 0] = order + 1
    matrix[:, :, 1] = -1
    for (i, j) in G.edges():
        matrix[i, j, 0] = matrix[j, i, 0] = 1
        matrix[i, j, 1] = i
        matrix[j, i, 1] = j
    for k in xrange(order):
        for i in xrange(order):
            for j in xrange(order):
                if matrix[i, j, 0] > (matrix[i, k, 0] + matrix[k, j, 0]):
                    matrix[i, j, 0] = (matrix[i, k, 0] + matrix[k, j, 0])
                    matrix[i, j, 1] = matrix[k, j, 1]
    return matrix
