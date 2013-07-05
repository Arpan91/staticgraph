"""
module to implement different centrality measures for undirected graphs.
"""

import staticgraph as sg
from numpy import empty, float64, array, arange, uint32

def degree_centrality(G):
    """
    Compute the degree centrality for nodes.

    The degree centrality for a node v is the fraction of nodes it
    is connected to.

    Parameters
    ----------
    G : An undirected staticgraph 

    Returns
    -------
    degree_centrality : numpy array having degree centrality of the nodes.

    See Also
    --------
    betweenness_centrality, load_centrality, eigenvector_centrality

    Notes
    -----
    The degree centrality values are normalized by dividing by the maximum 
    possible degree in a simple graph n-1 where n is the number of nodes in G.
    """

    degree_centrality = empty(G.order(), dtype = float64)
    d = G.order() - 1
    for u in G.nodes():
        degree_centrality[u] = float(G.degree(u)) / d
    return degree_centrality

def closeness_centrality(G, nodes=None, dijkstra = False, normalized=True):
    """
    Compute closeness centrality for nodes.

    Closeness centrality at a node is 1/average distance to all other nodes.

    Parameters
    ----------
    G          : An undirected staticgraph 

    nodes      : List of nodes, optional
                 Return only the values for vertices in nodes
                 If None, then computation done for all nodes

    dijkstra   : bool, optional
                 If true, computation is done using dijkstra's algorithm.
                 If false, computation is done using BFS traversal.
                 
    normalized : bool, optional      
                 If True (default) normalize by the graph size.

    Returns
    -------
    2 numpy arrays

    nodes                : Node labels.
    closeness_centrality : corresponding closeness-centalities. 

    See Also
    --------
    betweenness_centrality, load_centrality, eigenvector_centrality,
    degree_centrality

    Notes
    -----
    The closeness centrality is normalized to to n-1 / order(G)-1 where
    n is the number of nodes in the connected part of graph containing
    the node.  If the graph is not completely connected, this
    algorithm computes the closeness centrality for each connected
    part separately.
    """

    
    if nodes is None:
        nodes = arange(G.order(), dtype = uint32)
    else:
        nodes = array(nodes, dtype = uint32)
    
    closeness_centrality=empty(nodes.size, dtype = float64)
    order = G.order()

    sp = []
    for i in xrange(nodes.size):
        if dijkstra == True:
            sp = sg.dijkstra.dijkstra_all(G, nodes[i])
        else:
            sp = sg.graph_shortest_paths.single_source_shortest_path_dist(G, nodes[i])
        totsp=sum(sp[1])
        if totsp > 0.0 and order > 1:
            closeness_centrality[i]= float((sp[1].size - 1)) / totsp
            
            # normalize to number of nodes-1 in connected part
            if normalized:
                s=float(sp[1].size - 1) / (order - 1)
                closeness_centrality[i] *= s
        else:                                                                
            closeness_centrality[i]=0.0           
    
    return nodes, closeness_centrality

def betweenness_centrality(G, nodes, normalized=True, weight=None, endpoints=False):

    """
    Compute the shortest-path betweenness centrality for nodes.

    Betweenness centrality of a node `v` is the sum of the
    fraction of all-pairs shortest paths that pass through `v`:

    .. math::

       c_B(v) =\sum_{s,t \in V} \frac{\sigma(s, t|v)}{\sigma(s, t)}

    where `V` is the set of nodes, `\sigma(s, t)` is the number of 
    shortest `(s, t)`-paths,  and `\sigma(s, t|v)` is the number of those 
    paths  passing through some  node `v` other than `s, t`. 
    If `s = t`, `\sigma(s, t) = 1`, and if `v \in {s, t}`,  
    `\sigma(s, t|v) = 0` [2]_.

    Parameters
    ----------
    G : An undirected staticgraph 

    nodes : List of nodes, optional
      Return only the values for vertices in nodes
      If None, then computation done for all nodes

    normalized : bool, optional  
      If True the betweenness values are normalized by `2/((n-1)(n-2))` 
      for graphs, and `1/((n-1)(n-2))` for directed graphs where `n` 
      is the number of nodes in G.

    weight : None or string, optional  
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.

    endpoints : bool, optional  
      If True include the endpoints in the shortest path counts.

    Returns
    -------
    2 numpy arrays

    nodes       : Node labels.
    betweenness : numpy array of corresponding betweenness-centalities.
    
    See Also
    --------
    edge_betweenness_centrality
    load_centrality

    Notes
    -----
    The algorithm is from Ulrik Brandes [1]_.
    See [2]_ for details on algorithms for variations and related metrics.

    For approximate betweenness calculations set k=#samples to use 
    k nodes ("pivots") to estimate the betweenness values. For an estimate
    of the number of pivots needed see [3]_.

    For weighted graphs the edge weights must be greater than zero.
    Zero edge weights can produce an infinite number of equal length 
    paths between pairs of nodes.

    References
    ----------
    .. [1]  A Faster Algorithm for Betweenness Centrality.
       Ulrik Brandes, 
       Journal of Mathematical Sociology 25(2):163-177, 2001.
       http://www.inf.uni-konstanz.de/algo/publications/b-fabc-01.pdf
    .. [2] Ulrik Brandes: On Variants of Shortest-Path Betweenness 
       Centrality and their Generic Computation. 
       Social Networks 30(2):136-145, 2008.
       http://www.inf.uni-konstanz.de/algo/publications/b-vspbc-08.pdf
    .. [3] Ulrik Brandes and Christian Pich: 
       Centrality Estimation in Large Networks. 
       International Journal of Bifurcation and Chaos 17(7):2303-2318, 2007.
       http://www.inf.uni-konstanz.de/algo/publications/bp-celn-06.pdf
    """

    betweenness = empty(G.order(), dtype = float64)
    
    if nodes is None:
        nodes = arange(G.order(), dtype = uint32)
    else:
        nodes = array(nodes, dtype = uint32)
    for i in xrange(nodes.size):
        s = nodes[i]
        # single source shortest paths
        if weight is None:  # use BFS
            S,P,sigma=_single_source_shortest_path_basic(G,s)
        else:  # use Dijkstra's algorithm
            S,P,sigma=_single_source_dijkstra_path_basic(G,s,weight)
        # accumulation
        if endpoints:
            betweenness=_accumulate_endpoints(betweenness,S,P,sigma,s)
        else:
            betweenness=_accumulate_basic(betweenness,S,P,sigma,s)
    # rescaling
    betweenness=_rescale(betweenness, G.order(),normalized=normalized)

    return betweenness

# helpers for betweenness centrality

def _single_source_shortest_path_basic(G,s):
    if s >= G.order():
        raise StaticGraphNodeAbsentException("source node absent in graph!!")
    
    order = G.order()
    path = empty(order, dtype = uint32)
    pred = empty(order ** 2, dtype = uint32)
    dist = empty(order, dtype = uint32)
    sigma = zeros(order, dtype = uint32)
    sigma[s] = 1
    dist[s] = 0
    pred[:] = (2 ** 32) - 1
    pred = pred.reshape(order, order)
    pred_index = zeros(order, dtype = uint32)
    
    queue = empty(order, dtype = uint32)
    index = 0
    
    front = rear = 0
    queue[rear] = s
    rear = 1

    while (front != rear):
        u = queue[front]
        path[index] = u
        index += 1
        du = d[u]
        sigma_u = sigma[u]
        front = front + 1
        start = G.n_indptr[u]
        stop = G.n_indptr[u + 1]
        for v in imap(int, G.n_indices[start:stop]):
            if pred_index[v] == 0 and v != s:
                dist[v] = dist[u] + 1
                queue[rear] = v
                rear = rear + 1
            if dist[v] == dist[u] + 1:
                pred[v, pred_index[v]] = u
                pred_index[v] += 1
                sigma[v] += sigma[u]
    
    return path, pred, sigma

def _isEmpty(prior_queue):
    """
    Returns whether the priority queue is empty or not.
    """

    return prior_queue[0] == (2 ** 32) - 1

def _min_heapify(heap, weights, i, heap_size):
    """
    Module that maintains heap property of the heap.
    """

    left = (2 * i) + 1
    right = (2 * i) + 2
    smallest = i
    if left < heap_size and weights[heap[left]] < weights[heap[i]]:
        smallest = left
    if right < heap_size and weights[heap[right]] < weights[heap[smallest]]:
        smallest = right
        
    if smallest != i:

        #exchanging the values in the heap
        heap[i], heap[smallest] = heap[smallest], heap[i]
        min_heapify(heap, weights, smallest, heap_size)

def _build_min_heap(p_queue, heap_size, weights):
    """
    Module to create a heap from an unsorted array p_queue
    according to the corresponding values of weights.
    """

    for i in range(heap_size // 2, -1, -1):
        _min_heapify(p_queue, weights, i, heap_size)

def _extract_min_dist(p_queue, weights, heap_size):
    """
    Module to extract the minimum element from the heap
    """

    minimum = p_queue[0]
    p_queue[0] = p_queue[heap_size - 1]
    p_queue[heap_size - 1] = (2 ** 32) - 1
    heap_size -= 1
    
    #Restoring heap property
    _min_heapify(p_queue, weights, 0, heap_size)
    return minimum

def _single_source_dijkstra_path_basic(G, s):
    """
    Returns a sequence of vertices alongwith the length of their 
    shortest paths from source node s for a weighted staticgraph G.
    
    """
    
    if s >= G.order():
        raise StaticGraphNodeAbsentException("source node absent in graph!!")

    order = G.order()
    path = empty(order, dtype = uint32)
    pred = empty(order ** 2, dtype = uint32)
    dist = empty(order, dtype = uint32)
    sigma = zeros(order, dtype = uint32)
    sigma[s] = 1
    dist[s] = 0
    pred[:] = (2 ** 32) - 1
    pred = pred.reshape(order, order)
    pred_index = zeros(order, dtype = uint32)
    count = 0
    index = 0
    heap_size = order
    _build_min_heap(nodes, heap_size, dist)

    while _isEmpty(nodes) == False:
        u = _extract_min_dist(nodes, dist, heap_size)
        path[index] = u
        index += 1
        if dist[u] == (2 ** 64) - 1:
            break
        count += 1
        visited[u] = 1
        heap_size -= 1

           
        for v in G.neighbours(u):
            if visited[v] == 0:
                if dist[v] > (dist[u] + G.weight(u, v)):
                    dist[v] = (dist[u] + G.weight(u, v))
                    sigma[v] += u
                    pred[v, pred_index[v]] = u
                    pred_index[v] += 1     
                
            elif dist[v] == (dist[u] + G.weight(u, v)):
                sigma[v] += u
                pred[v, pred_index[v]] = u
                pred_index[v] += 1     
                    
        _build_min_heap(nodes, heap_size, dist)

    sort_indices = dist.argsort()
    path[:] = path[:][sort_indices]
    sigma = sigma[:][sort_indices]
    return nodes[:count], dist[:count], sigma[:count]
