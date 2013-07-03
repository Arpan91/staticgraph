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
                 If true, computation is done using 
                 dijkstra's algorithm for weighted graphs.
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

    if dijkstra == True:
        if nodes is None:
            nodes = arange(G.order(), dtype = uint32)
        else:
            nodes = array(nodes, dtype = uint32)
    
        closeness_centrality=empty(nodes.size, dtype = float64)

        order = G.order()

        for i in xrange(nodes.size):
            sp=sg.dijkstra.dijkstra_all(G, nodes[i])
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
