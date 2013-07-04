"""
Functions for the standard distance measures for undirected graphs.
"""

import staticgraph as sg
from numpy import empty, uint32, amax, amin, array, arange
from random import sample
from staticgraph.exceptions import StaticGraphNodeAbsentException
from staticgraph.exceptions import StaticGraphDisconnectedGraphException

def eccentricity(G, nodes = None):
    """
    Return the eccentricity of nodes in G.

    The eccentricity of a node v is the maximum distance from v to
    all other nodes in G.

    Parameters
    ----------
    G     :  An undirected staticgraph

    nodes :  Iterator of nodes for which eccentricity is to be found, optional.
             If empty, returns eccentricity for all nodes

    Returns
    -------
    ecc   :  A 2D numpy array.
             Row 1: Node labels
             Row 2: Eccentricity of corresponding nodes.

    Notes
    -----
    Exception is raised if given graph has disconnected components.
    """

    if nodes == None:
        nodes = arange(order, dtype = uint32)

    order = G.order()
    nodes = array(nodes, dtype = uint32)
    n_nodes = nodes.size
    
    ecc = empty((2, n_nodes), dtype = uint32)
    index = 0
    for i in xrange(n_nodes):
        if nodes[i] >= order:
            raise StaticGraphNodeAbsentException("given node absent in graph!!")
        sp = sg.graph_traversal.bfs_all(G, nodes[i])
        ecc[0, i] = nodes[i]
        if sp[1].size < n_nodes:
           raise StaticGraphDisconnectedGraphException("disconnected graph!!")
        else:
            ecc[1, i] = sp[0].size - 2

    return ecc

def diameter(G,  n_nodes, e = None):
    """
    Return the diameter of G.

    The diameter is the maximum eccentricity.

    Parameters
    ----------
    G       : A static graph

    n_nodes : Total no.of nodes considered.

    e       : eccentricity numpy 2D array, optional
              A precomputed numpy 2D array of eccentricities.

    Returns
    -------
    Diameter of graph

    See Also
    --------
    eccentricity
    """

    if e is None:
        nodes = sample(xrange(G.order()), n_nodes)    
        e = eccentricity(G , nodes)
    e = e[1]
    if e.size == 0:
        return None
    return amax(e)

def radius(G, n_nodes, e = None):
    """
    Return the radius of a subgraph of G.

    The radius is the minimum eccentricity.

    Parameters
    ----------
    G : A static graph

    n_nodes : Total no.of nodes considered.

    e : eccentricity numpy 2D array, optional
      A precomputed numpy 2D array of eccentricities.

    Returns
    -------
    Radius of graph

    See Also
    --------
    eccentricity
    """

    if e is None:
        nodes = sample(xrange(G.order()), n_nodes)    
        e = eccentricity(G , nodes)
    e = e[1]
    if e.size == 0:
        return None
    return amin(e)

def periphery(G, nodes = None, e = None):
    """
    Return the periphery of a subgraph of G. 

    The periphery is the set of nodes with eccentricity equal to the diameter. 

    Parameters
    ----------
    G     : An undirected staticgraph

    nodes : Iterator of nodes for which periphery is to be found, optional.
            If empty, computation done for all nodes

    e     : eccentricity numpy 2D array, optional
            A precomputed numpy 2D array of eccentricities.

    Returns
    -------
    p : numpy array of nodes in periphery
    """
    
    if e is None:
        e = eccentricity(G, nodes)

    sort_indices = e[1].argsort()
    e[0] = e[0][sort_indices]
    e[1] = e[1][sort_indices]

    i = e[0].size - 1
    while i >= 0:
        if e[1, i] < e[1, -1]:
            break
        i -= 1

    return e[0][i + 1:]

def center(G, nodes = None, e = None):
    """
    Return the center of a subgraph of G. 

    The center is the set of nodes with eccentricity equal to the radius. 

    Parameters
    ----------
    G     : An undirected staticgraph

    nodes : Iterator of nodes for which center is to be found, optional.
            If empty, computation done for all nodes

    e     : eccentricity numpy 2D array, optional
            A precomputed numpy 2D array of eccentricities.


    Returns
    -------
    p : numpy array of nodes in center
    """
    
    if e is None:
        e = eccentricity(G, nodes)

    sort_indices = e[1].argsort()
    e[0] = e[0][sort_indices]
    e[1] = e[1][sort_indices]

    i = 0
    while i < e[0].size:
        if e[1, i] > e[1, 0]:
            break
        i += 1

    return e[0][:i]
