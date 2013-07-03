"""
Tests for different centrality measures for graphs.
"""

import networkx as nx
import staticgraph as sg
from numpy import array, uint32
from numpy.testing import assert_equal
from random import triangular

def create_iter(edges):
    """
    Returns iterator for edges and their weights (u,v,w) of networkx graphs.
    """

    for i in edges:
        yield i[0], i[1], i[2]['weight']

def pytest_generate_tests(metafunc):
    """
    Generate the arguments for test funcs.
    """

    if "testgraph" in metafunc.funcargnames:
        testgraphs = []

    # 100 vertex random graph
        a = nx.gnp_random_graph(100, 0.1)
        deg = sg.graph.make_deg(a.order(), a.edges_iter())
        b = sg.graph.make(a.order(), a.size(), a.edges_iter(), deg)
        

    # 100 vertex random weighted graph
        c = nx.gnp_random_graph(100, 0.1)
        for e in c.edges_iter(data = True):
            e[2]['weight'] = triangular(10, 80, 30)
        deg = sg.wgraph.make_deg(c.order(), create_iter(c.edges_iter(data = True)))
        d = sg.wgraph.make(c.order(), c.size(), create_iter(c.edges_iter(data = True)), deg)
        testgraphs.append((a, b, c, d))

        metafunc.parametrize("testgraph", testgraphs)

def test_degree_centrality(testgraph):
    """
    Testing degree centrality function for graphs.
    """

    a, b = testgraph[:2]
    nx_deg = nx.degree_centrality(a)
    sg_deg = sg.graph_centrality.degree_centrality(b)
    for i in b.nodes():
        assert abs(sg_deg[i] - nx_deg[i]) < 1e-5

def test_closeness_centrality(testgraph):
    """
    Testing closeness centrality function for graphs.
    """

    a, b,c, d = testgraph
    
    nx_cc = nx.closeness_centrality(a)
    sg_cc = sg.graph_centrality.closeness_centrality(b)
    for i in b.nodes():
        assert abs(sg_cc[1][i] - nx_cc[sg_cc[0][i]]) < 1e-5

    nx_cc = nx.closeness_centrality(c, distance = True)
    sg_cc = sg.graph_centrality.closeness_centrality(d, dijkstra = True)
    for i in b.nodes():
        assert abs(sg_cc[1][i] - nx_cc[sg_cc[0][i]]) < 1e-5
