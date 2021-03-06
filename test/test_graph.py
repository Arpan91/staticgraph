"""
Tests for undirected graph structure.
"""

import networkx as nx
import staticgraph as sg
from numpy.testing import assert_equal

def pytest_generate_tests(metafunc):
    """
    Generate the arguments for test functions.
    """

    if "graph" in metafunc.funcargnames:
        graphs = []

        # 100 vertex random graph
        a = nx.gnp_random_graph(100, 0.1)
        deg = sg.graph.make_deg(a.order(), a.edges_iter())
        b = sg.graph.make(a.order(), a.size(), a.edges_iter(), deg)
        graphs.append((a, b))

        # 100 vertex random graph with parallel edges
        a = nx.gnp_random_graph(100, 0.1)
        deg = sg.graph.make_deg(a.order(), a.edges() + a.edges())
        b = sg.graph.make(a.order(), 2 * a.size(), a.edges() + a.edges(), deg)
        graphs.append((a, b))

        # 100 vertex random graph with overestimated edge count
        a = nx.gnp_random_graph(100, 0.1)
        deg = sg.graph.make_deg(a.order(), a.edges_iter())
        b = sg.graph.make(a.order(), 2 * a.size(), a.edges_iter(), deg)
        graphs.append((a, b))

        metafunc.parametrize("graph", graphs)

def test_nodes(graph):
    """
    Test the nodes of the graph.
    """

    a = sorted(graph[0].nodes_iter())
    b = sorted(graph[1].nodes())
    assert a == b

def test_edges(graph):
    """
    Test the edges of the graph.
    """

    a = sorted(graph[0].edges_iter())
    b = sorted(graph[1].edges())
    assert a == b

def test_neighbours(graph):
    """
    Test the neighbours for every node.
    """

    for u in graph[0].nodes_iter():
        a = sorted(graph[0].neighbors_iter(u))
        b = sorted(graph[1].neighbours(u))
        assert (u, a) == (u, b)

def test_basics(graph):
    """
    Test graph order, size, and node degrees.
    """

    assert graph[0].order() == graph[1].order()
    assert graph[0].size() == graph[1].size()

    for u in graph[0].nodes_iter():
        assert graph[0].degree(u) == graph[1].degree(u)

def test_load_save(tmpdir, graph):
    """
    Test graph persistance.
    """

    a = graph[1]

    sg.graph.save(tmpdir.strpath, a)
    b = sg.graph.load(tmpdir.strpath)

    assert a.n_nodes == b.n_nodes
    assert a.n_edges == b.n_edges
    assert_equal(a.n_indptr, b.n_indptr)
    assert_equal(a.n_indices, b.n_indices)
