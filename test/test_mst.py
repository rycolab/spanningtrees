import numpy as np
from spanningtrees.mst import MST
from spanningtrees.graph import Graph
from spanningtrees.brute_force import best_tree, best_rc_tree
from spanningtrees.util import random_instance
from arsenal import ok, colors


def test_1b():
    """
    Test that MST decoding's tree and score matches the
    brute-force decoding's tree and score on randomly generated dense graphs.
    """
    print(colors.yellow % '# Regular')
    n = 6
    for _ in range(50):
        A = random_instance(n)
        graph = Graph.build(A)
        mst = MST(graph)
        tree = mst.mst().to_array()
        cost = graph.weight(tree)
        tree_bf, cost_bf = best_tree(A)
        if not np.allclose(tree, tree_bf):
            print(A)
            print("expected:", tree_bf)
            print("\t cost:", cost_bf)
            print("got:     ", tree)
            print("\t cost:", cost)
        assert np.allclose(tree, tree_bf)
        assert np.allclose(cost, cost_bf)

    print(ok)


def test_1b_scc():
    """
    Test that MST (using SCC) decoding's tree and score matches the
    brute-force decoding's tree and score on randomly generated dense graphs.
    """
    print(colors.yellow % '# Regular SCC')
    n = 6
    for _ in range(50):
        A = random_instance(n)
        graph = Graph.build(A)
        mst = MST(graph)
        tree = mst.mst_scc().to_array()
        cost = graph.weight(tree)
        tree_bf, cost_bf = best_tree(A)
        if not np.allclose(tree, tree_bf):
            print(A)
            print("expected:", tree_bf)
            print("\t cost:", cost_bf)
            print("got:     ", tree)
            print("\t cost:", cost)
        assert np.allclose(tree, tree_bf)
        assert np.allclose(cost, cost_bf)

    print(ok)


def test_c1b():
    """
    Test that root-constrained MST decoding's tree and score matches the
    brute-force decoding's tree and score on randomly generated dense graphs.
    """
    print(colors.yellow % '# Root constraint')
    n = 6
    for _ in range(50):
        A = random_instance(n)
        graph = Graph.build(A)
        mst = MST(graph, True)
        tree = mst.mst().to_array()
        cost = graph.weight(tree)
        tree_bf, cost_bf = best_rc_tree(A)
        if not np.allclose(tree, tree_bf):
            print(A)
            print("expected:", tree_bf)
            print("\t cost:", cost_bf)
            print("got:     ", tree)
            print("\t cost:", cost)
        assert np.allclose(tree, tree_bf)
        assert np.allclose(cost, cost_bf)
    print(ok)


def test_c1b_scc():
    """
    Test that root-constrained MST (using SCC) decoding's tree and score matches the
    brute-force decoding's tree and score on randomly generated dense graphs.
    """
    print(colors.yellow % '# Root constraint SCC')
    n = 6
    for _ in range(50):
        A = random_instance(n)
        graph = Graph.build(A)
        mst = MST(graph, True)
        tree = mst.mst_scc().to_array()
        cost = graph.weight(tree)
        tree_bf, cost_bf = best_rc_tree(A)
        if not np.allclose(tree, tree_bf):
            print(A)
            print("expected:", tree_bf)
            print("\t cost:", cost_bf)
            print("got:     ", tree)
            print("\t cost:", cost)
        assert np.allclose(tree, tree_bf)
        assert np.allclose(cost, cost_bf)
    print(ok)


if __name__ == '__main__':
    test_1b()
    test_1b_scc()
    test_c1b()
    test_c1b_scc()
