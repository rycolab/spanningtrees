import numpy as np
from spanningtrees.kbest import KBest
from spanningtrees.kbest_camerini import KBest as KBestCamerini
from spanningtrees.graph import Graph
from spanningtrees.brute_force import kbest, kbest_rc
from spanningtrees.util import random_instance
from arsenal import ok, colors
from tqdm import tqdm


def test_kbest():
    """
    Test that MST decoding's tree and score matches the
    brute-force decoding's tree and score on randomly generated dense graphs.
    """
    print(colors.yellow % '# K-Best Regular')
    n = 6
    for _ in tqdm(range(100)):
        A = random_instance(n)
        graph = Graph.build(A)
        mst = KBest(graph)
        trees_bf = kbest(A)
        trees = list(mst.kbest())
        assert len(trees) == len(trees_bf)
        for tree, tree_bf in zip(trees, trees_bf):
            cost = graph.weight(tree.to_array())
            cost_bf = graph.weight(tree_bf[0])
            assert np.allclose(cost, cost_bf)

    print(ok)


def test_kbest_rc():
    """
    Test that MST decoding's tree and score matches the
    brute-force decoding's tree and score on randomly generated dense graphs.
    """
    print(colors.yellow % '# K-Best Root constrained')
    n = 6
    for _ in tqdm(range(100)):
        A = random_instance(n)
        graph = Graph.build(A)
        mst = KBest(graph, True)
        trees_bf = kbest_rc(A)
        trees = list(mst.kbest())
        assert len(trees) == len(trees_bf)
        for tree, tree_bf in zip(trees, trees_bf):
            cost = graph.weight(tree.to_array())
            cost_bf = graph.weight(tree_bf[0])
            assert np.allclose(cost, cost_bf)

    print(ok)


def test_kbest_camerini():
    """
    Test that MST decoding's tree and score matches the
    brute-force decoding's tree and score on randomly generated dense graphs.
    """
    print(colors.yellow % '# K-Best Camerini et al.')
    n = 6
    for _ in tqdm(range(100)):
        A = random_instance(n)
        graph = Graph.build(A)
        mst = KBestCamerini(graph)
        trees_bf = kbest(A)
        trees = list(mst.kbest())
        assert len(trees) == len(trees_bf)
        for tree, tree_bf in zip(trees, trees_bf):
            cost = graph.weight(tree.to_array())
            cost_bf = graph.weight(tree_bf[0])
            assert np.allclose(cost, cost_bf)

    print(ok)


if __name__ == '__main__':
    test_kbest()
    test_kbest_camerini()
    test_kbest_rc()
