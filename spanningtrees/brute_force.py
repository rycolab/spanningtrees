import numpy as np


def enumerate_directed_spanning_trees(A, root, root_weight):
    """
    Enumerate all spanning trees in A with an edge coming from the root,
    given a root_weight.
    Spanning trees are given in the form of (i, j, w) indicating a an edge
    from i to j with weight w.
    Weights are additive.
    """
    n = len(A)

    def enum_dst(cost, included, rest, excluded):
        if len(included) == n:
            return [(rest, cost)]
        dsts = []
        new_excluded = list(excluded)
        for i in included:
            for j in range(n):
                cost_ij = A[i, j]
                if j not in included and (i, j) not in excluded and cost_ij:
                    new_excluded += [(i, j)]
                    dsts += enum_dst(cost + cost_ij, included + [j],
                                     rest + [(i, j, cost_ij)], new_excluded)
        return dsts
    return enum_dst(root_weight, [root], [], [])


def all_unrooted_spanning_trees(A):
    """
    Enumerate all spanning trees that are "unrooted".
    Unrooted trees mean that more than one edge can emanate from the
    dummy root at 0.
    Trees aer returned as numpy arrays where tree[j] = i indicates
    that the tree has the edge (i, j) in it.
    Trees are accompanied by their associated cost
    """
    n = len(A)

    A = np.copy(A)
    # No self loops
    np.fill_diagonal(A, 0)
    # No incoming edges to root
    A[:, 0] = 0.
    dsts = []
    unrooted_dsts = []
    for i in range(n):
        unrooted_dsts += enumerate_directed_spanning_trees(A, i, 0)
    for tree, cost in unrooted_dsts:
        t = - np.ones(n)
        for i, j, _ in tree:
            t[j] = i
        dsts.append((t, cost))
    return dsts


def all_rooted_spanning_trees(W):
    """
        Enumerate all spanning trees that are "rooted".
        Rooted trees mean that exactly one edge can emanate from the
        dummy root at 0.
        Trees aer returned as numpy arrays where tree[j] = i indicates
        that the tree has the edge (i, j) in it.
        Trees are accompanied by their associated cost
        """
    W = np.copy(W)
    # Root weights
    r = W[0, 1:]
    # Adjacency matrix
    A = W[1:, 1:]
    n = len(A)
    # No self loops
    np.fill_diagonal(A, 0)
    dsts = []
    for root, weight in enumerate(r):
        if weight:
            rooted_dsts = enumerate_directed_spanning_trees(A, root, weight)
            for r_tree, cost in rooted_dsts:
                tree = - np.ones(n + 1)
                tree[root + 1] = 0
                for i, j, _ in r_tree:
                    tree[j + 1] = i + 1
                dsts.append((tree, cost))
    return dsts


def best_tree(W):
    """
    Return the best "unrooted" spanning tree and its cost of a weight matrix
    """
    trees = all_unrooted_spanning_trees(W)
    trees = sorted(trees, key=lambda x: x[1])
    return trees[0]


def best_rc_tree(W):
    """
    Return the best "rooted" spanning tree and its cost of a weight matrix
    """
    trees = all_rooted_spanning_trees(W)
    trees = sorted(trees, key=lambda x: x[1])
    return trees[0]


def kbest(W):
    trees = all_unrooted_spanning_trees(W)
    trees = sorted(trees, key=lambda x: x[1])
    return trees


def kbest_rc(W):
    trees = all_rooted_spanning_trees(W)
    trees = sorted(trees, key=lambda x: x[1])
    return trees
