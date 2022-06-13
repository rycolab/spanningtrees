import numpy as np


def _enum_dst(graph, cost, included, rest, excluded):
    n = len(graph)
    if len(included) == n:
        return [(rest, cost)]
    dsts = []
    new_excluded = list(excluded)
    for i in included:
        for j in range(n):
            cost_ij = graph[i, j]
            if j not in included and (i, j) not in excluded and np.isfinite(cost_ij):
                new_excluded += [(i, j)]
                dsts += _enum_dst(graph, cost + cost_ij, included + [j],
                                 rest + [(i, j, cost_ij)], new_excluded)
    return dsts


def all_unrooted_spanning_trees(graph):
    n = len(graph)

    graph = graph.copy()
    np.fill_diagonal(graph, -np.inf)
    graph[:, 0] = -np.inf
    dsts = []
    unrooted_dsts = []
    for i in range(n):
        unrooted_dsts += _enum_dst(graph, 0, [i], [], [])
    for tree, cost in unrooted_dsts:
        t = - np.ones(n)
        for i, j, _ in tree:
            t[j] = i
        dsts.append((t, cost))
    return dsts


def all_rooted_spanning_trees(graph):
    graph = graph.copy()
    root_weights = graph[0, 1:]
    graph = graph[1:, 1:]
    n = len(graph)
    np.fill_diagonal(graph, 0)
    dsts = []
    for root, weight in enumerate(root_weights):
        if np.isfinite(weight):
            rooted_dsts = _enum_dst(graph, weight, [root], [], [])
            for r_tree, cost in rooted_dsts:
                tree = - np.ones(n + 1)
                tree[root + 1] = 0
                for i, j, _ in r_tree:
                    tree[j + 1] = i + 1
                dsts += [(tree, cost)]
    return dsts