import numpy as np
from collections import deque

from spanningtrees.heap import Heap
from spanningtrees.mst import MST
from spanningtrees.forest import Forest


class Swap:
    def __init__(self, weight, edge, included, excluded):
        self.weight = weight
        self.edge = edge
        self.included = included
        self.excluded = excluded

    def __lt__(self, other):
        if other is None: return True
        return self.weight < other.weight


class KBest(MST):
    """
    A class to run the K-best MST algorithm over the root-constrained or unconstrained setting.
    Unlike the MST class, we purely use the SCC algorithms here
    """

    def extract_next(self, G, F):
        # Expansion phase
        nodes = [node for node in G if G[node] == node and node != self.dummy_root]
        tree = Forest()
        best_node, edge_cost = None, np.inf
        weight = 0
        for node in nodes:
            tree[node] = F[node]
            weight += tree[node].weight
        while nodes:
            node = nodes.pop()
            cost = G.best_blue_weight(node, tree)
            if best_node is None or cost < edge_cost:
                best_node, edge_cost = node, cost
        while G.contractions:
            c = G.contractions.pop()    # LIFO
            e = tree.pop(c)
            tgt = c.expand(e.tgt)
            tree[tgt] = e
            for node in c.nodes:
                if node == tgt:
                    continue
                tree[node] = F[node]
            for node in c.nodes:
                if node == tgt:
                    if best_node == c:
                        best_node = node
                    continue
                cost = G.best_blue_weight(node, tree)
                if best_node is None or cost < edge_cost:
                    best_node, edge_cost = node, cost
        return tree, best_node, weight + edge_cost

    def next(self, included=None, excluded=None):
        """
        Find the minimum spanning tree subject to included/excluded constraints
        Throws an IndexError if no MST exists (subject to those constraints).
        """
        G = self._init(included, excluded)

        F = Forest()
        nodes = deque(G)
        assert self.dummy_root == nodes.popleft()

        # Contraction phase: Each edge greedily pick's its favorite edge.
        # Whenever doing so induces a cycle, we call that cycle "critical".  In
        # this phase, all critical cycles are contracted as they are
        # encountered.  At all points, the contracted graph is acyclic.

        while nodes:
            while nodes:
                w = nodes.popleft()
                e = G.best_edge_no_self_loop(w)
                if e is None:
                    return None, None, None
                F[w] = e

            cycles = G.scc(F)

            for cycle in cycles:
                c = G.contract(cycle)
                nodes.append(c)

        return self.extract_next(G, F)

    def _kbest(self):
        swaps = Heap([])

        tree, node, weight = self.next()
        yield tree
        if node is not None and np.isfinite(weight):
            swaps.push(Swap(weight, tree[node], {}, set()))

        while not swaps.empty():
            swap = swaps.pop()
            edge, included, excluded = swap.edge, swap.included, swap.excluded
            new_excluded = set(excluded)
            new_excluded.add((edge.src, edge.tgt))
            tree, node, weight = self.next(included, new_excluded)
            yield tree
            if node is not None and np.isfinite(weight):
                swaps.push(Swap(weight, tree[node], included, new_excluded))
            new_included = included.copy()
            new_included[edge.tgt] = edge.src
            tree, node, weight = self.next(new_included, excluded)
            if node is not None and np.isfinite(weight):
                swaps.push(Swap(weight, tree[node], new_included, excluded))

    def _kbest_rc(self):
        swaps = Heap([])

        tree = self.mst()
        yield tree

        e_root = tree.filter_src(self.dummy_root).pop()
        _, node, weight = self.next({e_root: self.dummy_root})
        if node is not None and np.isfinite(weight):
            swaps.push(Swap(weight, tree[node], {e_root: self.dummy_root}, set()))
        tree1 = self.mst_scc({}, {(self.dummy_root, e_root)})
        if tree1 is not None:
            weight = self.graph.weight(tree1.to_array())
            swaps.push(Swap(weight, tree[e_root], {}, {}))

        while not swaps.empty():
            swap = swaps.pop()
            edge, included, excluded = swap.edge, swap.included, swap.excluded
            new_excluded = set(excluded)
            new_excluded.add((edge.src, edge.tgt))
            if edge.src == self.dummy_root:
                tree = self.mst_scc(included, new_excluded)
                yield tree
                e_root = tree.filter_src(self.dummy_root).pop()
                new_included = included.copy()
                new_included[e_root] = self.dummy_root
                _, node, weight = self.next(new_included, new_excluded)
                if node is not None and np.isfinite(weight):
                    swaps.push(Swap(weight, tree[node], new_included, new_excluded))
                new_excluded_root = set(new_excluded)
                new_excluded_root.add((self.dummy_root, e_root))
                tree1 = self.mst_scc(included, new_excluded_root)
                if tree1 is not None:
                    weight = self.graph.weight(tree1.to_array())
                    swaps.push(Swap(weight, tree[e_root], included, new_excluded))
            else:
                tree, node, weight = self.next(included, new_excluded)
                yield tree
                if node is not None and np.isfinite(weight):
                    swaps.push(Swap(weight, tree[node], included, new_excluded))
                new_included = included.copy()
                new_included[edge.tgt] = edge.src
                tree, node, weight = self.next(new_included, excluded)
                if node is not None and np.isfinite(weight):
                    swaps.push(Swap(weight, tree[node], new_included, excluded))

    def kbest(self):
        if self.root_constraint:
            yield from self._kbest_rc()
        else:
            yield from self._kbest()
