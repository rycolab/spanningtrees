from collections import deque

from spanningtrees.forest import Forest
from spanningtrees.heap import Heap
from spanningtrees.mst import MST


class Swap:
    def __init__(self, weight, edge, included, excluded, F):
        self.weight = weight
        self.edge = edge
        self.included = included
        self.excluded = excluded
        self.F = F

    def __lt__(self, other):
        if other is None: return True
        return self.weight < other.weight

    def __repr__(self):
        return f'Swap{{{self.edge}}}w={self.weight}'


class KBest(MST):
    def kbest(self):
        tree = self.mst_scc()
        yield tree

        swaps = Heap([])

        def push(swap):
            if swap is None:
                return
            swaps.push(swap)

        push(self._next_swap_scc(tree, {}, {}))

        while not swaps.empty():
            swap = swaps.pop()
            edge = swap.edge
            excluded = set(swap.excluded)
            excluded.add((edge.src, edge.tgt))
            tree = self.mst_scc(swap.included, excluded)
            yield tree
            push(self._next_swap_scc(tree, swap.included, excluded))
            included = swap.included.copy()
            included[edge.tgt] = edge.src
            push(self._next_swap_scc(swap.F, included, excluded))

    def _next_swap_scc(self, tree, included, excluded):
        G = self._init(included, excluded)
        F = Forest()
        tree_weight = self.graph.weight(tree)

        best_swap = None

        nodes = deque(G)
        assert self.dummy_root == nodes.popleft()

        while nodes:
            while nodes:
                w = nodes.popleft()
                e = G.best_edge_no_self_loop(w)

                if tree[e.tgt].src == e.src:  # Only find swaps for edges in tree
                    f = self.best_edge_camerini(e.tgt, tree, G)
                    if f is not None:
                        swap = Swap(tree_weight + f.weight - e.weight, e,
                                    included, excluded, tree)
                        if swap < best_swap:
                            best_swap = swap

                F[w] = e

            cycles = G.scc(F)
            for cycle in cycles:
                c = G.contract(cycle)
                nodes.append(c)

        return best_swap

    def best_edge_camerini(self, tgt, tree, G):
        # Slightly different than function given in ContractedGraph
        reachable = tree.reachable_from(tgt)
        w = G[tgt]
        best = None
        undo = []
        while not w.edges.empty():
            e = w.edges.pop()
            undo.append(e)
            if e.src not in reachable:
                best = e
                break
        for f in undo:
            w.edges.push(f)
        return best
