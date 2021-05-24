import numpy as np
from spanningtrees.heap import Heap


class Edge(object):
    """
    An Edge is between an ordered pair of nodes (src and tgt) and
    has an associated weight.
    It may optionally also have a label (in the case of multigraphs)
    """
    __slots__ = 'src', 'tgt', 'weight', 'label'

    def __init__(self, src, tgt, weight, label=None):
        self.src = src
        self.tgt = tgt
        self.weight = weight
        self.label = label

    def copy(self):
        return Edge(self.src, self.tgt, self.weight, self.label)

    def __lt__(self, other):
        return self.weight < other.weight

    def __repr__(self):
        return f'{self.src}→ {self.tgt}'


class Node(object):
    """
    Represents the state of a node in the MST algorithm

    - a heap of incoming edge preferences.

    """
    __slots__ = 'name', 'edges', 'id'

    def __init__(self, name, edges, node_id):
        self.name = name
        self.edges = Heap(edges)
        self.id = node_id

    def __eq__(self, other):
        return self is other or self.name == other

    def __hash__(self):
        return self.id

    def __repr__(self):
        return f'{self.name}'


class Graph(dict):

    def __repr__(self):
        return 'Graph({\n%s\n})' % (
            '\n'.join(
                f'  {node}: {[(src, w[0] if isinstance(w, tuple) else w.weight) for (src, w) in self[node].items()]}'
                for node in self
            )
        )

    def target_nodes(self, src):
        """Get the set of nodes pointed to by `src`, this method expensive O(|V|)."""
        return {tgt for tgt in self
                if tgt != src
                if src in self[tgt]}

    def w(self, src, tgt):
        "Lookup edge weight"
        return self[tgt][src][0] if src in self[tgt] else 0.0

    def weight(self, tree):
        if isinstance(tree, np.ndarray):
            return sum(self.w(src, tgt + 1)
                       for tgt, src in enumerate(tree[1:]))
        else:  # tree is a dict
            return sum(self.w(e.src, e.tgt) for e in tree.values())

    @classmethod
    def build(cls, graph):
        """
        Build a graph from a numpy array. We assume that 0 represents the dummy root
        """
        G = {}
        n,m = graph.shape
        assert n == m
        ninf = -np.inf
        for tgt in range(1, n):
            G[tgt] = {}
            # dummy_root given as 0 in numpy array
            for src in range(n):
                if tgt == src or graph[src, tgt] == ninf:
                    continue
                G[tgt][src] = graph[src, tgt], None
        return cls(G)

    def node_list(self):
        """
        Create representation of graph as a list of (node, incomming_edges).
        This is needed for the MST algorithm
        """
        return [(tgt, [Edge(src, tgt, self[tgt][src][0]) for src in self[tgt]])
                for tgt in self]

    @classmethod
    def from_multigraph(cls, graph):
        """"
        Create a graph from a multigraph. We consider a multigraph to be represented as a dict
        where graph[tgt][src] is a list of all edges from src to tgt. We only need to take the
        best scoring (minimum weight) edge in order to compute the MST.
        """
        G = {}
        for tgt in graph:
            G[tgt] = {}
            for src in graph[tgt]:
                w = graph[tgt][src]
                if isinstance(w, list):
                    if isinstance(w, tuple):
                        w = min(w, key=lambda x: x[0])
                    else:
                        w = min(w)
                if isinstance(w, tuple):
                    cost, label = w
                else:
                    cost, label = w, None
                G[tgt][src] = cost, label
        return cls(G)
