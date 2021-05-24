import numpy as np
from spanningtrees.heap import Heap, HeapMerge, MonotonicMapHeap
from spanningtrees.union_find import UnionFind
from spanningtrees.graph import Graph, Node, Edge


class ContractionGraph(UnionFind):
    """
    Efficient data structure for working with graphs that have had contraction
    operations performed on them.  The key to efficiency is the use the
    union-find data structure to efficiently indirect any node in a cluser of
    nodes to cluster representative.
    """
    def __init__(self, nodes):
        super().__init__(nodes)
        self.contractions = []
        self.n = len(nodes)
        self.c_id = self.n

    def __repr__(self):
        return '%s({\n%s\n})' % (self.__class__.__name__,
            '\n'.join(
                f'  {node}: {[(e.src, e.weight) for e in node.edges]}'
                for node in self if node.edges is not None
            )
        )

    def to_graph(self):
        """
        Create Graph object from ContractedGraph
        """
        graph = {}
        for node in self:
            if node == self[node]:
                graph[node] = {}
                for edge in self[node].edges:
                    if isinstance(node.name, int) or edge.src not in node.name:
                        if edge.src not in graph[node] or edge.weight <= graph[node][edge.src][0]:
                            graph[node][edge.src] = edge.weight, None
                if not graph[node]:
                    graph.pop(node)
        return Graph(graph)

    def scc(self, F):
        """
        Finds the strongly connected components in a forest of edges.
        """
        visited = set()
        cycles = []
        for w in range(1, self.n):
            w = self[w]
            if w in visited: continue
            visited.add(w)
            # Look for cycle
            cycles_nodes = {w}
            cycle = []
            e = F[w]
            found = False
            while e.src != 0:
                cycle.append(e)
                w = self[e.src]
                if w in cycles_nodes:
                    found = True
                    break
                elif w in visited:
                    break
                cycles_nodes.add(w)
                e = F[w]
            visited.update(cycles_nodes)
            if found:
                i = 0
                src = self[cycle[i].tgt]
                while src != w:
                    i += 1
                    src = self[cycle[i].tgt]
                cycles.append(cycle[i:])
        return cycles

    def find_cycle(self, F, e):
        """
        Find a cycle in a forest of edges F that includes edge e
        """
        # funny ordering is intentional because we are working backward
        # (traversing incoming edges)
        src = self[e.tgt]
        tgt = self[e.src]
        cycle = [e]
        while tgt != src:
            if tgt not in F: return
            e = F[tgt]
            cycle.append(e)
            tgt = self[e.src]
        return cycle

    def contract(self, cycle):
        """
        Create a contracted node in the graph given a cycle
        """
        nodes = [self[e.tgt] for e in cycle]
        c = ContractedNode(cycle, nodes, sum_cavities(cycle), self, self.c_id)
        self.c_id += 1
        self.contractions.append(c)
        return c

    def best_edge_no_self_loop(self, w):
        """
        Pick the best incoming edges that is not a self loop in the graph.
        Note that this method will pop edges from the node's edge queue.
        This might be slower than O(log(|w.edges|)) if lots of low-cost
        self loops are created during contraction ("dead edges").
        """
        w = self[w]
        if w.edges.empty():
            return None
        e = w.edges.pop()
        while self[e.src] is w and not w.edges.empty():   # pop until we stop getting self loops
            e = w.edges.pop()
        if self[e.src] is w:
            return None
        return e

    def best_edge_acyclic(self, tgt, tree):
        """
        Find the next best edge that does not create a directed cycle.
        """
        reachable = tree.reachable_from(tgt)
        w = tgt
        best = None
        undo = []
        while not w.edges.empty():
            e = w.edges.pop()           # XXX: Do we need to filter self loops?
            undo.append(e)              # ran: no, the tree.has_path should catch those anyway
            if e.src not in reachable:
                best = e
                break
        for f in undo:
            w.edges.push(f)
        return best

    def best_blue_weight(self, node, tree):
        edge = self.best_edge_acyclic(node, tree)
        old_edge = tree[node]
        if edge is None:
            return np.inf
        return edge.weight - old_edge.weight

    def best_edge_not_from(self, node, bad_src):
        """
        Find the next best edge that does not eminate from bad_src.
        """
        node = self[node]
        bad_src = self[bad_src]
        while not node.edges.empty():
            e = node.edges.peek()
            if self[e.src] != bad_src:
                return e
            node.edges.pop()

    def cheapest_swap(self, F, nodes, bad_src=0):
        """
        Find the cheapest swap that does not add a new root
        """
        # used to enforce root constraint
        weights = sum_cavities([F[tgt] for tgt in nodes])
        best_swap = (np.inf, None, None)
        for weight, tgt in zip(weights, nodes):
            if tgt.edges.empty(): continue
            # Find an an alternative edge f that is not from the a node.
            # Note that e != f because e was already popped from tgt's queue.
            f = self.best_edge_not_from(tgt, bad_src)
            swap = (weight + f.weight, (F[tgt], f), tgt)
            if swap < best_swap:
                best_swap = swap
        if best_swap[2] is not None:
            best_swap[2].edges.pop()
        return best_swap[1]


def offset(o):
    def f(e):
        e = e.copy()
        e.weight += o
        return e
    return f


def sum_cavities(edges):
    w = np.array([e.weight for e in edges])
    if 1:
        # If we have a subtraction operations, use it.
        return np.sum(w) - w

    else:
        # Fallback method for when subtraction is not viable
        n = len(w)
        A = np.cumsum(w)
        B = 0.0
        C = np.zeros(n)
        for i in reversed(range(1, n)):
            C[i] = A[i-1] + B   # requires associativity
            B += w[i]
        C[0] = B
        return C


#___________________________________________________________________________
# DISCUSSION: LAZY VS. EAGER CONTRACTION
#
# MSTs should not have self loops.  A self-loop edge could become the cheapest
# incoming edge when we greedily add it in the MST algorithm.  There are two
# approaches to preventing this (1) eagerly filter edges to prevent self loops,
# (2) lazily filter the best edges if they happen to be self loops.  The benefit
# of the lazy method (2) is that we only filter the "good" self loops (i.e.,
# "critical" self loops).  It also has the benefit that we do not have to
# preprocess the input graph to eagerly drop self loops.  The downside is that
# we have to pop several times from a priority queue.  If there are many self
# loop (which will happen on a heavily contracted graph), then we will pay a
# logarithmic factor in the size of the queue for each pop.  (If there have been
# M contractions and there are B edges, this is O(M log B); contrasting with
# eager pruning, which I believe is more like O(B).)
#
# There are even more benefits of laziness!  Here we are eagerly running all of
# the each edges.  Lazy merge would scale logarithmically with the number of
# merges M.  This avoid a potentially quadractic cost O(M*B), instead we have
# O(M log B).
#

HOWLAZY = 3


class ContractedNode(Node):
    """
    A contracted node represents a cycle in the graph.
    There are multiple ways to perform a contraction, see discussion on
    the different lazy approaches in the comments above"
    """
    def __init__(self, cycle, nodes, cavities, graph, node_id):
        name = frozenset(nodes)
        super().__init__(name, [], node_id)

        self.cycle = cycle
        self.nodes = nodes
        self.g_nodes = None

        # PERFORMING UNION HERE FOR ANNOYING REASONS.
        graph.add(self)
        for x in nodes: graph.union(self, x)

        if HOWLAZY == 0:
            edges = []
            for node, cavity in zip(nodes, cavities):
                for e in node.edges:
                    edges.append(Edge(e.src, e.tgt, e.weight + cavity))
            self.edges = Heap(edges)

        elif HOWLAZY == 1:
            edges = []
            for node, cavity in zip(nodes, cavities):
                for e in node.edges:
                    if graph[e.src] is not self:   # EAGER FILTERING OF SELF LOOPS
                        e.weight += cavity
                        edges.append(e)
            self.edges = Heap(edges)

        else:
            self.edges = HeapMerge([Heap([Edge(e.src, e.tgt, e.weight + o) for e in x.edges])
                                   for o, x in zip(cavities, self.nodes) if not x.edges.empty()])

    def expand(self, tgt):
        """
        Expand the cycle by a ground target node.  This operation essentially undoes
        the unification that created this cycle.
        """
        for x in self.nodes:
            if isinstance(x, ContractedNode):
                if tgt in x.ground:
                    return x
            else:
                if x == tgt:
                    return x
        assert False

    @property
    def ground(self):
        if self.g_nodes is None:
            self.g_nodes = set(self.ground_nodes())
        return self.g_nodes

    # Note: This is only used in expand
    def ground_nodes(self):
        """
        Return the ground nodes of the contracted graph
        """
        for x in self.nodes:
            if isinstance(x, ContractedNode):
                yield from x.ground_nodes()
            else:
                yield x

    def __repr__(self):
        return f'Cycle{set(self.nodes)}'
