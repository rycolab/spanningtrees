from collections import deque

from spanningtrees.graph import Graph, Node
from spanningtrees.forest import Forest
from spanningtrees.contract import ContractionGraph
from spanningtrees.union_find import UnionFind


class MST:
    """
    A class to run the MST algorithm over the root-constrained or unconstrained setting.
    The current runtime of the implementation is O(n^2 log n). This can be improved to
    O(n^2) through Radix sort and will be added to the implementation soon.
    """
    def __init__(self, graph, root_constraint=False, dummy_root=0):
        assert isinstance(graph, Graph)
        self.root_constraint = root_constraint
        self.dummy_root = dummy_root

        self.graph = graph.node_list

        # The set of possible roots (nodes with edges coming from dummy_root).
        self._roots = graph.target_nodes(self.dummy_root)

    def _init(self, included, excluded):
        """
        Initialization procedure to setup MST algorithm.
        We run this overhead to allow using included and excluded sets which
        enable k-best MST.
        The k-best algorithm will be released soon.
        """
        if included is None: included = {}
        if excluded is None: excluded = set()
        dummy_excl = set()
        if self.root_constraint:
            # In the root constrained setting, forcing an edge from dummy_root
            # implies no other edges leaving dummy_root should be considered.
            for tgt, src in included.items():
                if src == self.dummy_root:
                    for w in self._roots:
                        if w == tgt: continue
                        dummy_excl.add((self.dummy_root, w))

        nodes = [Node(self.dummy_root, [])]
        for tgt, edges in self.graph:
            es = []
            if tgt in included:
                src = included[tgt]
                for e in edges:
                    if e.src == src:
                        es.append(e)
                        break
            else:
                for e in edges:
                    if (e.src, tgt) in excluded: continue
                    if (e.src, tgt) in dummy_excl: continue
                    es.append(e)
            nodes.append(Node(tgt, es))
        return ContractionGraph(nodes)

    # ___________________________________________________________________________
    # DISCUSSION: LAZY VS. EAGER ADDING EDGES TO FOREST
    #
    # There are two approaches to getting the MST, one is to greedily find all the best edges, and then iteratively
    # resolve cycles through contraction. The other way is to add an edge and check if it introduces a cycle, and if so,
    # contract the graph. If we don't care about which cycle we contract in the former approach, then we get the same
    # runtime analysis as for the second approach. However, finding any cycle is much messier than finding a specific
    # cycle. Alternatively, we can find the SCC and pick the largest cycle in the first approach. Unless we do this
    # dynamically, this becomes quadratic and so does not provide a benefit. Therefore, we choose to check for cycles
    # after each edge addition to the forest.
    #
    # The algorithm presented in the paper (see README) is recursive and eagerly adds edges to the forest.
    # Therefore, Alg 1 in the paper is similar to mst_scc, however we still implement an iterative rather than
    # recursive approach for efficiency.
    #

    def mst_scc(self, included=None, excluded=None):
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
                F[w] = e

            cycles = G.scc(F)

            for cycle in cycles:
                c = G.contract(cycle)
                nodes.append(c)

        # Enforce degree constraints on dummy_root
        if self.root_constraint:
            self.resolve_root_constraint(G, F)

        F = self.stitch(G, F)

        return F

    def mst(self, included=None, excluded=None):
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

        check = UnionFind(G)

        while nodes:
            w = nodes.popleft()

            # pop until we stop getting self loops; also aids in handling the
            # degree constraint on the dummy_root.
            e = G.best_edge_no_self_loop(w)

            creates_cycle = check.union(w, e.src)

            F[w] = e

            # if adding edge `e` creates a "critical cycle," we will contract
            # that cycle (treat the nodes the cycle as a "meganode").
            # Subsequent steps work on that contracted graph.

            if creates_cycle:
                # Note that we can't just use the connected component to
                # identifyextract the cycle because it is undirected.  Thus, it gives
                # an overestimate of the set of nodes in the cycle.
                # Additionally, the data structure's invariants are so weak that
                # they do not support efficient connected component extraction
                # (we have to fall back to linear scan to do it).
                c = G.contract(G.find_cycle(F, e))
                check.add(c)
                check.union(c, w)
                nodes.append(c)

        # Enforce degree constraints on dummy_root
        # if we require a dependency tree
        if self.root_constraint:
            self.resolve_root_constraint(G, F)

        F = self.stitch(G, F)

        return F

    def stitch(self, G, F):
        """
        Stitch F together from the contractions in G.
        """
        # Expansion phase
        while G.contractions:
            c = G.contractions.pop()    # LIFO
            e = F.pop(c)
            tgt = c.expand(e.tgt)
            F[tgt] = e
        return F

    def swap(self, G, F, swap):
        """
        Apply swap in forest F and return any cycle that is formed by the swap
        """
        e, f = swap
        head = G[e.tgt]
        F[head] = f                        # Make swap

        # Todo: Patch union-find to support swapping partitions?
        cycle = G.find_cycle(F, f)
        G[f.tgt].edges.push(e)
        return cycle

    # Constrain(G) in paper
    def resolve_root_constraint(self, G, F):
        """Swap edges in F until it is a single-rooted tree."""
        roots = F.filter_src(self.dummy_root)  # Root nodes
        while len(roots) > 1:                  # while we have more than one root...

            swap = G.cheapest_swap(F, roots)   # Find the cheapest root edge to swap
            if swap is None: raise ValueError("Graph does not have MST with a single root.")
            e, f = swap  #; assert e.src == 0
            # Make swap
            cycle = self.swap(G, F, swap)
            roots.remove(G[e.tgt])              # Edge is no longer a root edge

            if cycle is not None:              # Contract cycle if one was made
                cycle = G.contract(cycle)
                F[cycle] = cycle.edges.pop()   # This will always be a root edge
                assert F[cycle].src == 0
                roots.add(cycle)               # ..so we have to add the new root
