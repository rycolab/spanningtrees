"""
This code for unconstrained mst is adapted from the tarjan implementation of Miloš Stanojević
See original code at https://github.com/stanojevic/Fast-MST-Algorithm/blob/main/mst.py
"""
import numpy as np
from dataclasses import dataclass

from spanningtrees.edge_priority_queue import EdgePriorityQueue
from spanningtrees.brute_force import all_unrooted_spanning_trees, all_rooted_spanning_trees


class TreeError(LookupError):
    pass


@dataclass
class Edge:
    u: int
    v: int
    w: float

    def __eq__(self, other):
        return self.u == other.u and self.v == other.v and self.w == other.w


class MST:
    def __init__(self, graph):
        self.graph = graph.copy() # This version will not be changed
        self.N, _N = self.graph.shape
        assert self.N == _N, f"Graph must be given as square matrix. Got {self.N} x {_N} matrix instead."
        # Set incoming edges to root (at index 0) to -inf. This is needed instead of 0 as we support negative weights
        # Consider that we take weights to be log weights (thus allowing negative values)
        self.graph[:, 0] = np.nan
        # Set any self loops to nan.
        np.fill_diagonal(self.graph, np.nan)

    def weight(self, tree):
        return self.graph[tree[1:], np.arange(self.N)[1:]].sum()

    def _graph_init(self, include_set, exclude_set, root_constraint, reweight):
        # Initialise graph
        self._graph = np.copy(self.graph)
        # The reweight strategy applies the root constraint without alterations to the mst algorithm.
        # For more details, see https://github.com/stanojevic/Fast-MST-Algorithm/blob/main/mst.py
        if reweight:
            weights_no_inf = np.where(np.isfinite(self._graph), self._graph, np.nan)
            n = self.N - 1
            correction = n * (np.nanmax(weights_no_inf) - np.nanmin(weights_no_inf)) + 1
            self._graph[0] -= correction
            self._graph[0, 0] = np.nan
        for edge in exclude_set:
            self._graph[edge.u, edge.v] = - np.inf
        for edge in include_set:
            w = self._graph[edge.u, edge.v]
            self._graph[:, edge.v] = - np.inf
            if edge.u == 0 and root_constraint: # Check if we must also condition on root constraint
                self._graph[0, :] = - np.inf
            self._graph[edge.u, edge.v] = w
        if np.all(np.isnan(self._graph[0]) + ~np.isfinite(self._graph[0])):
            raise TreeError()

    def _init(self, include_set=None, exclude_set=None, root_constraint=False, reweight=False):
        if include_set is None:
            include_set = []
        if exclude_set is None:
            exclude_set = []
        self._graph_init(include_set, exclude_set, root_constraint, reweight)
        # We will have at most self.N - 1 contractions, and so we can have 2 * self.N - 1
        self._max_num_nodes = 2 * self.N - 1
        # Each node requires a priority queue of incoming edges
        self._queues = [EdgePriorityQueue(node, self._graph[:, node]) for node in range(self.N)]
        for c in range(self.N, self._max_num_nodes):
            self._queues.append(EdgePriorityQueue(c, np.ones(self.N)))
            self._queues[c].weights *= -np.inf
        self._incoming_nodes = [None for _ in range(self._max_num_nodes)]
        self._previous_nodes = - np.ones(self._max_num_nodes, dtype=int)
        self._children = [[] for _ in range(self._max_num_nodes)]
        self._ground_nodes = [[] for _ in range(self._max_num_nodes)]  # Transitive closure for children
        self._parent_nodes = np.arange(self._max_num_nodes)
        self._union_find = np.arange(self._max_num_nodes) # Transitive closure for parent_nodes
        self._num_nodes = self.N
        # Structure to keep best edges to enforce root constraint
        self._root_swap_costs = {}

    def _contract(self, cycle):
        # Contraction operations
        c = self._num_nodes
        self._num_nodes += 1
        cycle_weight = sum(self._incoming_nodes[j].w for j in cycle)
        for j in range(1, self._num_nodes):
            if self._union_find[j] in cycle:
                self._union_find[j] = c
        for i in cycle:
            self._children[c].append(i)
            if i < self.N:
                self._ground_nodes[c].append(i)
            else:
                self._ground_nodes[c].extend(self._ground_nodes[i])
            self._parent_nodes[i] = c
            self._queues[c].meld_inplace(self._queues[i], cycle_weight-self._incoming_nodes[i].w)
        return c

    def _find_cycle(self, b):
        # Cycle detection
        cycle = [b]
        node = self._previous_nodes[b]
        while node != b:
            if node == -1:  # Non cyclic path
                return []
            node = self._union_find[node]
            cycle.append(node)
            node = self._previous_nodes[node]
        return cycle

    def _compress(self):
        # Compression phase
        visited = {0}
        for a in range(1, self.N):
            while a not in visited:
                if self._queues[a].is_empty():
                    raise TreeError
                visited.add(a)
                u, v, w = self._queues[a].extract_max()
                self._incoming_nodes[a] = Edge(u, v, w)
                b = self._union_find[u]  # find super-node of source node
                self._previous_nodes[a] = b
                if b in visited:
                    # Check whether path is completed or cycle is formed
                    cycle = self._find_cycle(b)
                    if cycle:
                        a = self._contract(cycle)
                    else:
                        break
                else:  # Path extension
                    a = b

    def _expand(self):
        # Expansion phase
        dismanteled = set()
        for c in reversed(range(self.N, self._num_nodes)):
            if c in dismanteled:
                continue
            edge = self._incoming_nodes[c]
            self._incoming_nodes[edge.v] = edge
            p = self._parent_nodes[edge.v]
            while p != c:
                dismanteled.add(p)
                p = self._parent_nodes[p]

    def _find_root_nodes(self):
        # Find all nodes with edges emanating from the root
        root_nodes = []
        visited = set()
        for node in range(1, self._max_num_nodes):
            if self._incoming_nodes[node] is None:
                break
            parent = self._union_find[node]
            if parent in visited:
                continue
            visited.add(parent)
            u = self._incoming_nodes[parent].u
            if u == 0:
                root_nodes.append(parent)
        return root_nodes

    def _compute_swap_costs(self, root_nodes):
        # Update root swap bookkeeping structure
        for node in root_nodes:
            if node in self._root_swap_costs:
                continue
            w = self._incoming_nodes[node].w
            u_, v_, w_ = self._queues[node].preview_max()
            self._root_swap_costs[node] = w - w_, Edge(u_, v_, w_)

    def _best_root_swap(self, root_nodes):
        # Get best root swap
        self._compute_swap_costs(root_nodes)
        node, swap = min(self._root_swap_costs.items(), key=lambda x: x[1][0])
        self._root_swap_costs.pop(node)  # Remove swap from structure
        return node, swap

    def _resolve_root_constraint(self):
        # Enforce root constraint
        root_nodes = self._find_root_nodes()
        while len(root_nodes) > 1:
            swap_node, (swap_cost, edge) = self._best_root_swap(root_nodes)
            root_edge_weight = self._incoming_nodes[swap_node].w
            self._incoming_nodes[swap_node] = edge
            u, v = self._union_find[edge.u], self._union_find[edge.v]
            self._previous_nodes[v] = u
            cycle = self._find_cycle(u)
            if cycle:
                self._queues[swap_node].weights[0] = root_edge_weight
                while cycle:
                    c = self._contract(cycle)
                    u, v, w = self._queues[c].extract_max()
                    self._incoming_nodes[c] = Edge(u, v, w)
                    u = self._union_find[u]  # find
                    self._previous_nodes[c] = u
                    cycle = self._find_cycle(u)
            else:
                self._queues[swap_node].extract_max()
                self._queues[swap_node].weights[0] = - np.inf

            root_nodes = self._find_root_nodes()

    def _make_tree(self):
        # Use incoming_nodes to create array to represent tree
        tree = - np.ones(self.N, dtype=int)
        for j in range(1, self.N):
            i = self._incoming_nodes[j].u
            tree[j] = i
        return tree

    def mst(self, root_constraint=False, include_set=None, exclude_set=None, reweight=False):
        self._init(include_set, exclude_set, root_constraint, root_constraint and reweight)
        self._compress()
        if root_constraint and not reweight:
            self._resolve_root_constraint()
        self._expand()
        return self._make_tree()


def test_mst(seed, N, root_constraint):
    np.random.seed(seed)
    graph = np.random.uniform(0, 1, size=(N, N))
    mst = MST(graph)
    tree = mst.mst(root_constraint)
    weight = mst.weight(tree)
    dsts = all_rooted_spanning_trees(graph) if root_constraint else all_unrooted_spanning_trees(graph)
    dsts = list(sorted(dsts, key=lambda d: d[1], reverse=True))
    w_true = dsts[0][1]
    assert np.isclose(weight, w_true)


def test_reweight(seed, N):
    np.random.seed(seed)
    graph = np.random.uniform(0, 1, size=(N, N))
    mst = MST(graph)
    tree = mst.mst(True, reweight=True)
    true_tree = mst.mst(True)
    assert np.isclose(mst.weight(tree), mst.weight(true_tree))


def tests():
    from tqdm import tqdm

    print("\tTesting unconstrained mst")
    for seed in tqdm(range(10000)):
        test_mst(seed, 5, False)
    for seed in tqdm(range(1000)):
        test_mst(seed, 6, False)

    print("\tTesting constrained mst")
    for seed in tqdm(range(10000)):
        test_mst(seed, 5, True)
    for seed in tqdm(range(1000)):
        test_mst(seed, 6, True)

    print("\tTesting reweight constrained mst")
    for seed in tqdm(range(10000)):
        test_reweight(seed, 10)


if __name__ == '__main__':
    tests()
