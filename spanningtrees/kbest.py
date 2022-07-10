import numpy as np
import heapq
from dataclasses import dataclass

from spanningtrees.mst import Edge, MST, TreeError
from spanningtrees.edge_priority_queue import DoubleEdgePriorityQueue
from spanningtrees.brute_force import all_unrooted_spanning_trees, all_rooted_spanning_trees


@dataclass
class KBestQueueElement:
    w: float
    e: Edge
    include_set: list
    exclude_set: list

    def __lt__(self, other):
        return self.w < other.w


class KBest(MST):

    def __init__(self, graph):
        super().__init__(graph)

    def _next_init(self, include_set=None, exclude_set=None, root_constraint=False):
        super()._init(include_set, exclude_set, root_constraint)
        # We need queues with the best 2 edges from each target to execute next
        self._queues = [DoubleEdgePriorityQueue(node, self._graph[:, node]) for node in range(self.N)]
        for c in range(self.N, self._max_num_nodes):
            self._queues.append(DoubleEdgePriorityQueue(c, -np.inf * np.ones(self.N)))

    def _best_reachable(self, node, reachable):
        # Find best reachable edge to node
        weights = self._queues[node].weights[:, 0]
        i, j, w = None, None, -np.inf
        for i_ in reachable:
            if not np.isnan(weights[i_]) and w < weights[i_]:
                i, j, w = i_, self._queues[node].target[i_, 0], weights[i_]
        return Edge(i, j, w)

    def _find_reachable(self, node, tree_nodes):
        # Find all ground nodes of tree_nodes that have a path from node
        visited = {0, node}
        desc = {node}
        for i in tree_nodes:
            path = set()
            while i not in visited:
                visited.add(i)
                path.add(i)
                i = self._incoming_nodes[i].u
                while i not in tree_nodes:
                    i = self._parent_nodes[i]
            if i in desc:
                desc.update(path)
        anc = tree_nodes.difference(desc)
        reachable = set()
        for i in anc:
            reachable.update({i} if i < self.N else self._ground_nodes[i])
        return reachable

    def _best_blue_weight(self, node, tree_nodes):
        # Find best edge from ancestor of node
        reachable = self._find_reachable(node, tree_nodes)
        edge = self._best_reachable(node, reachable)
        return self._incoming_nodes[node].w - edge.w

    def _next_expand(self):
        # Expansion phase to find second best tree
        nodes = [j for j, edge in enumerate(self._incoming_nodes) if edge is not None and self._union_find[j] == j]
        weight = sum(edge.w for j, edge in enumerate(self._incoming_nodes) if edge is not None and self._union_find[j] == j)
        best_edge, edge_cost = None,  -np.inf

        tree_nodes = set(nodes)
        tree_nodes.add(0)
        for node in nodes:
            cost = self._best_blue_weight(node, tree_nodes)
            if np.isfinite(cost) and weight - cost > edge_cost:
                best_edge, edge_cost = self._incoming_nodes[node], weight - cost
        dismanteled = set()
        for c in reversed(range(self.N, self._num_nodes)):
            if c in dismanteled:
                continue
            edge = self._incoming_nodes[c]
            self._incoming_nodes[c] = None
            self._incoming_nodes[edge.v] = edge
            self._queues[edge.v].zero_out(edge.u)
            new_nodes = set(self._children[c])
            tree_nodes.update(self._children[c])
            tree_nodes.remove(c)
            p = self._parent_nodes[edge.v]
            ps = set()
            while p != c:
                dismanteled.add(p)
                ps.add(p)
                for child in self._children[p]:
                    self._parent_nodes[child] = child
                    tree_nodes.add(child)
                    new_nodes.add(child)
                p = self._parent_nodes[p]
            new_nodes.difference_update(ps)
            tree_nodes.difference_update(ps)
            for node in self._children[c]:
                self._parent_nodes[node] = node
            for node in new_nodes:
                cost = self._best_blue_weight(node, tree_nodes)
                if np.isfinite(cost) and weight - cost > edge_cost:
                    best_edge, edge_cost = self._incoming_nodes[node], weight - cost
        return best_edge, edge_cost

    def next(self, include_set=None, exclude_set=None, root_constraint=False):
        self._next_init(include_set, exclude_set, root_constraint)
        self._compress()
        edge, w = self._next_expand()
        return self._make_tree(), (w, edge)

    def _kbest(self):
        # Unconstrained K-best algorithm
        tree, (w, edge) = self.next()
        yield tree
        if not np.isfinite(w):
            return
        queue = [KBestQueueElement(-w, edge, [], [])]
        while queue:
            item = heapq.heappop(queue)
            exclude_set_ = list(item.exclude_set)
            exclude_set_.append(item.e)
            tree, (w_, e_) = self.next(item.include_set, exclude_set_)
            yield tree
            if np.isfinite(-w_):
                heapq.heappush(queue, KBestQueueElement(-w_, e_, item.include_set, exclude_set_))
            include_set_ = list(item.include_set)
            include_set_.append(item.e)
            _, (w__, e__) = self.next(include_set_, item.exclude_set)
            if np.isfinite(-w__):
                heapq.heappush(queue, KBestQueueElement(-w__, e__, include_set_, item.exclude_set))

    def _root_edge(self, tree):
        # Assumes tree has only one root edge
        for j, i in enumerate(tree):
            if i == 0:
                return Edge(i, j, self.graph[i, j])

    def _kbest_rc(self):
        # Root constrained K-best algorithm
        tree = self.mst(True)
        yield tree
        e_root = self._root_edge(tree)
        queue = []
        try:
            tree_ = self.mst(True, [], [e_root])
            heapq.heappush(queue, KBestQueueElement(-self.weight(tree_), e_root, [], []))
        except TreeError:
            pass
        _, (w, edge) = self.next([e_root], [], True)
        if np.isfinite(w):
            heapq.heappush(queue, KBestQueueElement(-w, edge, [e_root], []))
        while queue:
            item = heapq.heappop(queue)
            if item.e.u != 0:
                exclude_set_ = list(item.exclude_set)
                exclude_set_.append(item.e)
                tree, (w_, e_) = self.next(item.include_set, exclude_set_, True)
                if np.isfinite(w_):
                    heapq.heappush(queue, KBestQueueElement(-w_, e_, item.include_set, exclude_set_))
                include_set_ = list(item.include_set)
                include_set_.append(item.e)
                _, (w__, e__) = self.next(include_set_, item.exclude_set, True)
                if np.isfinite(w__):
                    heapq.heappush(queue, KBestQueueElement(-w__, e__, include_set_, item.exclude_set))
            else:
                exclude_set_ = list(item.exclude_set)
                exclude_set_.append(item.e)
                tree = self.mst(True, item.include_set, exclude_set_)
                e_root = self._root_edge(tree)
                exclude_set_root = list(exclude_set_)
                exclude_set_root.append(e_root)
                try:
                    tree_ = self.mst(True, item.include_set, exclude_set_root)
                    heapq.heappush(queue, KBestQueueElement(-self.weight(tree_), e_root, item.include_set, exclude_set_))
                except TreeError:
                    pass
                include_set_ = list(item.include_set)
                include_set_.append(e_root)
                _, (w_, e_) = self.next(include_set_, exclude_set_, True)
                if np.isfinite(w_):
                    heapq.heappush(queue, KBestQueueElement(-w_, e_, include_set_, exclude_set_))
            yield tree

    def kbest(self, root_constraint=False, reweight=False):
        if root_constraint:
            if reweight:
                orig_graph = np.copy(self.graph)
                weights_no_inf = np.where(np.isfinite(self.graph), self.graph, np.nan)
                n = self.N - 1
                correction = n * (np.nanmax(weights_no_inf) - np.nanmin(weights_no_inf)) + 1
                self.graph[0] -= correction
                self.graph[0, 0] = np.nan
                yield from self._kbest()
                self.graph = orig_graph
            else:
                yield from self._kbest_rc()
        else:
            yield from self._kbest()


def test_next(seed, N):
    np.random.seed(seed)
    graph = np.random.uniform(0, 1, size=(N, N))
    kbest = KBest(graph)
    tree, (w, e) = kbest.next()
    second_best = kbest.mst(False, [], [e])
    assert np.isclose(kbest.weight(second_best), w)
    seconds = []
    for j in range(1, N):
        e_ = Edge(tree[j], j, graph[tree[j], j])
        second = kbest.mst(False, [], [e_])
        seconds.append((kbest.weight(second), e_, second))
    w_, e_, second = max(seconds, key=lambda x: x[0])
    assert np.isclose(w, w_)


def test_number_kbest(seed, N, root_constraint=False):
    np.random.seed(seed)
    graph = np.random.uniform(0, 1, size=(N, N))
    kbest = KBest(graph)
    dsts = all_rooted_spanning_trees(graph) if root_constraint else all_unrooted_spanning_trees(graph)
    dsts = list(sorted(dsts, key=lambda d: d[1], reverse=True))
    trees = list(tree for tree in kbest.kbest(root_constraint))
    assert len(trees) == len(dsts)
    for i in range(len(trees)):
        assert np.isclose(dsts[i][1], kbest.weight(trees[i]))


def test_graph(seed, N, K, root_constraint=False):
    np.random.seed(seed)
    graph = np.random.uniform(0, 1, size=(N, N))
    dsts = all_rooted_spanning_trees(graph) if root_constraint else all_unrooted_spanning_trees(graph)
    dsts = list(sorted(dsts, key=lambda d: d[1], reverse=True))
    kbest = KBest(graph)
    k = 0
    for tree in kbest.kbest(root_constraint):
        if k == K:
            break
        d_true, w_true = dsts[k]
        weight = kbest.weight(tree)
        assert np.isclose(weight, w_true)
        k += 1


def test_reweight(seed, N):
    np.random.seed(seed)
    graph = np.random.uniform(0, 1, size=(N, N))
    kbest = KBest(graph)
    kbest_orig = kbest.kbest(True, reweight=False)
    kbest_reweight = kbest.kbest(True, reweight=True)
    for i, (true_tree, tree) in enumerate(zip(kbest_orig, kbest_reweight)):
        true_weight = kbest.weight(true_tree)
        weight = kbest.weight(tree)
        assert np.isclose(true_weight, weight)
        if i == 50:
            break


def tests():
    from tqdm import tqdm

    print("Testing next")
    for seed in tqdm(range(10000)):
        test_next(seed, 6)
    for seed in tqdm(range(1000)):
        test_next(seed, 10)

    print("Testing unconstrained K-best")
    print("\tTesting all trees are returned")
    # Test all trees are returned
    for seed in tqdm(range(100)):
        test_number_kbest(seed, 5, False)
    for seed in tqdm(range(10)):
        test_number_kbest(seed, 6, False)

    print("\tTesting kbest")
    # Test many graphs
    for seed in tqdm(range(10000)):
        test_graph(seed, 5, 20, False)
    for seed in tqdm(range(1000)):
        test_graph(seed, 6, 20, False)

    print("Testing constrained K-best")
    print("\tTesting all trees are returned")
    # Test all trees are returned
    for seed in tqdm(range(100)):
        test_number_kbest(seed, 5, True)
    for seed in tqdm(range(10)):
        test_number_kbest(seed, 6, True)

    print("\tTesting kbest")
    # Test many graphs
    for seed in tqdm(range(10000)):
        test_graph(seed, 5, 20, True)
    for seed in tqdm(range(1000)):
        test_graph(seed, 6, 20, True)

    print("Testing reweight")
    for seed in tqdm(range(1000)):
        test_reweight(seed, 10)


if __name__ == '__main__':
    tests()
