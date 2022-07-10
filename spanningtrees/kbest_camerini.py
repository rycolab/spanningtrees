import numpy as np
import heapq
from dataclasses import dataclass

from spanningtrees.mst import Edge, MST, TreeError
from spanningtrees.kbest import KBestQueueElement
from spanningtrees.edge_priority_queue import DoubleEdgePriorityQueue
from spanningtrees.brute_force import all_unrooted_spanning_trees


@dataclass
class KBestQueueElement:
    w: float
    e: Edge
    tree: np.array
    include_set: list
    exclude_set: list

    def __lt__(self, other):
        return self.w < other.w


class KBestCamerini(MST):

    def __init__(self, graph):
        super().__init__(graph)

    def _next_init(self, include_set=None, exclude_set=None, root_constraint=False):
        super()._init(include_set, exclude_set, root_constraint)
        self._queues = [DoubleEdgePriorityQueue(node, self._graph[:, node]) for node in range(self.N)]
        for c in range(self.N, self._max_num_nodes):
            self._queues.append(DoubleEdgePriorityQueue(c, -np.inf * np.ones(self.N)))

    def _best_reachable(self, node, reachable):
        weights = self._queues[node].weights[:, 0]
        i, j, w = None, None, -np.inf
        for i_ in reachable:
            if not np.isnan(weights[i_]) and w < weights[i_]:
                i, j, w = i_, self._queues[node].target[i_, 0], weights[i_]
        return Edge(i, j, w)

    def _best_blue_weight(self, node, tree):
        visited = {0, node}
        desc = {node}
        for i in range(1, self.N):
            path = set()
            while i not in visited:
                visited.add(i)
                path.add(i)
                i = tree[i]
            if i in desc:
                desc.update(path)
        anc = set(range(self.N)).difference(desc)
        node = self._union_find[node]
        edge = self._best_reachable(node, anc)
        return self._incoming_nodes[node].w - edge.w

    def next(self, tree, include_set=None, exclude_set=None, root_constraint=False):
        self._next_init(include_set, exclude_set, root_constraint)
        best_edge, edge_cost = None, -np.inf
        weight = self.weight(tree)

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
                if tree[v] == u: # Find best swap if edge is in tree
                    cost = self._best_blue_weight(v, tree)
                    if np.isfinite(cost) and weight - cost > edge_cost:
                        best_edge, edge_cost = self._incoming_nodes[a], weight - cost
                if b in visited:
                    # Check whether path is completed or cycle is formed
                    cycle = self._find_cycle(b)
                    if cycle:
                        a = self._contract(cycle)
                    else:
                        break
                else:  # Path extension
                    a = b
        return edge_cost, best_edge

    def kbest(self):
        tree = self.mst()
        yield tree
        w, edge = self.next(tree)
        if not np.isfinite(w):
            return
        queue = [KBestQueueElement(-w, edge, tree, [], [])]
        while queue:
            item = heapq.heappop(queue)
            exclude_set_ = list(item.exclude_set)
            exclude_set_.append(item.e)
            tree = self.mst(False, item.include_set, exclude_set_)
            yield tree
            w_, e_ = self.next(tree, item.include_set, exclude_set_)
            if np.isfinite(-w_):
                heapq.heappush(queue, KBestQueueElement(-w_, e_, tree, item.include_set, exclude_set_))
            include_set_ = list(item.include_set)
            include_set_.append(item.e)
            w__, e__ = self.next(item.tree, include_set_, item.exclude_set)
            if np.isfinite(-w__):
                heapq.heappush(queue, KBestQueueElement(-w__, e__, item.tree, include_set_, item.exclude_set))


def test_next(seed, N):
    np.random.seed(seed)
    graph = np.random.uniform(0, 1, size=(N, N))
    kbest = KBestCamerini(graph)
    tree = kbest.mst(False)
    w, e = kbest.next(tree)
    second_best = kbest.mst(False, [], [e])
    assert np.isclose(kbest.weight(second_best), w)
    seconds = []
    for j in range(1, N):
        e_ = Edge(tree[j], j, graph[tree[j], j])
        second = kbest.mst(False, [], [e_])
        seconds.append((kbest.weight(second), e_, second))
    w_, e_, second = max(seconds, key=lambda x: x[0])
    assert np.isclose(w, w_)


def test_number_kbest(seed, N):
    np.random.seed(seed)
    graph = np.random.uniform(0, 1, size=(N, N))
    kbest = KBestCamerini(graph)
    dsts = all_unrooted_spanning_trees(graph)
    dsts = list(sorted(dsts, key=lambda d: d[1], reverse=True))
    trees = list(tree for tree in kbest.kbest())
    assert len(trees) == len(dsts)
    for i in range(len(trees)):
        assert np.isclose(dsts[i][1], kbest.weight(trees[i]))


def test_graph(seed, N, K):
    np.random.seed(seed)
    graph = np.random.uniform(0, 1, size=(N, N))
    dsts = all_unrooted_spanning_trees(graph)
    dsts = list(sorted(dsts, key=lambda d: d[1], reverse=True))
    kbest = KBestCamerini(graph)
    k = 0
    for tree in kbest.kbest():
        if k == K:
            break
        d_true, w_true = dsts[k]
        weight = kbest.weight(tree)
        assert np.isclose(weight, w_true)
        k += 1


def tests():
    from tqdm import tqdm

    print("Testing next")
    for seed in tqdm(range(10000)):
        test_next(seed, 6)
    for seed in tqdm(range(1000)):
        test_next(seed, 10)

    print("Testing unconstrained K-best")
    print("\tTesting all trees are returned from K-best")
    # Test all trees are returned
    for seed in tqdm(range(100)):
        test_number_kbest(seed, 5)
    for seed in tqdm(range(10)):
        test_number_kbest(seed, 6)

    print("\tTesting K-best")
    # Test many graphs
    for seed in tqdm(range(10000)):
        test_graph(seed, 5, 20)
    for seed in tqdm(range(1000)):
        test_graph(seed, 6, 20)


if __name__ == '__main__':
    tests()
