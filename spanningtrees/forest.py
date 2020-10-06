import numpy as np


class Forest(dict):
    """
    A forest, is a collection of edges in which each node has
    at most one incoming edge.
    A forest is a generalization of a spanning tree.
    """
    def has_path(self, src, tgt):
        "Is there is (ground) path from src to tgt?"
        while tgt != src:
            if tgt not in self: return False
            e = self[tgt]
            tgt = e.src
        return True

    # Note: if we compute the reachability graph up front we can probably reduce
    # the cost substantially.  Basically, it is the SCC algorithm, which is
    # linear time instead of quadratic.
    def reachable_from(self, src):       # cost is linear per call
        children = set()
        for node in self:
            if self.has_path(src, node):
                children.add(node)
        return children

    def filter_src(self, node):
        "Set of nodes pointed to by `node`."
        return {tgt for tgt in self if self[tgt].src == node}

    def to_array(self):
        # Note: assumes nodes are numbered 0 to n.
        tree = -np.ones(len(self) + 1, dtype=np.int)
        for tgt, edge in self.items():
            tree[tgt.name] = edge.src
        return tree
