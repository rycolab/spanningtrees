import numpy as np
from spanningtrees.contract import ContractedNode


class Forest(dict):
    """
    A forest, is a collection of edges in which each node has
    at most one incoming edge.
    A forest is a generalization of a spanning tree.
    """
    def reachable_from(self, src):
        reach = src.ground if isinstance(src, ContractedNode) else {src}
        visited = set(reach)
        visited.add(0)
        for tgt in self:
            if tgt == src:
                continue
            path = set()
            while tgt not in visited:
                if tgt not in self:
                    for node_ in self:
                        if isinstance(node_, ContractedNode) and tgt in node_.ground:
                            tgt = node_
                            path.update(node_.ground)
                            break
                    if tgt not in self:
                        break
                elif isinstance(tgt, ContractedNode):
                    path.update(tgt.ground)
                else:
                    path.add(tgt)
                visited.add(tgt)
                edge = self[tgt]
                tgt = edge.src
            if tgt in reach:
                reach.update(path)
        return reach

    def filter_src(self, node):
        "Set of nodes pointed to by `node`."
        return {tgt for tgt in self if self[tgt].src == node}

    def to_array(self):
        # Note: assumes nodes are numbered 0 to n.
        tree = -np.ones(len(self) + 1, dtype=np.int)
        for tgt, edge in self.items():
            tree[tgt.name] = edge.src
        return tree
