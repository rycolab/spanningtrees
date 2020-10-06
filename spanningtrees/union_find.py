"""
Union-find disjoint set data structure.
"""


class UnionFind:
    """Union-find datastructure for disjoint sets.

    This implementation uses an impure merge heuristic: left-term wins.
    """

    def __init__(self, elements=None):
        self._par = {}  # parent: for the internal tree structure
        if elements is not None:
            for x in elements:
                self.add(x)

    def __repr__(self):
        return  f'UnionFind(components={self.components()})'

    def __len__(self):
        return len(self._par)

    def __contains__(self, x):
        return x in self._par

    def __iter__(self):
        return iter(self._par)

    def add(self, x):
        "Add a single disjoint element."
        if x not in self._par:
            self._par[x] = x

    # The usual union-find algorithm does not support "stable" cluster
    # representatives.
    def __getitem__(self, x):
        "Find the root of the disjoint set containing the given element."
        while True:
            # path compression
            p = self._par[x]
            self._par[x] = p
            if x is p: return x
            x = p

    def connected(self, x, y):
        """
        [True/False] x and y are belong to the same component.
        """
        return self[x] is self[y]

    def union(self, x, y):
        """
        Union the sets represented by x and y, always using x as the root node (*);
        returns None if nodes were previously disjoint, true otherwise.
        """
        x = self[x]; y = self[y]
        if x is y: return True
        self._par[y] = x

    def component(self, x):
        "Find the connected component containing the given element."
        return {y for y in self if self[y] is self[x]}

    def components(self):
        "Return a mapping from components to their elements."
        roots = {}
        for y in self:
            r = self[y]
            if r not in roots: roots[r] = []
            roots[r].append(y)
        return roots


def test():
    d = UnionFind()
    for x in range(10):
        d.add(x % 3)
        d.add(x)
        d.union(x % 3, x)
    print(d.components())
    assert d.components() == {0: [0, 3, 6, 9], 1: [1, 4, 7], 2: [2, 5, 8]}

    d = UnionFind()
    for x in range(10):
        d.add(x % 3)
        d.add(x)
        d.union(x, x % 3)
    print(d.components())

    print(d)
    assert set(d) == set(range(10))
    assert d.components() == {9: [0, 3, 6, 9], 7: [1, 4, 7], 8: [2, 5, 8]}


if __name__ == '__main__':
    test()
