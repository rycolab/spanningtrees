from spanningtrees.mst import MST
from spanningtrees.graph import Edge, Graph
from arsenal import ok


def test_contraction():
    """
    Test that graph contraction operation on example graph yields correct contracted graph.
    """
    # Check the contraction operations from Fig1-Fig2 of Georgiadis 2003.

    # Leonidas Georgiadis, (2003) "Arborescence optimization problems solvable
    # by Edmonds’ algorithm."  Theoretical Computer Science.

    G = Graph.from_multigraph({
        2: {1: 60, 6: 70},
        3: {1: 50, 2: 70, 8: 20},
        4: {2: 80, 3: 50},
        5: {4: 20},
        6: {2: 60, 5: 65},
        7: {5: 40, 6: 75},
        8: {5: 90, 7: 10},
        9: {8: 55, 3: 90},
    })

    C = frozenset({(3, 4), (4, 5), (5, 7), (7, 8), (8, 3)})

    g = MST(G)._init(None, None)

    C = frozenset({Edge(i, j, G.w(i, j)) for i, j in C})

    g.contract(C)

    # Check that this equals
    G1 = Graph.from_multigraph({
        2: {1: 60, 6: 70},
        6: {2: 60, 5: 65},
        9: {8: 55, 3: 90},
        frozenset({3, 4, 5, 7, 8}): {
            1: 170,
            2: 170,  # have to be careful that we take the min of the two incoming edges from 2.
            6: 175,
        },
    })

    G2 = g.to_graph()
    assert G1 == G2
    print(ok)


if __name__ == '__main__':
    test_contraction()
