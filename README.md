# Minimum Spanning Tree
This library contains a modified implementation of the Minimum Spanning Tree (MST) algorithm proposed by Tarjan (1977) and Camerini et al. (1979).
The MST algorithm is a popular decoding algorithm for graph-based depenedency parsing.
However, dependency trees often have a constraint that only one edge may emanate from the root.
Gabow and Tarjan (1984) suggest an efficient extension to the algorithm for this which is included in this library.

The algorithms above can be made to run in O(n^2) by using Radix sort.
The implementation of this library is O(n^2 log n), with the speed-up to O(n^2) being added soon.

This library also contains a simplified version of the K-Best MST algorithm
of Camerini et al. (1980) that runs in O(K * cost(MST)). Additionally, the
library contains a new algorithm for finding the K-Best dependency trees subject
to a root constraint.

A detailed description of these algorithms including proofs of correctness can be found in
["Mind the Root: Decoding Arborescences for Dependency Parsing"](https://www.aclweb.org/anthology/2020.emnlp-main.390/)
and
["On Finding the K-best Non-projective Dependency Trees"](https://arxiv.org/abs/2106.00780)
.

## Citation

This code is for the papers _Please Mind the Root: Decoding Arborescences for Dependency Parsing_ and
_On Finding the K-best Non-projective Dependency Trees_ featured in EMNLP 2020 and ACL 2021 respectively.
Please cite as:

```bibtex
@inproceedings{zmigrod-etal-2020-please,
    title = "Please Mind the Root: {D}ecoding Arborescences for Dependency Parsing",
    author = "Zmigrod, Ran  and
      Vieira, Tim  and
      Cotterell, Ryan",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.390",
    doi = "10.18653/v1/2020.emnlp-main.390",
    pages = "4809--4819",
}

@inproceedings{zmigrod-etal-2021-finding,
    title = "On Finding the K-best Non-projective Dependency Trees",
    author = "Zmigrod, Ran  and
      Vieira, Tim  and
      Cotterell, Ryan",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.106",
    doi = "10.18653/v1/2021.acl-long.106",
    pages = "1324--1337",
}
```

## Requirements and Installation

* Python version >= 3.6
* For running the tests, we use the library package [arsenal](https://github.com/timvieira/arsenal)

Installation:
```bash
git clone https://github.com/rycolab/spanningtree
cd spanningtree
pip install -e .
```


## Example
We support graph creations either using numpy arrays or dictionaries.
A dictionary graph `g` is such that `g[tgt][src]` contains a single or list of edges from `src` to `tgt`
We consider `0` to be the root node of the graph.
Note that the root node has no incoming edges and we do not have any self loops.
```python
import numpy as np
from spanningtrees.graph import Graph

G = Graph.build(
    np.array([
        [0, 17, 36, 21],
        [0,  0, 56, 42],
        [0, 86,  0,  9],
        [0, 23, 72,  0]
    ])
)

G = Graph.from_multigraph({
        1: {0: 17, 2: 86, 3: [23, 67]},
        2: {0: 36, 1: 56, 3: 72},
        3: {0: 21, 1: 42, 2: 9},
    })
```
The MST and root-constrained MST can then be calculated by:
```python
from spanningtrees.mst import MST

mst = MST(G).mst()
mst_constrained = MST(G, True).mst()
print(mst.to_array())
print(mst_constrained.to_array())
```
Output:
```
[-1  0  0  2]
[-1  3  0  2]
```

The following can be instead used to find the K-Best trees
```python
from spanningtrees.kbest import KBest

for tree in KBest(G).kbest():
    print(tree.to_array())
# Root constraint
for tree in KBest(G, True).kbest():
    print(tree.to_array())
```

## Related Work
This code repository focuses on decoding MSTs.
A useful library to use during training and learning of edges weights
can be found [here](https://github.com/rycolab/tree_expectations).

Other libraries for performing MST computations are [networkx](https://networkx.github.io/documentation/stable/index.html)
and [stanza](https://stanfordnlp.github.io/stanza/).
We will include a runtime comparison soon.
