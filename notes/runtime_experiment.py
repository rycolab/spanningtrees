"""
This script executes some runtime experiments on random data.
It may also be of interest to run these experiments using trained output,
for which this script can be easily modified.

Running this script requires the timer of arsenal and the stanza library installed
"""

import numpy as np
from tqdm import tqdm
import pylab as pl

from arsenal.timer import timers
from stanza.models.common.chuliu_edmonds import chuliu_edmonds, chuliu_edmonds_one_root

from spanningtrees.mst import MST
from spanningtrees.kbest import KBest
from spanningtrees.kbest_camerini import KBestCamerini


def onebest_runtime():
    print("Testing random unconstrained one-best runtime")
    T = timers()
    for N in tqdm(range(5, 101, 5)):
        for _ in range(10):
            graph = np.random.uniform(0, 1, size=(N, N))
            np.fill_diagonal(graph, -np.inf)
            mst = MST(graph)
            weights = np.copy(graph).T
            with T["CLE"](n=N):
                tree_cle = chuliu_edmonds(weights)
            with T["MST"](n=N):
                tree = mst.mst()
            assert np.isclose(mst.weight(tree), mst.weight(tree_cle))
    T.compare()
    T.plot_feature("n")
    pl.savefig("notes/onebest_runtime.png")


def onebest_rc_runtime():
    print("Testing random constrained one-best runtime")
    T = timers()
    for N in tqdm(range(5, 101, 5)):
        for _ in range(10):
            graph = np.random.uniform(0, 1, size=(N, N))
            np.fill_diagonal(graph, -np.inf)
            mst = MST(graph)
            weights = np.copy(graph).T
            with T["CLE"](n=N):
                tree_cle = chuliu_edmonds_one_root(weights)
            with T["MST"](n=N):
                tree = mst.mst(True)
            with T["Reweight"](n=N):
                tree_reweight = mst.mst(True, reweight=True)
            assert np.isclose(mst.weight(tree), mst.weight(tree_cle))
            assert np.isclose(mst.weight(tree), mst.weight(tree_reweight))
    T.compare()
    T.plot_feature("n")
    pl.savefig("notes/onebest_rc_runtime.png")


def kbest_runtime(K):
    print(f"Testing random unconstrained kbest runtime with K={K}")
    T = timers()
    for N in tqdm(range(5, 101, 5)):
        for _ in range(5):
            graph = np.random.uniform(0, 1, size=(N, N))
            np.fill_diagonal(graph, -np.inf)
            kbest = KBest(graph)
            camerini = KBestCamerini(graph)
            trees = []
            trees_camerini = []
            with T["Kbest"](n=N):
                for i, tree in enumerate(kbest.kbest()):
                    trees.append(tree)
                    if i == K-1:
                        break
            with T["Camerini"](n=N):
                for i, tree in enumerate(camerini.kbest()):
                    trees_camerini.append(tree)
                    if i == K-1:
                        break
            for i in range(K):
                assert np.isclose(kbest.weight(trees[i]), kbest.weight(trees_camerini[i]))
    T.compare()
    T.plot_feature("n")
    pl.savefig(f"notes/kbest_{K}_runtime.png")


def kbest_rc_runtime(K):
    print(f"Testing random constrained kbest runtime with K={K}")
    T = timers()
    for N in tqdm(range(5, 101, 5)):
        for _ in range(5):
            graph = np.random.uniform(0, 1, size=(N, N))
            np.fill_diagonal(graph, -np.inf)
            kbest = KBest(graph)
            trees = []
            trees_reweight = []
            with T["K-Best"](n=N):
                for i, tree in enumerate(kbest.kbest(True)):
                    trees.append(tree)
                    if i == K-1:
                        break
            with T["Reweight"](n=N):
                for i, tree in enumerate(kbest.kbest(True, reweight=True)):
                    trees_reweight.append(tree)
                    if i == K-1:
                        break
            for i in range(K):
                if i < len(trees):
                    break
                assert np.isclose(kbest.weight(trees[i]), kbest.weight(trees_reweight[i]))
    T.compare()
    T.plot_feature("n")
    pl.savefig(f"notes/kbest_{K}_runtime.png")


if __name__ == '__main__':
    onebest_runtime()
    onebest_rc_runtime()
    for K in [10, 20, 50]:
        kbest_runtime(K)
        kbest_rc_runtime(K)
