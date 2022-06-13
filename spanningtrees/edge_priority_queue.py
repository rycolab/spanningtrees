import numpy as np


class EdgePriorityQueue:
    def __init__(self, node_id: int, edge_weights: np.ndarray, target=None):
        self.target = np.full(edge_weights.shape, node_id) if target is None else target
        self.weights = edge_weights
        if node_id < len(edge_weights):  # Avoid self-loops for ground nodes
            self.weights[node_id] = np.nan

    def __len__(self):
        return np.count_nonzero(np.logical_and(~np.isnan(self.weights), np.isfinite(self.weights)))

    def is_empty(self):
        return len(self) == 0

    def preview_max(self) -> (int, int, float):
        i = np.nanargmax(self.weights)
        if np.isnan(self.weights[i]):  # nanargmax bug with -inf
            i = np.argmax(np.isinf(self.weights))
        w = self.weights[i]
        return i, self.target[i], w

    def extract_max(self) -> (int, int, float):
        i, t, w = self.preview_max()
        self.weights[i] = np.nan
        return i, t, w

    def meld_inplace(self, other, shift) -> None:
        to_replace = (self.weights < other.weights + shift)
        self.target[to_replace] = other.target[to_replace]
        self.weights[to_replace] = other.weights[to_replace] + shift
        self.weights[np.isnan(other.weights)] = np.nan


# Priority Queue that contains the top two edges from each target node
class DoubleEdgePriorityQueue:
    def __init__(self, node_id: int, edge_weights: np.ndarray, target=None):
        self.target = np.full((len(edge_weights), 2), node_id) if target is None else target
        self.weights = np.zeros((len(edge_weights), 2))
        self.weights[:, 0] = edge_weights
        self.weights[:, 1] = -np.inf
        if node_id < len(edge_weights):  # Avoid self-loops for ground nodes
            self.weights[node_id] = np.nan

    def __len__(self):
        return np.count_nonzero(np.logical_and(~np.isnan(self.weights), np.isfinite(self.weights)))

    def is_empty(self):
        return len(self) == 0

    def preview_max(self) -> (int, int, float):
        i = np.nanargmax(self.weights[:, 0])
        w = self.weights[i, 0]
        return i, self.target[i, 0], w

    def extract_max(self) -> (int, int, float):
        i, t, w = self.preview_max()
        self.weights[i, 0] = self.weights[i, 1]
        self.weights[i, 1] = np.nan
        self.target[i, 0] = self.target[i, 1]
        return i, t, w

    def zero_out(self, i):
        self.weights[i, 0] = self.weights[i, 1]
        self.target[i, 0] = self.target[i, 1]
        self.weights[i, 1] = np.nan

    def meld_inplace(self, other, shift) -> None:
        # import ipdb; ipdb.set_trace()
        for i in range(len(self.weights)):
            if np.isnan(other.weights[i, 0]):
                self.weights[i] = np.nan
                continue
            if self.weights[i, 0] < other.weights[i, 0] + shift:
                self.weights[i, 1] = self.weights[i, 0]
                self.target[i, 1] = self.target[i, 0]
                self.weights[i, 0] = other.weights[i, 0] + shift
                self.target[i, 0] = other.target[i, 0]
                if not np.isnan(other.weights[i, 1]) and self.weights[i, 1] < other.weights[i, 1] + shift:
                    self.weights[i, 1] = other.weights[i, 1] + shift
                    self.target[i, 1] = other.target[i, 1]
            elif np.isnan(self.weights[i, 1]) or self.weights[i, 1] < other.weights[i, 0] + shift:
                self.weights[i, 1] = other.weights[i, 0] + shift
                self.target[i, 1] = other.target[i, 0]
