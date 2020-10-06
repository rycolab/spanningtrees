from heapq import heapify, heappush, heappop, _siftup


class AbstractHeap:
    def pop(self):
        """
        Return and remove the best item in the heap
        """
        raise NotImplementedError

    def push(self, e):
        """
        Push element e onto the heap
        """
        raise NotImplementedError

    def peek(self):
        """
        Return the best item in the heap without removing it
        """
        raise NotImplementedError

    def empty(self):
        """
        Returns True if the heap is empty, false otherwise
        """
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __lt__(self, other):
        return self.peek() < other.peek()


class Heap(AbstractHeap):
    """
    Standard implementation of a heap using the heapq library
    """
    def __init__(self, xs):
        self.xs = xs
        if xs is not None:
            heapify(self.xs)

    def pop(self):
        return heappop(self.xs)

    def push(self, e):
        heappush(self.xs, e)

    def peek(self):
        return self.xs[0]

    def empty(self):
        return len(self.xs) == 0

    def __iter__(self):
        # WARNING: heaps do not iterate in sorted order.
        return iter(self.xs)


# Note: this class has some constant-factor inefficiencies (1) when we acutally
# use it we call the function f and it copies the edge. -- it would be better if
# we could lazily mutate the edge weight, but this is tricky because some
# functions will push the edge back on some heap.
class MonotonicMapHeap(AbstractHeap):
    "Lazily apply a monotonic function over the heap."

    def __init__(self, heap, f):
        self.heap = heap
        self.f = f
        # Use caching to reduce the number of calls to f, which is very helpful
        # in the event of frequent `peek`ing (e.g., heaps merge).
        self.peek_cache = self._peek()

    def __iter__(self):
        for e in self.heap:
            yield self.f(e)

    def empty(self):
        return self.heap.empty()

    def peek(self):
        return self.peek_cache

    def _peek(self):
        if self.heap.empty(): return None
        return self.f(self.heap.peek())

    def pop(self):
        x = self.peek_cache
        self.heap.pop()
        self.peek_cache = self._peek()
        return x

    def push(self, e):
        raise NotImplementedError


class HeapMerge(Heap):
    "Lazy merging of heaps"

    def peek(self):
        return super().peek().peek()

    def pop(self):
        s = super().peek()
        e = s.pop()                   # advance the top heap
        if s.empty():
            super().pop()             # remove empty heap
        else:
            _siftup(self.xs, 0)       # restore heap condition
        return e

    def push(self, e):
        super().push(Heap([e]))

    def __iter__(self):
        for h in self.xs:
            yield from h
