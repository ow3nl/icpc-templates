"""Data Structures"""
import math


class Segment:
    
    # Maintains and updates a segment tree for range queries
    # Remember to modify the fill element!
    def __init__(self, A, n):
        self.original = A

        p = 1
        while p < n:
            p *= 2
        self.len = p
        self.array = [0 for _ in range(2 * self.len)]
        for i in range(self.len, 2 * self.len):
            if i - self.len >= n:
                self.array[i] = math.inf  # Remember to change this
            else:
                self.array[i] = A[i - self.len]
        
        for i in range(self.len - 1, 0, -1):
            self.array[i] = min(self.array[2 * i], self.array[2 * i + 1])
            # alternatively Segment[2 * i] + Segment[2 * i + 1]
            # or any other range query
        
    # Query the range(lo, hi) -- O(log(n))
    # Remember to modify the query!
    def query(self, lo, hi):
        lo += self.len
        hi += self.len
        m = math.inf  # remember to modify this too
        while lo <= hi:
            if lo % 2 == 1:
                m = min(m, self.array[lo])  # any other query here
                lo += 1
            if hi % 2 == 0:
                m = min(m, self.array[hi])
                hi -= 1
            lo //= 2
            hi //= 2

        return m

    # Add x to the i-th element of the original list
    # Remember to modify the query!
    def add(self, i, x):
        self.original[i] += x

        i += self.len
        self.array[i] += x
        while i >= 1:
            i //= 2
            # any other query
            self.array[i] = min(self.array[2*i], self.array[2*i + 1])


class Fenwick:
    
    # Prefix sum, with updates
    def __init__(self, A, n):
        self.original = A
        self.len = n
        prefix = [0]
        for x in A:
            prefix.append(prefix[-1] + x)
        
        self.array = [0 for _ in range(self.len+1)]
        for i in range(1, self.len+1):
            self.array[i] = prefix[i] - prefix[i - (i & -i)]
    
    # Gets sum of first i elements
    def prefix(self, i):
        s = 0
        while i >= 1:
            s += self.array[i]
            i -= i & -i
        return s

    # Adds x to the i-th value in the original list
    def add(self, i, x) -> None:
        self.original[i] += x
        i += 1
        while i <= self.len:
            self.array[i] += x
            i += i & -i


class DisjointSetUnion:
    
    # Maintains disjoint sets of integers 1 to n (technically 0 to n)
    def __init__(self, n):
        self.n = n
        self.link = [i for i in range(self.n+1)]
        self.size = [1 for _ in range(self.n+1)]
    
    # Find the representative of the set containing [elem]
    def find(self, elem):
        """Returns the top node in the tree containing elem"""
        if self.link[elem] == elem:
            return elem
        self.link[elem] = self.find(self.link[elem])
        return self.link[elem]

    # Check if [a] and [b] belong to the same set
    def same(self, a, b):
        """Check if [a] and [b] belong to the same tree"""
        return self.find(a) == self.find(b)

    # Join the sets containing [a] and [b]
    def unite(self, a, b):
        """Join the smaller tree to the larger tree"""
        a = self.find(a)
        b = self.find(b)
        if a != b:
            if self.size[a] < self.size[b]:
                a, b = b, a
            self.size[a] += self.size[b]
            self.link[b] = a


if __name__ == "__main__":
    A = [1, 5, 4, 5, 9, 12, 5, 2, 4, 73, 245, 2, 4, 5, 3]
    n = len(A)
    S = Fenwick(A, n)
