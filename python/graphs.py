"""Graph Algorithms"""
import queue, math, heapq as hq
from data_structures import DisjointSetUnion

# TODO: https://ncduy0303.github.io/Competitive-Programming/
# TODO: test all functions!


class Graph:

    # Number of nodes and one-indexed adjacency list
    def __init__(self, n, Adj=None):
        self.n = n
        if Adj is None:
            self.Adj = [[] for _ in range(n+1)]
        else:
            self.Adj = Adj

    # List of edges (a, b)
    @property
    def edge_list(self):
        E = []
        for i, nodes in enumerate(self.Adj):
            for node in nodes:
                if node > i:
                    E.append((i, node))
        
        return E

    # Check if the graph is connected (req. dfs) -- O(V+E)
    @property
    def is_connected(self):
        visited = set()
        for node in self.dfs():
            visited.add(str(node))
        
        return len(visited) == self.n

    # Check if the graph is a tree (assumes connected) -- O(V)
    @property
    def is_tree(self):
        return sum(map(len, self.Adj)) // 2 + 1 == self.n

    # Check if Eulerian path exists (traverses every edge exactly once) -- O(V)
    @property
    def eulerian_path_exists(self):
        oddCount = 0
        for nodes in self.Adj:
            if len(nodes) % 2 == 1:
                oddCount += 1
        return oddCount == 0 or oddCount == 2
    
    # Check if Eulerian cycle exists (traverses every edge exactly once) -- O(V)
    @property
    def eulerian_cycle_exists(self):
        for nodes in self.Adj:
            if len(nodes) % 2 == 1:
                return False
        return True

    # Check if graph is bipartite (2-coloring exists, no odd cycles) -- O(V+E)
    @property
    def is_bipartite(self):
        root = 1
        visited = set()
        Q = queue.Queue()
        Q.put_nowait(root)

        Colored = [False for _ in range(self.n+1)]

        visited.add(str(root))

        while not Q.empty():
            prev = Q.get_nowait()
            for node in self.Adj[prev]:
                if str(node) not in visited:
                    visited.add(str(node))

                    Colored[node] = not Colored[prev]

                    Q.put_nowait(node)
                
                else:
                    if Colored[node] == Colored[prev]:
                        return False
        
        return True

    # Remove edge between [a] and [b] -- O(V)
    def remove_edge(self, a, b):
        if b in self.Adj[a]:
            self.Adj[a].remove(b)
            self.Adj[b].remove(a)

    # Add an edge between [a] and [b] -- O(1)
    def add_edge(self, a, b):
        self.Adj[a].append(b)
        self.Adj[b].append(a)

    # Generator, all nodes -- O(V+E)
    def dfs(self, root=1):
        stack = [root]
        visited = set()
        while stack != []:
            cur = stack.pop()
            if str(cur) not in visited:
                visited.add(str(cur))

                yield cur

                for node in self.Adj[cur]:
                    if str(node) not in visited:
                        stack.append(node)

    # Generator, all nodes with their parents -- O(V+E)
    def dfs_parents(self, root=1):
        stack = [(root, -1)]
        visited = set()
        while stack != []:
            cur, prev = stack.pop()
            if str(cur) not in visited:
                visited.add(str(cur))

                yield cur, prev

                for node in self.Adj[cur]:
                    if str(node) not in visited:
                        stack.append((node, cur))

    # Generator, all nodes including backtracked nodes -- O(V+E)
    def dfs_backtrack(self, root=1):
        stack = [(root, -2)]
        visited = set()
        while stack != []:
            cur, prev = stack.pop()

            if prev == -1:
                yield cur

            elif str(cur) not in visited:
                visited.add(str(cur))

                yield cur

                for node in self.Adj[cur]:
                    if str(node) not in visited:
                        stack.append((cur, -1))
                        stack.append((node, cur))

            else:
                stack.pop()

    # Generator, all nodes -- O(V+E)
    def bfs(self, root=1):
        visited = set()
        Q = queue.Queue()
        Q.put_nowait(root)

        visited.add(str(root))
        yield root

        while not Q.empty():
            prev = Q.get_nowait()
            for node in self.Adj[prev]:
                if str(node) not in visited:
                    visited.add(str(root))

                    yield node

                    Q.put_nowait(node)

    # Generator, all nodes with their parents -- O(V+E)
    def bfs_parents(self, root=1):
        visited = set()
        Q = queue.Queue()
        Q.put_nowait(root)

        visited.add(str(root))
        yield root, -1

        while not Q.empty():
            prev = Q.get_nowait()
            for node in self.Adj[prev]:
                if str(node) not in visited:
                    visited.add(str(node))

                    yield node, prev

                    Q.put_nowait(node)

    # Generator, yields tuples of nodes (BFS from a and b) -- O(V+E)
    def bfs_double(self, a, b):
        V1 = set()
        V1.add(str(a))
        Q1 = queue.Queue()
        Q1.put_nowait(a)
        
        V2 = set()
        V2.add(str(b))
        Q2 = queue.Queue()
        Q2.put_nowait(b)

        while (not Q1.empty()) and (not Q2.empty()):
            cur1 = Q1.get_nowait()
            for node in self.Adj[cur1]:
                if str(node) not in V1:
                    V1.add(str(node))
                    Q1.put_nowait(node)
            
            cur2 = Q2.get_nowait()
            for node in self.Adj[cur2]:
                if str(node) not in V2:
                    V2.add(str(node))
                    Q2.put_nowait(node)
            
            yield cur1, cur2

    # Any path from [root] to [target] (req. dfs_parents) -- O(V+E)
    def dfs_path(self, root, target):
        Parents = [-1 for _ in range(self.n + 1)]
        for cur, prev in self.dfs_parents(root):
            Parents[cur] = prev
            if cur == target:
                path = [cur]
                while cur != root:
                    cur = Parents[cur]
                    path.append(cur)
                path.reverse()
                return path

    # Shortest path from [root] to [target] (req. bfs_parents) -- O(V+E)
    def bfs_path(self, root, target):
        Parents = [-1 for _ in range(self.n+1)]
        for cur, prev in self.bfs_parents(root):
            Parents[cur] = prev

            if cur == target:
                path = [cur]
                while cur != root:
                    cur = Parents[cur]
                    path.append(cur)
                path.reverse()
                return path

    # Distances to all nodes (req. bfs_parents) -- O(V+E)
    def bfs_dists(self, root):
        D = [-1 for _ in range(self.n+1)]
        D[root] = 0
        for node, prev in self.bfs_parents(root):
            if node != root:
                D[node] = D[prev] + 1

        return D

    # Traverses all edges exactly once (Eulerian path must exist) -- O(V+E)
    def eulerian_path(self, root=None):
        E = list(map(set, self.Adj))  # List of edges in set format
        out_deg = list(map(len, self.Adj))  # outdegree of each node
        n_edges = sum(out_deg) // 2

        if root is None:
            for i in range(1, self.n+1):
                if out_deg[i] % 2 == 1:
                    root = i
                    break
            
            else:
                root = 1

        path = []
        start = []
        cur = root
        while len(path) < n_edges:
            if out_deg[cur] > 0:
                start.append(cur)

                temp = cur
                cur = E[cur].pop()
                E[cur].remove(temp)

                out_deg[temp] -= 1
                out_deg[cur] -= 1
            
            else:
                path.append(cur)
                cur = start.pop()
        
        path.append(root)
        path.reverse()
        return path

    # 2-coloring of graph, root is colored (req. bfs_parents) -- O(V+E)
    def bipartite_coloring(self, root=1):
        Coloring = [False for _ in range(self.n+1)]
        Coloring[root] = True
        for node, prev in self.bfs_parents(root):
            if node != root:
                Coloring[node] = not Coloring[prev]
        
        return Coloring

    # Tarjan's algorithm to find bridges -- O(V+E)
    def bridges(self):
        bridges = []

        discovery_times = [-1 for _ in range(self.n+1)]
        lowest_reachable = [math.inf for _ in range(self.n+1)]
        parents = [-1 for _ in range(self.n+1)]

        prev = -1
        time = 0
        stack = [(1, -2)]
        visited = set()
        while stack != []:
            cur, parent = stack.pop()
            
            if parent == -1:  # backtracked nodes
                lowest_reachable[cur] = min(lowest_reachable[cur],
                                            lowest_reachable[prev])
                if lowest_reachable[prev] > discovery_times[cur]:
                    bridges.append((cur, prev))

                prev = cur
            
            elif str(cur) not in visited:
                discovery_times[cur] = time
                time += 1
                visited.add(str(cur))
                parents[cur] = parent

                for node in self.Adj[cur]:
                    if node != parent:
                        if str(node) in visited:
                            lowest_reachable[cur] = min(lowest_reachable[cur],
                                                        discovery_times[node])
                        else:
                            parents[node] = cur
                            stack.append((cur, -1))    # traverse back to node
                            stack.append((node, cur))  # after traversing child
                
                prev = cur
            
            else:
                stack.pop()

        return bridges


class Tree:

    # Number of nodes and adjacency list
    def __init__(self, n, Adj=None):
        self.n = n
        if Adj is None:
            self.Adj = [[] for _ in range(n+1)]
        else:
            self.Adj = Adj

    # List of edges (a, b)
    @property
    def edge_list(self):
        E = []
        for i, nodes in enumerate(self.Adj):
            for node in nodes:
                if node > i:
                    E.append((i, node))
        
        return E

    # Maximum independent set, no two vertices adjacent -- O(V)
    @property
    def MIS(self):
        coloring = self.bipartite_coloring(1)[1:]
        colored = set()
        uncolored = set()
        for i, c in enumerate(coloring):
            if c:
                colored.add(i+1)
            else:
                uncolored.add(i+1)
        
        return colored if len(colored) > len(uncolored) else uncolored

    # Longest path in the tree, and end nodes (returns d, (s, e)) -- O(V)
    @property
    def diameter(self):
        D = self.bfs_dists(1)
        start = 0
        best = 0
        for i, x in enumerate(D):
            if x > best:
                start = i
                best = x

        D2 = self.bfs_dists(start)
        end = 0
        best = 0
        for i, x in enumerate(D2):
            if x > best:
                end = i
                best = x

        return best, (start, end)

    # Add an edge between [a] and [b] -- O(1)
    def add_edge(self, a, b):
        self.Adj[a].append(b)
        self.Adj[b].append(a)

    # Generator, traverses all nodes of the tree -- O(V+E)
    def dfs(self, root=1):
        stack = [(root, -1)]
        while stack != []:
            cur, prev = stack.pop()

            yield cur

            for node in self.Adj[cur]:
                if node != prev:
                    stack.append((node, cur))

    # Generator, all nodes with their parents -- O(V+E)
    def dfs_parents(self, root=1):
        stack = [(root, -1)]
        while stack != []:
            cur, prev = stack.pop()

            yield cur, prev

            for node in self.Adj[cur]:
                if node != prev:
                    stack.append((node, cur))

    # Generator, all nodes including backtracked nodes -- O(V+E)
    def dfs_backtrack(self, root=1):
        stack = [(root, -2)]
        while stack != []:
            cur, prev = stack.pop()

            yield cur

            if prev != -1:
                for node in self.Adj[cur]:
                    if node != prev:
                        stack.append((cur, -1))
                        stack.append((node, cur))

    # Generator, all nodes in subtree of [root] with parent [parent] -- O(V+E)
    def dfs_subtree(self, root=1, parent=-1):
        stack = [(root, parent)]
        while stack != []:
            cur, prev = stack.pop()

            yield cur

            for node in self.Adj[cur]:
                if node != prev:
                    stack.append((node, cur))

    # Generator, yields all nodes of the tree -- O(V+E)
    def bfs(self, root=1):
        Q = queue.Queue()
        Q.put_nowait((root, -1))

        yield root

        while not Q.empty():
            prev, ancestor = Q.get_nowait()
            for node in self.Adj[prev]:
                if node != ancestor:
                    yield node

                    Q.put_nowait((node, prev))

    # Generator, yields tuples of nodes (BFS from a and b) -- O(V+E)
    def bfs_double(self, a, b):
        Q1 = queue.Queue()
        Q1.put_nowait((a, -1))

        Q2 = queue.Queue()
        Q2.put_nowait((b, -1))

        while (not Q1.empty()) and (not Q2.empty()):
            cur1, anc1 = Q1.get_nowait()
            for node in self.Adj[cur1]:
                if node != anc1:
                    Q1.put_nowait((node, cur1))

            cur2, anc2 = Q2.get_nowait()
            for node in self.Adj[cur2]:
                if node != anc2:
                    Q2.put_nowait((node, cur2))
            
            yield cur1, cur2

    # Generator, all nodes in subtree of [root] with parent [parent] -- O(V+E)
    def bfs_subtree(self, root=1, parent=-1):
        Q = queue.Queue()
        Q.put_nowait((root, parent))

        yield root

        while not Q.empty():
            prev, ancestor = Q.get_nowait()
            for node in self.Adj[prev]:
                if node != ancestor:
                    yield node

                    Q.put_nowait((node, prev))

    # Generator, rows of nodes -- O(V+E)
    def bfs_rows(self, root=1, parent=-1):
        Q = queue.Queue()
        Q.put_nowait((root, parent))

        N = queue.Queue()

        yield [(root, parent)]

        while True:
            while not Q.empty():
                prev, ancestor = Q.get_nowait()
                for node in self.Adj[prev]:
                    if node != ancestor:
                        N.put_nowait((node, prev))

            if N.empty():
                break
            else:
                yield list(N.queue)

                Q = N
                N = queue.Queue()

    # Any path from [root] to [target] (req. dfs_parents) -- O(V+E)
    def dfs_path(self, root, target):
        Parents = [-1 for _ in range(self.n + 1)]
        for cur, prev in self.dfs_parents(root):
            Parents[cur] = prev
            
            if cur == target:
                path = [cur]
                while cur != root:
                    cur = Parents[cur]
                    path.append(cur)
                path.reverse()
                return path

        return []

    # Shortest path from [root] to [target] (req. bfs_parents) -- O(V)
    def bfs_path(self, root, target):
        Parents = [-1 for _ in range(self.n+1)]
        for node, parent in self.bfs_parents(root):
            Parents[node] = parent

            if node == target:
                path = [target]
                cur = target
                while cur != root:
                    cur = Parents[cur]
                    path.append(cur)

                path.reverse()
                return path

        return []

    # Distances to all nodes (req. bfs_parents) -- O(V)
    def bfs_dists(self, root=1):
        D = [-1 for _ in range(self.n+1)]
        
        for node, prev in self.bfs_parents(root):
            if node == root:
                D[node] = 0
            else:
                D[node] = D[prev] + 1

        return D

    # 2-coloring of graph, root is colored (req. bfs_parents) -- O(V)
    def bipartite_coloring(self, root=1):
        Coloring = [False for _ in range(self.n+1)]
        Coloring[root] = True
        
        for node, parent in self.bfs_parents(root):
            if node != root:
                Coloring[node] = not Coloring[parent]
        
        return Coloring

    # Number of nodes in the subtree from root (req. bfs_subtree) -- O(V)
    def subtree_count(self, root, parent):
        return len(self.bfs_subtree(root, parent))

    # TODO: with dfs_backtrack
    # Lowest common ancestor of [a] and [b], with respect to [root] -- O(V+E)
    def lca(self, a, b, root=1):
        H = self.bfs_dists(root)
        best_node = 0
        best_height = math.inf
        for node in self.bfs_path(a, b):
            if H[node] < best_height:
                best_height = H[node]
                best_node = node
        
        return best_node


class WeightedGraph:

    # Number of nodes and adjacency list of tuples (node, weight)
    def __init__(self, n, Adj=None):
        self.n = n
        if Adj is None:
            self.Adj = [[] for _ in range(n+1)]
        else:
            self.Adj = Adj

    # List of edges (weight, a, b)
    @property
    def edge_list(self):
        E = []
        for i, nodes in enumerate(self.Adj):
            for node, w in nodes:
                if node > i:
                    E.append((w, i, node))
        
        return E

    # TODO
    # Check if graph is connected
    @property
    def is_connected(self):
        pass

    # Check if the graph is a tree (assumes connected) -- O(V)
    @property
    def is_tree(self):
        return sum(map(len, self.Adj)) // 2 + 1 == self.n

    # Check if Eulerian path exists (traverses every edge exactly once) -- O(V)
    @property
    def eulerian_path_exists(self):
        oddCount = 0
        for nodes in self.Adj:
            if len(nodes) % 2 == 1:
                oddCount += 1
        return oddCount == 0 or oddCount == 2
    
    # Check if Eulerian cycle exists (traverses every edge exactly once) -- O(V)
    @property
    def eulerian_cycle_exists(self):
        for nodes in self.Adj:
            if len(nodes) % 2 == 1:
                return False
        return True

    # Check if graph is bipartite (2-coloring exists, no odd cycles) -- O(V+E)
    @property
    def bipartite(self):
        root = 1
        visited = set()
        Q = queue.Queue()
        Q.put_nowait(root)

        Colored = [False for _ in range(self.n+1)]

        visited.add(str(root))

        while not Q.empty():
            prev = Q.get_nowait()
            for node, _ in self.Adj[prev]:
                if str(node) not in visited:
                    visited.add(str(node))

                    Colored[node] = not Colored[prev]

                    Q.put_nowait(node)
                
                else:
                    if Colored[node] == Colored[prev]:
                        return False
        
        return True

    # Returns spanning tree with minimum weights (REQUIRES DSU) -- O(E log(E))
    @property
    def min_spanning_tree(self):
        D = DisjointSetUnion(self.n)
        A = [[] for _ in range(self.n+1)]
        for w, a, b in sorted(self.edge_list):
            if not D.same(a, b):
                A[a].append((b, w))
                A[b].append((a, w))
                D.unite(a, b)

        return A

    # Returns spanning tree with maximum weights (REQUIRES DSU) -- O(E log(E))
    @property
    def max_spanning_tree(self):
        D = DisjointSetUnion(self.n)
        A = [[] for _ in range(self.n+1)]
        for w, a, b in reversed(sorted(self.edge_list)):
            if not D.same(a, b):
                A[a].append((b, w))
                A[b].append((a, w))
                D.unite(a, b)

        return A

    # Remove edge between [a] and [b] -- O(V)
    def remove_edge(self, a, b):
        for i, (node, _) in enumerate(self.Adj[a]):
            if node == b:
                del self.Adj[a][i]
        
        for i, (node, _) in enumerate(self.Adj[b]):
            if node == a:
                del self.Adj[b][i]

    # Add an edge between [a] and [b] -- O(1)
    def add_edge(self, a, b, w):
        self.Adj[a].append((b, w))
        self.Adj[b].append((a, w))

    # Generator, yields (node) -- O(V+E)
    def dfs(self, root=1):
        stack = [root]
        visited = set()
        while stack != []:
            cur = stack.pop()
            if str(cur) not in visited:
                visited.add(str(cur))

                yield cur

                for node, _ in self.Adj[cur]:
                    if str(node) not in visited:
                        stack.append(node)

    # Generator, yields (node, parent, weight from parent) -- O(V+E)
    def dfs_parents(self, root=1):
        stack = [(root, -1, 0)]
        visited = set()
        while stack != []:
            cur, prev, w = stack.pop()
            if str(cur) not in visited:
                visited.add(str(cur))

                yield cur, prev, w

                for node, w in self.Adj[cur]:
                    if str(node) not in visited:
                        stack.append((node, cur, w))

    # Generator, yields (node) including all backtracked nodes -- O(V+E)
    def dfs_backtrack(self, root=1):
        stack = [(root, -2)]
        visited = set()
        while stack != []:
            cur, prev = stack.pop()
            
            if prev == -1:
                yield cur
            
            elif str(cur) not in visited:
                visited.add(str(cur))

                yield cur

                for node, _ in self.Adj[cur]:
                    if str(node) not in visited:
                        stack.append((cur, -1))
                        stack.append((node, cur))
            
            else:
                stack.pop()

    # Generator, yields (node) -- O(V+E)
    def bfs(self, root=1):
        visited = set()
        Q = queue.Queue()
        Q.put_nowait(root)

        visited.add(str(root))
        yield root

        while not Q.empty():
            prev = Q.get_nowait()
            for node, _ in self.Adj[prev]:
                if str(node) not in visited:
                    visited.add(str(node))

                    yield node

                    Q.put_nowait(node)

    # Generator, yields (node, parent, weight from parent) -- O(V+E)
    def bfs_parents(self, root=1):
        visited = set()
        Q = queue.Queue()
        Q.put_nowait(root)

        visited.add(str(root))
        yield root, -1

        while not Q.empty():
            prev = Q.get_nowait()
            for node, w in self.Adj[prev]:
                if str(node) not in visited:
                    visited.add(str(node))

                    yield node, prev, w

                    Q.put_nowait(node)

    # Generator, yields (node from [a], node from [b]) -- O(V+E)
    def bfs_double(self, a, b):
        V1 = set()
        V1.add(str(a))
        Q1 = queue.Queue()
        Q1.put_nowait(a)
        
        V2 = set()
        V2.add(str(b))
        Q2 = queue.Queue()
        Q2.put_nowait(b)

        while (not Q1.empty()) and (not Q2.empty()):
            cur1 = Q1.get_nowait()
            for node, _ in self.Adj[cur1]:
                if str(node) not in V1:
                    V1.add(str(node))
                    Q1.put_nowait(node)
            
            cur2 = Q2.get_nowait()
            for node, _ in self.Adj[cur2]:
                if str(node) not in V2:
                    V2.add(str(node))
                    Q2.put_nowait(node)
            
            yield cur1, cur2

    # Path with fewest steps from [root] to [target], and its distance -- O(V+E)
    def bfs_path(self, root, target):
        Parents = [(-1, 0) for _ in range(self.n+1)]
        
        for node, prev, w in self.bfs_parents(root):
            Parents[node] = (prev, w)

            if node == target:
                path = [target]
                cur = target
                d = 0
                while cur != root:
                    cur, w = Parents[cur]
                    path.append(cur)
                    d += w

                path.reverse()
                return path, d

    # Steps (not distances) to all nodes -- O(V+E)
    def bfs_dists(self, root):
        D = [-1 for _ in range(self.n+1)]
        D[root] = 0
        for node, prev, _ in self.bfs_parents(root):
            if node != root:
                D[node] = D[prev] + 1

        return D

    # Traverses all edges exactly once (Eulerian path must exist) -- O(V+E)
    def eulerian_path(self, root=None):
        # List of edges in set format
        E = list(map(set, [[x[0] for x in node] for node in self.Adj]))
        
        # outdegree of each node
        out_deg = list(map(len, self.Adj))

        n_edges = sum(out_deg) // 2

        if root is None:
            for i in range(1, self.n+1):
                if out_deg[i] % 2 == 1:
                    root = i
                    break
            
            else:
                root = 1

        path = []
        start = []
        cur = root
        while len(path) < n_edges:
            if out_deg[cur] > 0:
                start.append(cur)

                temp = cur
                cur = E[cur].pop()
                E[cur].remove(temp)

                out_deg[temp] -= 1
                out_deg[cur] -= 1
            
            else:
                path.append(cur)
                cur = start.pop()
        
        path.append(root)
        path.reverse()
        return path

    # 2-coloring of graph, root is colored (graph must be bipartite) -- O(V+E)
    def bipartite_coloring(self, root=1):
        Coloring = [False for _ in range(self.n+1)]
        for node, parent, _ in self.bfs_parents(root):
            if node == root:
                Coloring[root] = True
            else:
                Coloring[node] = not Coloring[parent]
        
        return Coloring

    # Shortest distance to all nodes (no negative weights) -- O(V)
    def djikstra(self, node):
        D = [math.inf for _ in range(self.n+1)]
        visited = set()
        D[node] = 0

        H = []
        hq.heapify(H)
        hq.heappush(H, (0, node))
        while H != []:
            _, cur = hq.heappop(H)
            if str(cur) not in visited:
                visited.add(str(cur))
                for node, w in self.Adj[cur]:
                    if str(node) not in visited:
                        if D[cur] + w < D[node]:
                            D[node] = D[cur] + w
                            hq.heappush(H, (D[node], node))

        return D

    # Shortest path from [root] to [target] (no negative weights) -- O(V)
    def djikstra_path(self, root, target):
        D = [math.inf for _ in range(self.n+1)]
        Parents = [-1 for _ in range(self.n+1)]
        visited = set()
        D[root] = 0

        H = []
        hq.heapify(H)
        hq.heappush(H, (0, root))
        while H != []:
            _, cur = hq.heappop(H)
            
            if cur == target:
                path = [cur]
                while cur != root:
                    cur = Parents[cur]
                    path.append(cur)
                path.reverse()
                return path

            if str(cur) not in visited:
                visited.add(str(cur))
                for node, w in self.Adj[cur]:
                    if str(node) not in visited:
                        if D[cur] + w < D[node]:
                            D[node] = D[cur] + w
                            Parents[node] = cur
                            hq.heappush(H, (D[node], node))
        
        return D


class DirectedGraph:
    
    def __init__(self, n, Adj=None):
        self.n = n
        if Adj is None:
            self.Adj = [[] for _ in range(n+1)]
        else:
            self.Adj = Adj

    # List of edges (a, b), edge from [a] to [b]
    @property
    def edge_list(self):
        E = []
        for i, nodes in enumerate(self.Adj):
            for node in nodes:
                E.append((i, node))

        return E

    # Add an edge from [a] to [b] -- O(1)
    def add_edge(self, a, b):
        self.Adj[a].append(b)

    # Remove an edge from [a] to [b] -- O(V)
    def remove_edge(self, a, b):
        self.Adj[a].remove(b)

    # Generator, dfs from root -- O(V+E)
    def dfs(self, root):
        stack = [root]
        visited = set()
        while stack != []:
            cur = stack.pop()
            if str(cur) not in visited:
                visited.add(str(cur))

                yield cur

                for node in self.Adj[cur]:
                    if str(node) not in visited:
                        stack.append(node)

    # Generator, all nodes with their parents -- O(V+E)
    def dfs_parents(self, root):
        stack = [(root, -1)]
        visited = set()
        while stack != []:
            cur, prev = stack.pop()
            if str(cur) not in visited:
                visited.add(str(cur))

                yield cur, prev

                for node in self.Adj[cur]:
                    if str(node) not in visited:
                        stack.append((node, cur))

    # Generator, all nodes including backtracked nodes -- O(V+E)
    def dfs_backtrack(self, root=1):
        stack = [(root, -2)]
        visited = set()
        while stack != []:
            cur, prev = stack.pop()
            
            if prev == -1:
                yield cur
            
            elif str(cur) not in visited:
                visited.add(str(cur))

                yield cur

                for node in self.Adj[cur]:
                    if str(node) not in visited:
                        stack.append((cur, -1))
                        stack.append((node, cur))
            
            else:
                stack.pop()

    # Generator, all nodes -- O(V+E)
    def bfs(self, root):
        visited = set()
        Q = queue.Queue()
        Q.put_nowait(root)

        visited.add(str(root))
        yield root

        while not Q.empty():
            prev = Q.get_nowait()
            for node in self.Adj[prev]:
                if str(node) not in visited:
                    visited.add(str(node))

                    yield node

                    Q.put_nowait(node)

    # Generator, all nodes with their parents -- O(V+E)
    def bfs_parents(self, root):
        visited = set()
        Q = queue.Queue()
        Q.put_nowait(root)

        visited.add(str(root))
        yield root, -1

        while not Q.empty():
            prev = Q.get_nowait()
            for node in self.Adj[prev]:
                if str(node) not in visited:
                    visited.add(str(node))

                    yield node, prev

                    Q.put_nowait(node)

    # Generator, yields tuples of nodes (BFS from a and b) -- O(V+E)
    def bfs_double(self, a, b):
        V1 = set()
        V1.add(str(a))
        Q1 = queue.Queue()
        Q1.put_nowait(a)
        
        V2 = set()
        V2.add(str(b))
        Q2 = queue.Queue()
        Q2.put_nowait(b)

        while (not Q1.empty()) and (not Q2.empty()):
            cur1 = Q1.get_nowait()
            for node in self.Adj[cur1]:
                if str(node) not in V1:
                    V1.add(str(node))
                    Q1.put_nowait(node)
            
            cur2 = Q2.get_nowait()
            for node in self.Adj[cur2]:
                if str(node) not in V2:
                    V2.add(str(node))
                    Q2.put_nowait(node)
            
            yield cur1, cur2

    # Topological sort (requires acyclic) -- O(V+E)
    def topological_sort(self):
        indegree = [0 for _ in range(self.n+1)]
        for node in range(self.n+1):
            for vertex in self.Adj[node]:
                indegree[vertex] += 1
    
        # Queue to store vertices with indegree 0
        q = queue.Queue()
        for i in range(1, self.n+1):
            if indegree[i] == 0:
                q.put_nowait(i)
        result = []
        while not q.empty():
            node = q.get_nowait()
            result.append(node)
            for adjacent in self.Adj[node]:
                indegree[adjacent] -= 1
                if indegree[adjacent] == 0:
                    q.put_nowait(adjacent)

        return result

    # Number of paths from [root] to each node (requires acyclic) -- O(V+E)
    def num_paths(self, root):
        paths = [0 for _ in range(self.n+1)]
        paths[root] = 1
        for node in self.topological_sort():
            for neighbor in self.Adj[node]:
                paths[neighbor] += paths[node]
        
        return paths

    # Tarjan's algorithm to find strongly connected components -- O(V+E)
    def scc(self):
        components = []

        discovery_times = [-1 for _ in range(self.n+1)]
        lowest_reachable = [-1 for _ in range(self.n+1)]

        prev = -1
        time = 0
        stack = []
        subtree_stack = []
        subtree = set()

        for node in range(1, self.n+1):
            if discovery_times[node] == -1:
                stack.append((int(node), -3))
                while stack != []:
                    cur, parent = stack.pop()

                    if parent == -1:  # backtracked nodes
                        lowest_reachable[cur] = min(lowest_reachable[cur],
                                                    lowest_reachable[prev])

                        prev = cur
                    
                    elif parent == -2:  # make check
                        if lowest_reachable[cur] == discovery_times[cur]:
                            comp = []
                            w = -1
                            while w != cur:
                                w = subtree_stack.pop()
                                subtree.remove(str(w))
                                comp.append(w)
                            
                            components.append(comp)

                        prev = cur

                    elif discovery_times[cur] == -1:
                        discovery_times[cur] = time
                        lowest_reachable[cur] = time
                        time += 1
                        subtree_stack.append(cur)
                        subtree.add(str(cur))
                        stack.append((cur, -2))

                        for node in self.Adj[cur]:
                            if node != parent:
                                if str(node) in subtree:
                                    lowest_reachable[cur] = min(
                                        lowest_reachable[cur],
                                        discovery_times[node]
                                    )
                                elif discovery_times[node] == -1:
                                    stack.append((cur, -1))
                                    stack.append((node, cur))

                        prev = cur

                    else:
                        stack.pop()

        return components


if __name__ == "__main__":
    # https://hideoushumpbackfreak.com/algorithms/images/strongly-connected.png
    G = DirectedGraph(11)
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(1, 10)
    G.add_edge(2, 3)
    G.add_edge(2, 11)
    G.add_edge(3, 5)
    G.add_edge(4, 10)
    G.add_edge(4, 11)
    G.add_edge(5, 1)
    G.add_edge(6, 4)
    G.add_edge(6, 9)
    G.add_edge(7, 11)
    G.add_edge(8, 6)
    G.add_edge(9, 1)
    G.add_edge(9, 5)
    G.add_edge(9, 8)
    G.add_edge(10, 7)
    G.add_edge(11, 10)
    print(G.scc())
