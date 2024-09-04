#include <bits/stdc++.h>

using namespace std;

#define vi vector<int>
#define vb vector<bool>
#define vs vector<string>
#define vvi vector<vi>
#define ii pair<int, int>
#define vii vector<ii>
#define vvii vector<vii>
#define ll long long
#define vll vector<ll>
#define vvll vector<vll>
#define ld double
#define qi queue<int>
#define qii queue<ii>
#define pqi priority_queue<int>
#define pqii priority_queue<ii>

const int INF = 1e9 + 1;

class Graph {
public:
	int n;
	vvi adj;

	Graph(int n) {
		this->n = n;
		adj = vvi(n, vi(0));
	}

	Graph(int n, vvi adj) {
		this->n = n;
		this->adj = adj;
	}

	void add_edge(int a, int b) {
		adj[a].push_back(b);
		adj[b].push_back(a);
	}

	void dfs(int node, vb& V) {
		V[node] = true;
		cout << node << endl;  // DO SOMETHING AT EACH NODE
		for (int neighbor: adj[node]) {
			if (!V[neighbor]) {
				dfs(neighbor, V);
			}
		}
	}

	void iter_dfs(int root = 0) {
		vb V(n, false);
		vi S(1);
		S[0] = root;
		
		while (!S.empty()) {
			int node = S.back();
			S.pop_back();
			V[node] = true;
			cout << node << endl;  // DO SOMETHING AT EACH NODE
			for (int neighbor: adj[node]) {
				if (!V[neighbor]) {
					S.push_back(neighbor);
				}
			}
		}
	}

	void bfs(int root = 0) {
		vb V(n, false);
		deque<int> Q;
		Q.push_back(root);

		while (!Q.empty()) {
			int node = Q.front();
			Q.pop_front();
			V[node] = true;
			cout << node << endl;  // DO SOMETHING AT EACH NODE
			for (int neighbor: adj[node]) {
				if (!V[neighbor]) {
					Q.push_back(neighbor);
				}
			}
		}
	}

	// BFS to find parents
	vi parents(int root = 0) {
		vi P(n, -1);

		vb V(n, false);
		deque<int> Q;
		Q.push_back(root);

		while (!Q.empty()) {
			int node = Q.front();
			Q.pop_front();
			V[node] = true;
			for (int neighbor: adj[node]) {
				if (!V[neighbor]) {
					P[neighbor] = node;
					Q.push_back(neighbor);
				}
			}
		}

		return P;
	}

	// BFS to find distance to each node
	vi dists(int root = 0) {
		vi D(n, -1);
		D[root] = 0;

		vb V(n, false);
		deque<int> Q;
		Q.push_back(root);

		while (!Q.empty()) {
			int node = Q.front();
			Q.pop_front();
			V[node] = true;
			for (int neighbor: adj[node]) {
				if (!V[neighbor]) {
					D[neighbor] = D[node] + 1;
					Q.push_back(neighbor);
				}
			}
		}

		return D;
	}

	void bridges_inner(int node, vb& V, vi& D, vi& L, int parent, vii& ans) {
		// V: visited
		// D: discovery_times
		// L: lowest_reachable
		static int time = 0;

		V[node] = true;
		D[node] = L[node] = ++time;

		for (int neighbor: adj[node]) {
			if (neighbor == parent) {
				continue;
			} else if (V[neighbor]) {
				L[node] = min(L[node], D[neighbor]);
			} else {
				bridges_inner(neighbor, V, D, L, node, ans);
				L[node] = min(L[node], L[neighbor]);
				if (L[neighbor] > D[node]) {
					ans.push_back({node, neighbor});
				}
			}
		}
	}

	// Tarjan's bridge-finding algorithm
	vii bridges() {
		vb V(n, false);
		vi D(n, -1);
		vi L(n, -1);

		vii ans(0);

		for (int i = 0; i < n; i++) {
			if (!V[i]) {
				bridges_inner(i, V, D, L, -1, ans);
			}
		}

		return ans;
	}
};

class Tree {
public:
	int n;
	vvi adj;

	Tree(int n) {
		this->n = n;
		adj = vvi(n, vi(0));
	}

	Tree(int n, vvi adj) {
		this->n = n;
		this->adj = adj;
	}

	void add_edge(int a, int b) {
		adj[a].push_back(b);
		adj[b].push_back(a);
	}

	void dfs(int node = 0, int parent = -1) {
		cout << node << endl;  // DO SOMETHING AT EACH NODE
		for (int neighbor: adj[node]) {
			if (neighbor != parent) {
				dfs(neighbor, node);
			}
		}
	}

	void iter_dfs(int root = 0, int parent = -1) {
		vii S(1);
		S[0] = {root, parent};

		while (!S.empty()) {
			ii item = S.back();
			S.pop_back();
			cout << item.first << endl;  // DO SOMETHING AT EACH NODE
			for (int neighbor: adj[item.first]) {
				if (neighbor != item.second) {
					S.push_back({neighbor, item.first});
				}
			}
		}
	}

	void bfs(int root = 0, int parent = -1) {
		deque<ii> Q;
		Q.push_back({root, parent});

		while (!Q.empty()) {
			ii item = Q.front();
			Q.pop_front();
			cout << item.first << endl;  // DO SOMETHING AT EACH NODE
			for (int neighbor: adj[item.first]) {
				if (neighbor != item.second) {
					Q.push_back({neighbor, item.first});
				}
			}
		}
	}

	void bfs_rows(int root = 0, int parent = -1) {
		deque<ii> Q, N;
		Q.push_back({root, parent});

		while (!Q.empty()) {
			ii item = Q.front();
			Q.pop_front();
			for (int neighbor: adj[item.first]) {
				if (neighbor != item.second) {
					N.push_back({neighbor, item.first});
				}
			}

			Q = N;
			N = deque<ii>();

			for (deque<ii>::iterator it = Q.begin(); it != Q.end(); it++) {
				cout << (*it).first << ' ';
			}
			cout << endl;
		}
	}

	// BFS to find parents
	vi parents(int root = 0, int parent = -1) {
		vi P(n, -1);

		vii S(1);
		S[0] = {root, parent};

		while (!S.empty()) {
			ii item = S.back();
			P[item.first] = item.second;
			S.pop_back();
			for (int neighbor: adj[item.first]) {
				if (neighbor != item.second) {
					S.push_back({neighbor, item.first});
				}
			}
		}

		return P;
	}

	// BFS to find height of each node
	vi heights(int root = 0, int parent = -1) {
		vi H(n, -1);

		deque<ii> Q;
		Q.push_back({root, parent});

		while (!Q.empty()) {
			ii item = Q.front();
			Q.pop_front();
			H[item.first] = (item.first == root) ? 0 : H[item.second] + 1;
			for (int neighbor: adj[item.first]) {
				if (neighbor != item.second) {
					Q.push_back({neighbor, item.first});
				}
			}
		}

		return H;
	}

	// TODO: LCA, subtree stuff
};

class WeightedGraph {
public:
	int n;
	vvii adj;

	WeightedGraph(int n) {
		this->n = n;
		adj = vvii(n, vii(0));
	}

	WeightedGraph(int n, vvii adj) {
		this->n = n;
		this->adj = adj;
	}

	void add_edge(int a, int b, int w) {
		adj[a].push_back({b, w});
		adj[b].push_back({a, w});
	}

	void dfs(int node, vb& V) {
		cout << node << endl;
		V[node] = true;
		for (ii neighbor: adj[node]) {
			if (!V[neighbor.first]) {
				dfs(neighbor.first, V);
			}
		}
	}

	void iter_dfs(int root = 0) {
		vb V(n, false);
		vi S(1);
		S[0] = root;
		
		while (!S.empty()) {
			int node = S.back();
			S.pop_back();
			V[node] = true;
			cout << node << endl;  // DO SOMETHING AT EACH NODE
			for (ii neighbor: adj[node]) {
				if (!V[neighbor.first]) {
					S.push_back(neighbor.first);
				}
			}
		}
	}

	void bfs(int root = 0, int parent = -1) {
		vb V(n, false);
		deque<int> Q;
		Q.push_back(root);

		while (!Q.empty()) {
			int node = Q.front();
			Q.pop_front();
			V[node] = true;
			cout << node << endl;  // DO SOMETHING AT EACH NODE
			for (ii neighbor: adj[node]) {
				if (!V[neighbor.first]) {
					Q.push_back(neighbor.first);
				}
			}
		}
	}

	void bridges_inner(int node, vb& V, vi& D, vi& L, int parent, vii& ans) {
		// V: visited
		// D: discovery_times
		// L: lowest_reachable
		static int time = 0;

		V[node] = true;
		D[node] = L[node] = ++time;

		for (ii neighbor: adj[node]) {
			if (neighbor.first == parent) {
				continue;
			} else if (V[neighbor.first]) {
				L[node] = min(L[node], D[neighbor.first]);
			} else {
				bridges_inner(neighbor.first, V, D, L, node, ans);
				L[node] = min(L[node], L[neighbor.first]);
				if (L[neighbor.first] > D[node]) {
					ans.push_back({node, neighbor.first});
				}
			}
		}
	}

	// Tarjan's bridge-finding algorithm
	vii bridges() {
		vb V(n, false);
		vi D(n, -1);
		vi L(n, -1);

		vii ans(0);

		for (int i = 0; i < n; i++) {
			if (!V[i]) {
				bridges_inner(i, V, D, L, -1, ans);
			}
		}

		return ans;
	}

	vi djikstra(int root = 0) {
		vi D(n, INF);
		vb V(n, false);
		D[root] = 0;

		pqii H;
		H.push({0, root});
		while (!H.empty()) {
			int node = H.top().second;
			H.pop();

			if (!V[node]) {
				for (ii edge: adj[node]) {
					if (D[node] + edge.second < D[edge.first]) {
						D[edge.first] = D[node] + edge.second;
						H.push({-D[edge.first], edge.first});
					}
				}
			}
		}

		return D;
	}
};

class DirectedGraph {
public:
	int n;
	vvi adj;

	DirectedGraph(int n) {
		this->n = n;
		adj = vvi(n, vi(0));
	}

	DirectedGraph(int n, vvi adj) {
		this->n = n;
		this->adj = adj;
	}

	void add_edge(int a, int b) {
		adj[a].push_back(b);
	}

	void dfs(int node, vb& V) {
		V[node] = true;
		cout << node << endl;  // DO SOMETHING AT EACH NODE
		for (int neighbor: adj[node]) {
			if (!V[neighbor]) {
				dfs(neighbor, V);
			}
		}
	}

	void iter_dfs(int root = 0) {
		vb V(n, false);
		vi S(1);
		S[0] = root;
		
		while (!S.empty()) {
			int node = S.back();
			S.pop_back();
			V[node] = true;
			cout << node << endl;  // DO SOMETHING AT EACH NODE
			for (int neighbor: adj[node]) {
				if (!V[neighbor]) {
					S.push_back(neighbor);
				}
			}
		}
	}

	void bfs(int root = 0) {
		vb V(n, false);
		deque<int> Q;
		Q.push_back(root);

		while (!Q.empty()) {
			int node = Q.front();
			Q.pop_front();
			V[node] = true;
			cout << node << endl;  // DO SOMETHING AT EACH NODE
			for (int neighbor: adj[node]) {
				if (!V[neighbor]) {
					Q.push_back(neighbor);
				}
			}
		}
	}

	// BFS to find parents
	vi parents(int root = 0) {
		vi P(n, -1);

		vb V(n, false);
		deque<int> Q;
		Q.push_back(root);

		while (!Q.empty()) {
			int node = Q.front();
			Q.pop_front();
			V[node] = true;
			for (int neighbor: adj[node]) {
				if (!V[neighbor]) {
					P[neighbor] = node;
					Q.push_back(neighbor);
				}
			}
		}

		return P;
	}

	// BFS to find distance to each node
	vi dists(int root = 0) {
		vi D(n, -1);
		D[root] = 0;

		vb V(n, false);
		deque<int> Q;
		Q.push_back(root);

		while (!Q.empty()) {
			int node = Q.front();
			Q.pop_front();
			V[node] = true;
			for (int neighbor: adj[node]) {
				if (!V[neighbor]) {
					D[neighbor] = D[node] + 1;
					Q.push_back(neighbor);
				}
			}
		}

		return D;
	}

	vi topo_sort() {
		vi indegree(n, 0);
		for (int node = 0; node < n; node++) {
			for (int vertex: adj[node]) {
				indegree[vertex]++;
			}
		}

		qi q;
		for (int node = 0; node < n; node++) {
			if (indegree[node] == 0) {
				q.push(node);
			}
		}

		vi ans(n, 0);
		int cur = 0;
		while (!q.empty()) {
			int node = q.front();
			q.pop();

			ans[cur] = node;
			cur++;

			for (int neighbor: adj[node]) {
				indegree[neighbor]--;
				if (indegree[neighbor] == 0) {
					q.push(neighbor);
				}
			}
		}

		return ans;
	}

	void scc_inner(int node, vi& height, vi& low, vi& stack, vb& in_stack, vvi& ans) {
		static int time = 0;

		height[node] = low[node] = time++;

		stack.push_back(node);
		in_stack[node] = true;

		for (int neighbor: adj[node]) {
			if (height[neighbor] == -1) {
				scc_inner(neighbor, height, low, stack, in_stack, ans);
				low[node] = min(low[node], low[neighbor]);
			} else if (in_stack[neighbor]) {
				low[node] = min(low[node], height[neighbor]);
			}
		}

		if (low[node] == height[node]) {
			vi component;
			int top_node;
			while (stack[stack.size()-1] != node) {
				top_node = stack[stack.size()-1];
				component.push_back(top_node);
				in_stack[top_node] = false;
				stack.pop_back();
			}
			top_node = stack[stack.size()-1];
			component.push_back(top_node);
			in_stack[top_node] = false;
			stack.pop_back();

			ans.push_back(component);
		}
	}

	// SCC, returns components in reverse topological order (!)
	vvi scc() {
		vi height(n, -1);
		vi low(n, INF);
		vi stack;
		vb in_stack(n, false);
		vvi ans;

		for (int i = 0; i < n; i++) {
			if (height[i] == -1) {
				scc_inner(i, height, low, stack, in_stack, ans);
			}
		}

		return ans;
	}
};

int main() {
    #ifndef ONLINE_JUDGE
        freopen("input.txt", "r", stdin);
        freopen("output.txt", "w", stdout);
    #endif

    ios::sync_with_stdio(false);
    cin.tie(0);

	int n, m;
	cin >> n >> m;
	DirectedGraph G(n);
	while (m--) {
		int a, b;
		cin >> a >> b;
		G.add_edge(a, b);
	}

	for (int x: G.topo_sort()) {
		cout << x << " ";
	}
	cout << endl;

	return 0;
}
