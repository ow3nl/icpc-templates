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
#define pqi priority_queue

const int INF = 1e9 + 1;

class DSU {
public:
	int n;
	vi parent;
	vi size;

	DSU(int n) {
		this->n = n;
		parent = vi(n);
		for (int i = 0; i < n; i++) parent[i] = i;
		size = vi(n, 1);
	}

	int find(int elem) {
		if (parent[elem] == elem) return elem;
		parent[elem] = find(parent[elem]);
		return parent[elem];
	}

	void unite(int a, int b) {
		a = find(a);
		b = find(b);
		if (a != b) {
			if (size[a] < size[b]) {
				swap(a, b);
			}
			size[a] += size[b];
			parent[b] = a;
		}
	}
};

class Segment {
public:
	int n;
	int len;  // length of original array after adding dummy nodes
	int dummy;  // value of dummy variables inserted into the array
	vi array;

	Segment(int n, vi A, int dummy = 0) {
		this->n = n;

		len = 1;
		while (len < n) len *= 2;

		this->dummy = dummy;
		array = vi(2 * len, dummy);
		for (int i = len; i < len + n; i++) array[i] = A[i - len];

		for (int i = len - 1; i >= 0; i--) {
			array[i] = array[2*i] + array[2*i+1];  // CHANGE THE QUERY
		}
	}

	int query(int lo, int hi) {
		lo += len;
		hi += len;
		int s = dummy;
		while (lo <= hi) {
			if (lo % 2 == 1) s += array[lo++];  // CHANGE THE QUERY
			if (hi % 2 == 0) s += array[hi--];
			lo /= 2;
			hi /= 2;
		}
		return s;
	}

	// add x to the i-th element of the original array
	void add(int i, int x) {
		i += len;
		array[i] += x;
		for (i /= 2; i >= 1; i /= 2) {
			array[i] = array[2*i] + array[2*i+1];  // CHANGE THE QUERY
		}
	}
};

class Fenwick {
public:
	int n;
	vi array;

	Fenwick(int n, vi A) {
		this->n = n;

		vi prefix(n+1, 0);
		for (int i = 0; i < n; i++) {
			prefix[i+1] = prefix[i] + A[i];
		}

		array = vi(n+1, 0);
		for (int i = 1; i <= n; i++) {
			array[i] = prefix[i] - prefix[i - (i & -i)];
		}
	}

	// get the sum of the first i elements
	int prefix(int i) {
		int s = 0;
		while (i >= 1) {
			s += array[i];
			i -= i & -i;
		}
		return s;
	}

	// add x to the i-th element of the original array
	void add(int i, int x) {
		i++;
		while (i <= n) {
			array[i] += x;
			i += i & -i;
		}
	}
};

int main() {
    #ifndef ONLINE_JUDGE
        freopen("input.txt", "r", stdin);
        freopen("output.txt", "w", stdout);
    #endif
    
    ios::sync_with_stdio(false);
    cin.tie(0);

	vi A = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	Fenwick S(10, A);

	cout << S.prefix(4) << endl;
	cout << S.prefix(9) << endl;
	cout << S.prefix(7) << endl;
	cout << S.prefix(10) << endl;
	cout << S.prefix(0) << endl;
	cout << S.prefix(1) << endl;
	cout << S.prefix(2) << endl;

	S.add(1, 10);
	S.add(4, 10);
	// vi A = {1, 12, 3, 4, 15, 6, 7, 8, 9, 10}

	cout << S.prefix(4) << endl;
	cout << S.prefix(9) << endl;
	cout << S.prefix(7) << endl;
	cout << S.prefix(10) << endl;
	cout << S.prefix(0) << endl;
	cout << S.prefix(1) << endl;
	cout << S.prefix(2) << endl;

	return 0;
}
