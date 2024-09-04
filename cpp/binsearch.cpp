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

// USE lower_bound AND upper_bound FOR STANDARD BINARY SEARCH

int max_f(vi A, function<int(int)> f) {
	int i = 0;
	for (int b = A.size() / 2; b >= 1; b /= 2) {
		while (i + b + 1 < A.size() and f(A[i+b]) < f(A[i+b+1])) i += b;
	}
	return i + 1;
}

int max_f(int lo, int hi, function<int(int)> f) {
	int x = lo;
	for (int b = (hi - lo) / 2; b >= 1; b /= 2) {
		while (f(x + b) < f(x + b + 1)) x += b;
		if (x > hi) return hi;
	}
	return x + 1;
}

int min_f(vi A, function<int(int)> f) {
	int i = -1;
	for (int b = A.size() / 2; b >= 1; b /= 2) {
		while (i + b + 1 < A.size() and f(A[i+b]) > f(A[i+b+1])) i += b;
	}
	return i + 1;
}

int min_f(int lo, int hi, function<int(int)> f) {
	int x = lo;
	for (int b = (hi - lo) / 2; b >= 1; b /= 2) {
		while (f(x + b) > f(x + b + 1)) x += b;
		if (x > hi) return hi;
	}
	return x + 1;
}

int main() {
	return 0;
}
