# icpc-templates
A large number of common algorithms and data structures implemented in Python and C++.

## About

These are templates I use for competitive programming competitions such as those hosted on [codeforces.com](https://codeforces.com/).
The most common binary search, ternary search, and graph algorithms are implemented in both languages. Some number theory algorithms are also implemented in Python. Additionally, the data structures DSU, Segment Trees, and Fenwick Trees are implemented in both languages.

## Testing

While tests have mostly been removed, a few remain. To run these tests or run the functions yourself, you can do the following.

### Python

Write a test under `if __name__ == "__main__":` and run the file. Input can be taken from standard input.

### C++

Write a test in the `main` function. The input will be written from `input.txt` and output will be written to `output.txt`, unless the lines 
```
#ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif
```
are deleted from the `main` function.
