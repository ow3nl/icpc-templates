"""Binary Search Algorithms"""
import math

# Greatest i such that A[i] <= x
def bs(x, A, n):
    lo = 0
    hi = n - 1

    while lo < hi:
        mid = math.ceil((lo + hi) / 2)
        if A[mid] > x:
            hi = mid - 1
        else:
            lo = mid

    return lo

# Greatest i such that A[i] < x
def sbs(x, A, n) -> int:
    lo = 0
    hi = n - 1
    while lo < hi:
        mid = math.ceil((lo + hi) / 2)
        if A[mid] >= x:
            hi = mid - 1
        else:
            lo = mid

    return lo

# Least i such that A[i] >= x
def rbs(x, A, n):
    lo = 0
    hi = n - 1

    while lo < hi:
        mid = (lo + hi) // 2
        if A[mid] < x:
            lo = mid + 1
        else:
            hi = mid

    return lo

# Least i such that A[i] > x
def rsbs(x, A, n):
    lo = 0
    hi = n - 1

    while lo < hi:
        mid = (lo + hi) // 2
        if A[mid] <= x:
            lo = mid + 1
        else:
            hi = mid

    return lo

# Least i such that f(A[i]) is True
def fbs(f, A, n):
    lo = 0
    hi = n - 1

    while lo < hi:
        mid = (lo + hi) // 2
        if not f(A[mid]):
            lo = mid + 1
        else:
            hi = mid

    return lo

# Least i such that f(A[i]) is False
def rfbs(f, A, n):
    lo = 0
    hi = n - 1

    while lo < hi:
        mid = (lo + hi) // 2
        if f(A[mid]):
            lo = mid + 1
        else:
            hi = mid

    return lo

# least i such that f(A[i]) is maximal
def maxfbs(f, A, n):
    lo = 0
    hi = n - 1

    while lo < hi:
        mid = (lo + hi) // 2
        if f(A[mid]) < f(A[mid + 1]):
            lo = mid + 1
        else:
            hi = mid
    
    return lo

# least i such that f(A[i]) is minimal
def minfbs(f, A, n):
    lo = 0
    hi = n - 1

    while lo < hi:
        mid = (lo + hi) // 2
        if f(A[mid]) > f(A[mid + 1]):
            lo = mid + 1
        else:
            hi = mid
    
    return lo


# TODO: decreasing array binary search
