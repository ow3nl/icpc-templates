"""Number Theory Algorithms"""
import math


# Inverse of x modulo p -- O(log(x))
def inverse(x, p):
    return pow(x, -1, p)

# Factors of x, sorted -- O(sqrt(x))
def factors(x):
    factors = []
    i = 1
    while i * i <= x:
        if x % i == 0:
            factors.append(i)
            if i * i != x:
                factors.append(x // i)
        
        i += 1

    factors.sort()
    return factors

# Checks if x is prime -- O(sqrt(x))
def is_prime(x):
    if x == 2:
        return True
    elif x % 2 == 0:
        return False
    else:
        i = 3
        while i * i <= x:
            if x % i == 0:
                return False
            i += 2
        
        return True

# Prime factorization of x, sorted -- O(sqrt(x))
def prime_factorization(x):
    factors = []
    i = 2
    while i * i <= x:
        if x % i == 0:
            factors.append(i)
            x //= i
        else:
            i += 1

    if x > 1:
        factors.append(x)
    
    return factors

# All primes up to x -- O(x log(x))
def primes_until(x):
    primes = [True for _ in range(x+1)]
    res = []

    p = 2
    while p * p <= x:
        if primes[p]:
            res.append(p)
            for i in range(p * p, x+1, p):
                primes[i] = False
        
        p += 1

    for i in range(p, x+1, 2):
        if primes[i]:
            res.append(i)

    return res


if __name__ == "__main__":
    print(factors(120))
    print(is_prime(121))
    print(is_prime(120))
    print(is_prime(123))
    print(is_prime(127))
    print(inverse(5, 7))
    print(primes_until(100))
