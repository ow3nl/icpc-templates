"""String Algorithms"""


# Return the greatest common denominator of s and t
def string_gcd(s, t):
    while s != t:
        if len(s) < len(t):
            s, t = t, s
        if s[:len(t)] != t:
            return ''
        s = s[len(t):]
    return s

# Check if string is a palindrome
def is_palindrome(s):
    l = len(s)
    for i in range(l // 2):
        if s[i] != s[l-i-1]:
            return False
    return True


if __name__ == "__main__":
    print(string_gcd('abcabc', 'abc'))
    print(string_gcd('abcabca', 'abc'))
    print(string_gcd('abababab', 'abab'))
