def sushu(n):
    primes=[True]*(n+1)
    p =2
    while p*p<n:
        if primes[p]:
            for i in range(p*2,n+1,p):
                primes[i] = False
        p += 1
    primes = [i for i in range(2,n) if primes[i]]
    return primes

print(sushu(100))