#cython:language_level=3
from libc.math cimport sqrt as csqrt
from libc.math cimport pow as cpow

def sieve_of_ethen_v2(long n):
    """返回给定小于整数N的所有质数"""
    pr = [True for i in range(n + 1)]
    p = 2
    res=list()

    while (p * p <= n):
        if (pr[p] == True):
            for i in range(p * p, n + 1, p):
                pr[i] = False
            #end-for
        #end-if
        p += 1
    #end-while

    for k in range(2,n):
        if pr[k]:
            res.append(k)
        #end-if
    #end-for
    return res
#end-def

