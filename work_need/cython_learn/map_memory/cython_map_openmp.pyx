cimport cython
cimport openmp
from libc.math cimport log
from cython.parallel cimport prange
from cython.parallel cimport parallel
from libcpp.vector cimport vector

import time
ctypedef void FuncPrt(double[:],double[:],double[:])

cdef void serial_loop(double[:] A ,double[:] B ,double[:] C):
    cdef int N = A.shape[0]
    cdef int i 
    for i in range(N):
        C[i] = log[i]