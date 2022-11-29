import numpy as np
cimport numpy as np
import pandas as pd

from scipy import interpolate

from libc.math cimport sqrt as csqrt
from libc.math cimport pow as cpow
from libc.stdio cimport  sscanf
from libc.time cimport tm,mktime,time_t,strptime,strftime,localtime
from libc.stdio cimport sizeof

DTYPE = np.intc

cpdef calc_cashflow_cy(double bar,
                        double couponrate,
                        double start_date,
                        double next_coupon_rate,
                        double enddate,
                        freq = 1):

    cdef list cashflow = list()
    cdef list time_list = list()
    cdef double date_temp
    cdef double timedelata_365_s
    date_temp = next_coupon_rate
    timedelata_365_s = 365*24*60*60
    while enddate >= date_temp:
        cashflow.append(bar * couponrate)
        time_list.append((date_temp-start_date)/timedelata_365_s)
        date_temp =(date_temp +timedelata_365_s)
    cashflow.append(bar)
    return cashflow,time_list

def get_rate_list(duration,rate_list,time_list):
    f=interpolate.interp1d(x=duration,y=rate_list,kind='slinear')
    r_list = list(f(time_list))
    return r_list

# 债券精确定价函数
cpdef calc_precisePrice_cy(double bar,
                            double couponrate,
                            list r_list,
                            list time_list):

    per_coupon = bar * couponrate
    discount_coupon = 0
    cdef int arr_len = sizeof(time_list)
    for i in range(arr_len):
        r = r_list[i]
        time = time_list[i]
        if(r != r_list[-1]):
            discount_coupon = discount_coupon + per_coupon/(1 + r*0.01)**time
    return (discount_coupon + bar/(1 + r_list[-1]*0.01)**time_list[-1])


%timeit bond_preciseprice(bar,couponrate,r_list,time_list)

%timeit calc_precisePrice_py(bar,couponrate,r_list,time_list)

