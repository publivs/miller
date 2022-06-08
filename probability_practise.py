import pandas as pd
import numpy as np
import sympy as sy
import scipy as sp


def calc_C(n,m):
    a = sy.factorial(n)
    b = sy.factorial(m) * sy.factorial(n-m)
    return a/b
# practise_5
x = sy.symbols('x')
y = sy.symbols('y')

f_xy= 4.8*y*(2-x)

f_x = sy.integrate(f_xy,(y,0,x))
f_y = sy.integrate(f_xy,(x,y,1))

# practise 6
f_xy = sy.exp(-y)
f_x = sy.integrate(f_xy,(y,x,sy.oo))
f_y = sy.integrate(f_xy,(x,0,y))

# practise 7
f_xy = x**2*y
c_rever = sy.integrate(f_xy,(y,x**2,1),(x,-1,1))
c_ = 1/c_rever
f_xy = c_* x**2*y

f_x = sy.integrate(f_xy,(y,x**2,1))

f_y = sy.integrate(f_xy,(x,-sy.sqrt(y),sy.sqrt(y)))
# practise_11——1
f_xy_y= f_xy/f_y
f_xy_y.evalf(subs={'y':1/2})

# practise_11——2
f_xy_y= f_xy/f_y
f_xy_y.evalf(subs={'y':1/2})

f_xy_x= f_xy/f_x
f_xy_x.evalf(subs={'y':1/3,'x':1/2})

# practise_11——3
'''
由已知
x**2<y<1
'''
A = f_xy_x.evalf(subs={'x':1/2})
ans_11_3_1 = sy.integrate(A,(y,1/4,1))
ans_11_3_2 = sy.integrate(A,(y,3/4,1))

'''
条件概率 ： f(x|y) = f(x,y)/f(y)
'''


# 9
n = sy.symbols('n')
m = sy.symbols('m')
#
f_xy = sy.exp(-14)* 7.14**(m) * 6.86**(n - m) /( sy.factorial(m) *sy.factorial(n-m))
'''
提出:sy.exp(-14) ,上下同乘 n!
 = (sy.exp(-14)/sy.factorial(n) ) * (sy.factorial(n) * (7.14**(m) * 6.86**(n - m)) /( sy.factorial(m) *sy.factorial(n-m)))
二项式展开后
=sy.exp(-14)/sy.factorial(n)* (7.14 + 6.86)**(n)
'''
f_x =  sy.exp(-14)/sy.factorial(n)* (7.14 + 6.86)**(n)

'''
P{Y=m} = SUM[m,n]P{X=n,Y=m}
= sy.exp(-14)/sy.fatorial(m) *(7.14)**m   * SUM[m,oo]
'''

f_y_1 = sy.exp(-14)/sy.factorial(m) *(7.14)**m

f_y_2 = (6.86)**(n-m) / sy.factorial(n-m)
f_y_2 = sy.summation(f_y_2,(n,m,sy.oo))

# 对f_y_2变形
'''
令  k = n-m,
则f_y_2变形为
sy.summation(6.86**k/sy.factorial(k),k,(k,0,sy.oo))
'''
k = sy.symbols('k')
f_y_2 = sy.summation(6.86**k/sy.factorial(k),(k,0,sy.oo))
print(f_y_2)
f_y_2 = sy.exp(1)**6.86
f_y  = f_y_1 * f_y_2



# practise_12
f_xy = 1
f_x = sy.integrate(f_xy,(y,-x,-x))

# 下半部分 
f_y_1 = sy.integrate(f_xy,(x,-y,1))

# 上半部分
f_y_2 = sy.integrate(f_xy,(x,y,1))

f_xy_x_1 = f_xy

def normal_distributes(mu,sigma,symbols = 'x'):
    x = sy.symbols(symbols)
    func_normal_dist = 1/sy.sqrt(2*sy.pi) * sy.exp(-(x-mu)**2/2*(sigma**2))
    return func_normal_dist

# 第四章 12题
'''

'''
f_xy = 1
x =sy.symbols('x')
y = sy.symbols('y')
f_X = sy.integrate(f_xy,(y,-x,x))

# 分类讨论,因为上下不同
f_Y_up = sy.integrate(f_xy,(x,y,1))
f_Y_down = sy.integrate(f_xy,(x,-y,1))

# 14
'''
X是为1的均匀分布，且两者互相独立,相乘即可
确定了计算的域
x{0,1} ,y>0
'''
x = sy.symbols('x')
y = sy.symbols('y')
f_Y = (1/2)*sy.exp(-y/2)
f_X = 1
f_xy = f_Y*f_X
#
from scipy.stats import norm
integrate_y_14 = sy.integrate(f_xy,(y,0,x**2))
# 对于exp(-x^2/2),转换为正态分布函数求Fi即可
answer_14 = 1 - np.sqrt(2)*sy.pi*(norm.cdf(1) - norm.cdf(0))

# asnwer_15
'''
由题已知 X,Y的概率密度分布为1/sy.sqrt(2*sy.pi) * 
'''
func_x = normal_distributes(0,1,symbols = 'x')
func_y = normal_distributes(0,1,symbols = 'y')
f_xy = (func_x * func_y).simplify()
theta = sy.symbols('theta')
r = sy.symbols('r', nonnegative=True)
r_fxy = f_xy.subs(((x, r*sy.cos(theta)), (y, r*sy.sin(theta)))).simplify()
r_fxy = sy.integrate(r_fxy,(theta,0,2*sy.pi))