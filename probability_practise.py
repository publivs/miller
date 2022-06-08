import pandas as pd
import numpy as np
import sympy as sy
import scipy as sp

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