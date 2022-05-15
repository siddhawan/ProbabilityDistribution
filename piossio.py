# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 09:52:23 2022

@author: siddh
"""


import matplotlib.pyplot as plt

import math
#piossion distribution
def fact(x):
    prod = 1
    for i in range(x,0,-1):
        prod *= i
    return prod
def piossion(lambdaex , x):
    
    ans = ((lambdaex**x) * (math.e**(-lambdaex)))/fact(x)
    return ans

figure, axis = plt.subplots(2,1)
def piossioncdf(lambdaex,k):
    
    ans = 0
    for i in range(k+1):
        ans += piossion(lambdaex,i)
        axis[0].bar(i, ans,color = 'b')
        axis[1].bar(i, piossion(lambdaex, i),color = 'b')
        
   
    return ans
print(piossion(10, 10))
print(piossioncdf(30,50))
