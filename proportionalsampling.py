# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 08:33:50 2022

@author: siddh
"""

import random
import numpy as np

def fibonaccisum(normalizedlist):
    cummulativesum = 0
    final = []
    for i in normalizedlist:
        cummulativesum += i
        final.append(cummulativesum)
    return final

lst = [2,6,1.2,5.8,20]

summation = sum(lst)
normalizedlist = []
for i in lst:
    normalizedlist.append(i/summation)

cummulaivefinal = fibonaccisum(normalizedlist)

samplelist = []
for i in range(1000):
    r = np.random.uniform(0,1,1)

    for i in range(len(cummulaivefinal)):
        if r <= cummulaivefinal[i]:
            x = i
            break
    
    samplelist.append(lst[x])
    
dict1 = {}

for i in samplelist:
     dict1[i] = dict1.get(i,0) + 1
     
print(dict1)
    
    
