# -*- coding: utf-8 -*-
"""
Created on Thu May  5 19:17:10 2022

@author: siddh
"""


import numpy as np

np.random.seed(400)
sample1=np.random.randint(10,200,50)
sample2=np.random.randint(10,200,50)
delta=sample1.mean()-sample2.mean()
delta

sample3=np.concatenate((sample1,sample2))

mid_split=sample3.size//2
delt_list=np.array([])
for x in range(1000):
    np.random.shuffle(sample3)
    delt_list=np.append(delt_list,sample3[:mid_split].mean()-sample3[mid_split:].mean())

delt_list.sort()
for x,y in enumerate(delt_list):
    if y<=delta and delt_list[x+1]>delta:
        print("match Found at index: ",x)
        match_at=x

def percentile(lst,indx):
    count=0
    for x in lst: count+=1 if x==lst[indx] else 0
    return (lst[:indx].size+0.5*count)/lst.size
probability=1-percentile(delt_list,match_at)
print(probability)
