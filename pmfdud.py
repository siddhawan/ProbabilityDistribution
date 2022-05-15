# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 11:32:17 2022

@author: siddh
"""

import matplotlib.pyplot as plt


def pmfdud(samplespace):
    return 1/len(samplespace)
def cdfdud(samplespace,k):
    return ((k-samplespace[0]+1)/len(samplespace))
    
def meandud(samplespace):   
    return ((samplespace[0] + samplespace[-1])/2)


def vardud(samplespace):
    return ((samplespace[-1] - samplespace[0] + 1)**2 - 1)/12
    
samplespace = [1,2,3,4,5,6]


print(pmfdud(samplespace),cdfdud(samplespace, 5),meandud(samplespace),vardud(samplespace))

fig,axis = plt.subplot(2,1)
for i in samplespace:
    axis[0].line(pmfdud(samplespace),)
    