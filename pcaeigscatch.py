# -*- coding: utf-8 -*-
"""
Created on Thu May 12 22:12:48 2022

@author: siddh
"""

import numpy as np
import pandas as pd
import seaborn as sns

from numpy.linalg import eig

import matplotlib.pyplot as plt

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url,header = None,names=['feature1','feature2','feature3','featire4','response'])
sns.heatmap(df.corr(),annot=True)





def selectfloatfeatures(df):
    #df2 = df.select_dtypes(exclude = ['object']) // short cut for this logic and much better 
    boollist = []
    for i in range(df.shape[1]):
        if df.dtypes[i] == np.float64:
            boollist.append(True)
        else:
            boollist.append(False)
    df1 = df.loc[:,boollist]
    return df1



def covarriancematrix(df):
    #formula of co variance 1/n-1 sum( x-xbar  *  y - ybar   )
    covarmatrix = np.zeros(shape=(X.shape[1],X.shape[1]))
    x = 0
    for i in X.columns:
        y = 0
        for j in X.columns:
            
            covarmatrix[x,y] = sum((X[i] - X[i].mean()) * (X[j] - X[j].mean()))/(X.shape[0]-1)
            y += 1
        x += 1
    return covarmatrix

X = selectfloatfeatures(df)
covmat = covarriancematrix(X)
w,v=eig(covmat)
w/sum(w) * 100

Xmat = X.values
meanf1  = Xmat[:,0] - Xmat[:,0].mean()
meanf2 =  Xmat[:,1] - Xmat[:,1].mean()
meanf3=  Xmat[:,2] - Xmat[:,2].mean()
meanf4 = Xmat[:,3] - Xmat[:,3].mean()
resultant = np.column_stack(((meanf1,meanf2,meanf3,meanf4)))
pcamul = v[:,0].reshape(-1,1)
PCs = np.matmul(resultant,pcamul)

PCdot = np.dot(resultant,pcamul)


sns.scatterplot(data = PCs)
plt.show()

sns.scatterplot(data = df.iloc[:,0])
np.corrcoef(PCs)

rdf = pd.DataFrame(PCs)
rdf.corr()
sns.heatmap(rdf.corr(),annot=True)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(df.response) 
X = PCs
df_col = np.column_stack((X,y))
df_con = pd.DataFrame(df_col,columns = ['Freture','Response'])

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,cross_val_score

lr = LogisticRegression()
kfold = KFold(n_splits = 5,shuffle = True,random_state = 7)
cross_val_score(lr,X,y,scoring = 'accuracy',cv = kfold).mean()
cross_val_score(lr,df.iloc[:,:4],y,scoring = 'accuracy',cv = kfold).mean()


from sklearn.decomposition import PCA
frf = PCA(n_components=1)
new_df = frf.fit_transform(df.iloc[:,:4])


cross_val_score(lr,new_df,y,scoring = 'accuracy',cv = kfold).mean()
