import pandas as pd
import numpy as np

comp = 1
tol = 0.000001
repl = 100

miss = pd.read_csv('data/v8_missing.csv')
compl = pd.read_csv('data/v8_complete.csv')
#print(miss.describe())
#print(compl.describe())

def calc_mean(data):
    for i in range(data.shape[1]):
        col = data[:, i]
        mean = np.nanmean(col)
        col[np.isnan(col)] = mean
    return data

#def pca():
#    return new

imputed = calc_mean(np.copy(miss))
old = np.copy(imputed)

for j in range(repl):
    #pca_data = pca()
    #imputed[np.isnan(v8_miss)] = pca_data[np.isnan(v8_miss)]
    if np.linalg.norm(imputed - old) < tol:
        break
    old = imputed




#1.nahradit chybajuce hodnoty priemerom stlpca
#2.pca
#3.opakovat pca kym rozdiel vacsi ako tol