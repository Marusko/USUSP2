import pandas as pd
import numpy as np

comp = 1
tol = 0.000001
repl = 100

miss = pd.read_csv('data/v8_missing.csv')
compl = pd.read_csv('data/v8_complete.csv')
#print(miss.describe())
#print(compl.describe())

def calcMean(data):
    for i in range(data.shape[1]):
        col = data[:, i]
        mean = np.nanmean(col)
        col[np.isnan(col)] = mean
    return data


def pca(data, n):
    meanVec = np.mean(data, axis=0)
    centered = data - meanVec
    covMatrix = np.cov(centered, rowvar=False)
    eigVal, eigVec = np.linalg.eigh(covMatrix)

    sortedInd = np.argsort(eigVal)[::-1]
    topEigVec = eigVec[:, sortedInd[:n]]

    pc = centered @ topEigVec
    reconstructed = (pc @ topEigVec.T) + meanVec
    return reconstructed

comp = 1
tol = 0.000001
repl = 100
imputed = calcMean(np.copy(miss))
mask = np.isnan(miss)
oldObjective = float('inf')

for j in range(repl):
    #pca_data = pca()
    #imputed[np.isnan(v8_miss)] = pca_data[np.isnan(v8_miss)]
    if np.linalg.norm(imputed - old) < tol:
        break
    old = imputed




#1.nahradit chybajuce hodnoty priemerom stlpca
#2.pca
#3.opakovat pca kym rozdiel vacsi ako tol