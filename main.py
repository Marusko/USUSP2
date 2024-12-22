import pandas as pd
import numpy as np
#np.set_printoptions(threshold=np.inf)
miss = pd.read_csv('data/edit.csv').values
compl = pd.read_csv('data/editCompl.csv').values
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
    pcaData = pca(imputed, comp)
    imputed[mask] = pcaData[mask]
    objective = np.sum((miss[~mask] - imputed[~mask]) ** 2)

    if abs(oldObjective - objective) < tol:
        print(f"Converged after {j + 1} iterations")
        break
    oldObjective = objective

print("Imputed Data:")
print(imputed)
diff = np.abs(compl - imputed)
print("Difference between PCA and complete:")
print(diff)
print(f"Smallest difference: {np.min(diff[mask])}")
print(f"Largest difference: {np.max(diff[mask])}")
print(f"Average difference: {np.mean(diff[mask])}")




#1.nahradit chybajuce hodnoty priemerom stlpca
#2.pca
#3.opakovat pca kym rozdiel vacsi ako tol