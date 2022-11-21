import pandas as pd
import numpy as np
import matplotlib.pyplot as mpt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.optimize import least_squares
from sklearn.metrics import r2_score

df = pd.read_csv('Data\Data1.csv')
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=30)
reg = LinearRegression()
reg.fit(X_train, Y_train)
print('Linear Regressaion R2 = {}'.format(reg.score(X_test, Y_test)))

pca = PCA(0.80)
X_pca = pca.fit_transform(X)
print('X_pca shape = {}'.format(X_pca.shape))
print('X_pca = {}'.format(X_pca))

X_train_pca, X_test_pca, Y_train_pca, Y_test_pca = train_test_split(X_pca, Y, test_size=0.3, random_state=30)
reg_pca = LinearRegression()
reg_pca.fit(X_train_pca, Y_train_pca)
print('Linear Regressaion after PCA R2 = {}'.format(reg_pca.score(X_test_pca, Y_test_pca)))

# M = np.vstack([X, np.ones(len(X))]).T
# alpha = np.linalg.lstsq(M,Y, rcond=None)
# print(alpha)

M_pca = np.vstack([np.concatenate(X_pca), np.ones(len(X_pca))]).T
beta = np.linalg.lstsq(M_pca,Y, rcond=None)
r2_score()
print(beta)