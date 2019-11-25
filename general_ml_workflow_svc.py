#%%
# roc curve and auc
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd
from sklearn.datasets import make_blobs
# we create 40 separable points
X, y = make_blobs(n_samples=40, centers=2, random_state=6)
X = np.random.normal(0,3,40)
X = np.reshape(X, (40,1))
print(X, y)
# fit models

# X = np.array([[x] for x in [0, 0.5, 1.2, 2,3]]) 
# y = np.array([1, 1, 0, 0, 0])

model = SVC(kernel='linear')
model.fit(X, y)
coef = model.coef_
print(model.coef_)

print(model.intercept_)
print(model.predict([[1.1]]),model.predict([[0.7]]))
# plt.plot(X)

plt.scatter(X, y)
plt.show()
# print(model.predict([[1.3,1]]),model.predict([[0.45,1]]))
# plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
# # plot the decision function
# ax = plt.gca()
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# # create grid to evaluate model
# xx = np.linspace(xlim[0], xlim[1], 30)
# yy = np.linspace(ylim[0], ylim[1], 30)
# YY, XX = np.meshgrid(yy, xx)
# xy = np.vstack([XX.ravel(), YY.ravel()]).T
# Z = model.decision_function(xy).reshape(XX.shape)
# # plot decision boundary and margins
# ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
#            linestyles=['--', '-', '--'])
# # plot support vectors
# ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
#            linewidth=1, facecolors='none')
# plt.show()
# # %%


# %%
