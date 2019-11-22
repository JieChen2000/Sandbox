#%%
# roc curve and auc
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd
import statsmodels.api as sm

# generate 2 class dataset
X, y = make_regression(n_samples=500, n_features=10, noise=0.5, random_state=1)
#%%

# convert numpy array to data frame.
dataset = pd.DataFrame()
for i in range(X.shape[1]):
    feature_name = 'feature' + str(i)
    dataset[feature_name]= X[:, i]
dataset['target'] = y  
# #%%
# # plot pairplot
# sns.pairplot(dataset)
# plt.show()
# #%%
# # check correlations
# corr = dataset.corr()
# sns.heatmap(corr, 
#             xticklabels=corr.columns.values,
#             yticklabels=corr.columns.values)
# plt.show()

#%%%
# convert df to array 
features = dataset.columns[:-1]
feature_vectors = dataset[features]
print(feature_vectors)
type(feature_vectors)
#%%
# split into train/test sets
# trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
trainX, testX, trainy, testy = train_test_split(feature_vectors, y, test_size=0.5, random_state=2)

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(testy))]
# fit simple linear models
model = LinearRegression()
model.fit(trainX, trainy)


#%%
# obtain the coefficient of determination (𝑅²) 
r_sq = model.score(trainX, trainy)
print('Training coefficient of determination:', r_sq)
r_sq = model.score(testX, testy)
print('Test coefficient of determination:', r_sq)
trainy_pred = model.predict(trainX)
r_sq = r2_score(trainy, trainy_pred)
print('Training r2_score:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

# predict regression values
yhat = model.predict(testX)

# calculate scores
r_sq = r2_score(testy, yhat)
print('Test r2_score:', r_sq)
print("Mean squared error: %.2f"
      % mean_squared_error(testy, yhat))

# Plot outputs against first feature
plt.scatter(testX['feature0'], testy,  color='black')
plt.plot(testX['feature0'], yhat, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()

