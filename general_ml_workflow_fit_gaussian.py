#%%
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

# generate 2 class dataset
X, y = make_regression(n_samples=500, n_features=1, noise=0.5, random_state=1)
#%%

# convert numpy array to data frame.
dataset = pd.DataFrame()
for i in range(X.shape[1]):
    feature_name = 'feature' + str(i)
    dataset[feature_name]= X[:, i]
dataset['target'] = y  
# #%%
# plot pairplot
sns.pairplot(dataset)
plt.show()
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


# fit gaussian model
#%%
# gmm = GaussianMixture(n_components=1, covariance_type='spherical',max_iter=1000)
# a = np.random.normal(0,0.5,10000)
# r = gmm.fit(a[:, np.newaxis])
# print(r.get_params(), r.score(a[:, np.newaxis]), r.score(trainX), r.score(testX), r.means_, r.covariances_, r.weights_)  # one feature, one component gives single gausian.

# %%

# Define some test data which is close to Gaussian
# data = np.random.normal(1,2,10000)
data = trainX
print(np.percentile(data, [90,95]))

hist, bin_edges = np.histogram(data, density=True)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

# Define model function to be used to fit to the data above:

# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
p0 = [1., 0., 1.]

coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)

# Get the fitted curve
hist_fit = gauss(bin_centres, *coeff)

plt.plot(bin_centres, hist, label='Test data')
plt.plot(bin_centres, hist_fit, label='Fitted data')

# Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
print('Fitted mean = ', coeff[1])
print ('Fitted standard deviation = ', coeff[2])

plt.show()

# %%
