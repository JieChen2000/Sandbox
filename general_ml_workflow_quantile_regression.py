'''
Quantile Regression, 
Originally from https://stackoverflow.com/questions/51791790/identifying-outliers-with-quantile-regression-and-python
'''
#%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

df = pd.DataFrame(np.random.normal(0, 1, (100, 2)))
df.columns = ['LST', 'NDVI']

model = smf.quantreg('NDVI ~ LST', df)
quantiles = [0.05, 0.95]
fits = [model.fit(q=q) for q in quantiles]
figure, axes = plt.subplots()
x = df['LST']
y = df['NDVI']

fit = np.polyfit(x, y, deg=1)
_x = np.linspace(x.min(), x.max(), num=len(y))

# fit lines
_y_005 = fits[0].params['LST'] * _x + fits[0].params['Intercept']
_y_095 = fits[1].params['LST'] * _x + fits[1].params['Intercept']

# start and end coordinates of fit lines
p = np.column_stack((x, y))
a = np.array([_x[0], _y_005[0]]) #first point of 0.05 quantile fit line
b = np.array([_x[-1], _y_005[-1]]) #last point of 0.05 quantile fit line

a_ = np.array([_x[0], _y_095[0]])
b_ = np.array([_x[-1], _y_095[-1]])

#mask based on if coordinates are above 0.95 or below 0.05 quantile fitlines using cross product
mask = lambda p, a, b, a_, b_: (np.cross(p-a, b-a) > 0) | (np.cross(p-a_, b_-a_) < 0)
mask = mask(p, a, b, a_, b_)

axes.scatter(x[mask], df['NDVI'][mask], facecolor='r', edgecolor='none', alpha=0.3, label='data point')
axes.scatter(x[~mask], df['NDVI'][~mask], facecolor='g', edgecolor='none', alpha=0.3, label='data point')

axes.plot(x, fit[0] * x + fit[1], label='best fit', c='lightgrey')
axes.plot(_x, _y_095, label=quantiles[1], c='orange')
axes.plot(_x, _y_005, label=quantiles[0], c='lightblue')

axes.legend()
axes.set_xlabel('LST')
axes.set_ylabel('NDVI')

plt.show()

# %%
