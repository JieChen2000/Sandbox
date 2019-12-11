#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd 
# from sklearn.metrics import accuracy_score
# from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os 
# X, y_true = make_blobs(n_samples=3000, centers=4,
#                        cluster_std=0.9, random_state=0)
# plt.scatter(X[:, 0], X[:, 1], s=50);
file_path = r"C:\Users\Jay.J.Chen\Documents\Projects\Hackathon\Scotford\Problem2\data"
file_name = '5m_unsupervised_set_fix.csv'
df = pd.read_csv(os.path.join(file_path, file_name), encoding = "ISO-8859-1")
df = df.dropna()
df = df.drop(columns=['timestamp'])
df = df[(df != 0).all(1)]

# df['RPM'] = df['RPM'].astype(float)
# df['PUMP_OUTBOARD'] = df['PUMP_OUTBOARD'].astype(float)
features = list(df) 
df.head(2)

#%%
# df.isnull()
# plot pairplot
sns.pairplot(df)
plt.show()
# plt.savefig('data/pair-plot.png')
# plt.close()
# check correlations
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()
# plt.savefig('data/corr-plot.png')
# plt.close()

#%%
# for 1d fit 
# X = df.iloc[:, 1]
# X = X.to_numpy().reshape(-1,1)
# plt.hist(X)
# scaler = StandardScaler().fit(X)
# X = scaler.transform(X) 
# print(np.amax(X), np.amin(X))
# plt.plot(X)
#%%

# print(X.shape, np.amax(X))

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(df)
centers = kmeans.cluster_centers_
print("centers w/o scale", centers)
#%%
# centers = scaler.transform(centers)

df['class_pred'] = kmeans.predict(df)
#%%
# df['class_true'] = y_true
# print(accuracy_score(df['class_true'], df['class_pred']))
# print(accuracy_score(y_kmeans, y_true))

# print(df.head(5))

for cl in range(kmeans.n_clusters):
    # print('class=',cl)
    # print(df.loc[df['class_pred'] == cl])
    data_tmp = df.loc[df['class_pred'] == cl] 
    # print(data_tmp.head(3))
    for feature_name in features:
        print('cluster:', cl, 'centers at', centers[cl,:], 'for ', feature_name, ' with std deviation: ', np.std(data_tmp[feature_name]), ' and 95 percentile at ', np.percentile(data_tmp, 95))        

# %%
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['class_pred'], s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title(features[1] +' vs. '+ features[0])
plt.show()
# plt.savefig('data/cluster-plot.png')
# plt.close()
# %%
