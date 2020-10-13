import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sys

data = pd.read_csv("LA_Ozone.csv")
pd.options.mode.chained_assignment = None  # default='warn'
dt = datetime.datetime(1976, 1, 1)
for i in range(330):
    dtdelta = datetime.timedelta(np.int(data.doy[i]))
    data.doy[i] = dt + dtdelta

df = pd.DataFrame(data)
new_names = {
    "doy": "date",
    "vh" : "vh",
    "ibh" : "ibh",
    "dpg" : "dpg",
    "ibt" : "ibt",
    "vis" : "vis",
    "ozone" : "ozone",
    "wind" : "wind",
    "humidity": "humidity",
    "temp" : "temp"
}
df.rename(columns=new_names, inplace=True)
df.head(10)
df['date'] = pd.to_datetime(df['date'])
df = df.set_index("date")
df.head(10)
print(df.describe())

att_names = df.columns[1:9]
att_names_whole = df.columns

# HISTOGRAM

from matplotlib.pyplot import figure, subplot, hist, xlabel, ylim, show
import numpy as np

df_values = df.values
m = len(att_names_whole)
figure(figsize=(15, 10))
u = np.floor(np.sqrt(m))
v = np.ceil(float(9) / u)
for i in range(m):
    subplot(u, v, i + 1)
    hist(df_values[:, i], color=(0.2, 0.9 - i * 0.1, 0.4))
    xlabel(att_names_whole[i])
    ylim(0, len(df["ozone"]) / 2)

show()

#BOXPLOT

from matplotlib.pyplot import boxplot, xticks, ylabel, title, show
boxplot(df_values)
xticks(range(1,10),att_names_whole, rotation=45)
ylabel('cm')
title('LA_Ozone data - boxplot')
show()

from scipy.stats import zscore
zscored_data = zscore(df_values, ddof = 1)
from matplotlib.pyplot import (figure, title, boxplot, xticks)
figure(figsize=(12,6))
title('LA_Ozone data: Boxplot (standarized)')
boxplot(zscored_data)
xticks(range(1,m+1), att_names_whole, rotation=45)

#P-VALUES TABLE

z, pval = scipy.stats.normaltest(df)
p = ["P-values"]
Z = ["Z-values"]
table_of_pvalues = pd.DataFrame([pval], columns=df.columns[:], index=p)
print(table_of_pvalues)

# Q-Q PLOT

import pylab
import scipy.stats as stats
plt.figure(figsize=(9,9))
for i,col in enumerate(att_names_whole):
    ax = plt.subplot(3,3,i+1)
    stats.probplot(df[col], dist="norm", plot=pylab)
    ax.set_title(col)
pylab.show()

# CORRELATION HEAT MAP

corr = df.corr()
size = 15
fig, ax = plt.subplots(figsize=(size, size))
sys.heatmap(corr)

# CORRELATION WITH SCATTER PLOT

from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel,
                               xticks, yticks, show)
figure(figsize=(15,15))
for m1 in range(m):
    for m2 in range(m):
        subplot(m, m, m1*m + m2 + 1)
        for i in range(len(att_names_whole)):
            plot(df_values[:,m2], df_values[:,m1], '.', alpha = 0.9, color=(0.2, 0.9-i*0.1, 0.9), markersize=0.5)
            if m1==m-1:
                xlabel(att_names_whole[m2])
            else:
                xticks([])
            if m2==0:
                ylabel(att_names_whole[m1])
            else:
                yticks([])
show()


# PCA ANALYSIS

from sklearn import preprocessing
#Dataframe to numpy
PCA_X = df.drop(columns = ["ozone"])
print(PCA_X.head(10))
X = PCA_X.values
print(X)

# Normalizing whole data
scaler = preprocessing.StandardScaler()
# Fit our data on the scaler object
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=att_names_whole)

# This is for PCA
X_normalized = scaler.fit_transform(X)
df_X_normalized = pd.DataFrame(X_normalized, columns=att_names)

print(scaled_df.describe())

# PCA Analysis and Variance Explanation

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#%matplotlib inline

#Now, perform PCA
pca = PCA()
pca.fit(X_normalized)
rho = pca.explained_variance_ratio_
print(rho)
threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()


# PCA Components

cumsum_of_ = np.cumsum(rho)
threshold = 0.9
cumsum, Comp_numb, counter = 0, 0, 0
for i in range(len(rho)):
    if cumsum < threshold:
        cumsum += rho[i]
        counter += 1
        Comp_numb = counter
    #else: print(Comp_numb)
print("Total needed number of Components to exceed threshold are",Comp_numb)
# we can see component values as:
pca.components_
#Just see important ones which will be the directions:
directions = (pca.components_[0:Comp_numb])
row_names = ['PCA%d' % (i+1) for i in range(Comp_numb)]
table = pd.DataFrame(directions, columns=att_names, index=row_names)
print("Table of directions is as: \n", table)
print("PCA Components are as: \n",pca.components_)
#print(rho)
all_directions = (pca.components_)
all_row_names = ['PCA%d' % (i+1) for i in range(8)]
all_table = pd.DataFrame(all_directions, columns=att_names, index=all_row_names)
print(all_table)

# PRINTING PC's

from matplotlib.pyplot import xlabel, ylabel
from scipy.linalg import svd

# Subtract mean value from data
Y = X_normalized - np.ones((len(df["ozone"]), 1)) * X_normalized.mean(0)

# PCA by computing SVD of Y
U, S, Vh = svd(Y, full_matrices=False)
# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T

rho2 = (S * S) / (S * S).sum()
print(rho2)
print(rho)

# Project the centered data onto principal component space
Z = Y @ V


# PCA Component Coefficients


N, M = X.shape

pcs = [0, 1, 2, 3, 4]
legendStrs = ['PC1', "PC2", "PC3", "PC4", "PC5"]
c = ['r', 'g', 'b']
bw = .2
r = np.arange(1, M+1)
for i in pcs:
    plt.bar(r+i*bw, V[:, i], width=bw)
plt.xticks(r+bw, att_names, rotation=45)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Ozone: PCA Component Coefficients')
plt.show()

# Plot PCA of the data

# Z = array(Z)

for i in range(int(7)):
    plt.figure(figsize=(15, 15))
    plt.subplot(4, 4, 1 + i)
    plt.scatter(Z[:, 0], Z[:, i + 1], alpha=.7, c=df["ozone"])
    xlabel("PC1")
    ylabel("PC%d" % (i + 2))
    cbar = plt.colorbar()
    cbar.set_label('ozone')

for i in range(int(6)):
    plt.figure(figsize=(15, 15))
    plt.subplot(4, 4, 1 + i)
    plt.scatter(Z[:, 1], Z[:, i + 2], alpha=.7, c=df["ozone"])
    xlabel("PC2")
    ylabel("PC%d" % (i + 3))
    cbar = plt.colorbar()
    cbar.set_label('ozone')

for i in range(int(5)):
    plt.figure(figsize=(15, 15))
    plt.subplot(4, 4, 1 + i)
    plt.scatter(Z[:, 2], Z[:, i + 3], alpha=.7, c=df["ozone"])
    xlabel("PC3")
    ylabel("PC%d" % (i + 4))
    cbar = plt.colorbar()
    cbar.set_label('ozone')

plt.rcParams.update({'figure.max_open_warning': 0})

for i in range(int(4)):
    plt.figure(figsize=(15, 15))
    plt.subplot(4, 4, 1 + i)
    plt.scatter(Z[:, 3], Z[:, i + 4], alpha=.7, c=df["ozone"])
    xlabel("PC4")
    ylabel("PC%d" % (i + 5))
    cbar = plt.colorbar()
    cbar.set_label('ozone')

for i in range(int(3)):
    plt.figure(figsize=(15, 15))
    plt.subplot(4, 4, 1 + i)
    plt.scatter(Z[:, 4], Z[:, i + 5], alpha=.7, c=df["ozone"])
    xlabel("PC5")
    ylabel("PC%d" % (i + 6))
    cbar = plt.colorbar()
    cbar.set_label('ozone')

for i in range(int(2)):
    plt.figure(figsize=(15, 15))
    plt.subplot(4, 4, 1 + i)
    plt.scatter(Z[:, 5], Z[:, i + 6], alpha=.7, c=df["ozone"])
    xlabel("PC6")
    ylabel("PC%d" % (i + 7))
    cbar = plt.colorbar()
    cbar.set_label('ozone')

for i in range(int(1)):
    plt.figure(figsize=(15, 15))
    plt.subplot(4, 4, 1 + i)
    plt.scatter(Z[:, 6], Z[:, i + 7], alpha=.7, c=df["ozone"])
    xlabel("PC7")
    ylabel("PC%d" % (i + 8))
    cbar = plt.colorbar()
    cbar.set_label('ozone')

plt.show()




