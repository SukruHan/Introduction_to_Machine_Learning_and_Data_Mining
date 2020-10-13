import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import datetime

data = pd.read_csv("LA_Ozone.csv")

data.head(10)
dt = datetime.datetime(1976, 1, 1)
for i in range(330):
    dtdelta = datetime.timedelta(np.int(data.doy[i]))
    data.doy[i] = dt + dtdelta

from datetime import datetime
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

from sklearn import model_selection

y = df["ozone"].values
X = df.loc[:, df.columns != "ozone"]
X_array = X.values


y_df = df.ozone
low = int(y_df.mean() - y_df.std())
high = int(y_df.mean() + y_df.std())
class_y = pd.DataFrame()
class_y['low'] = (y_df <= low).astype('float')
class_y['medium'] = (np.logical_and(y_df > low, y_df <= high)).astype('float')
class_y['high'] = (y_df > high).astype('float')
y = class_y @ [0, 1, 2]
y = y.values

X_norm = (X_array - X_array.mean()) / X_array.std()

import sklearn.metrics.cluster as cluster_metrics


def gauss_2d(centroid, ccov, std=2, points=100):
    ''' Returns two vectors representing slice through gaussian, cut at given standard deviation. '''
    mean = np.c_[centroid];
    tt = np.c_[np.linspace(0, 2 * np.pi, points)]
    x = np.cos(tt);
    y = np.sin(tt);
    ap = np.concatenate((x, y), axis=1).T
    d, v = np.linalg.eig(ccov);
    d = std * np.sqrt(np.diag(d))
    bp = np.dot(v, np.dot(d, ap)) + np.tile(mean, (1, ap.shape[1]))
    return bp[0, :], bp[1, :]


def clusterplot(X, clusterid, centroids='None', y='None', covars='None'):
    '''
    CLUSTERPLOT Plots a clustering of a data set as well as the true class
    labels. If data is more than 2-dimensional it should be first projected
    onto the first two principal components. Data objects are plotted as a dot
    with a circle around. The color of the dot indicates the true class,
    and the cicle indicates the cluster index. Optionally, the centroids are
    plotted as filled-star markers, and ellipsoids corresponding to covariance
    matrices (e.g. for gaussian mixture models).

    Usage:
    clusterplot(X, clusterid)
    clusterplot(X, clusterid, centroids=c_matrix, y=y_matrix)
    clusterplot(X, clusterid, centroids=c_matrix, y=y_matrix, covars=c_tensor)

    Input:
    X           N-by-M data matrix (N data objects with M attributes)
    clusterid   N-by-1 vector of cluster indices
    centroids   K-by-M matrix of cluster centroids (optional)
    y           N-by-1 vector of true class labels (optional)
    covars      M-by-M-by-K tensor of covariance matrices (optional)
    '''
    import matplotlib.pyplot as plt
    X = np.asarray(X)
    cls = np.asarray(clusterid)
    if type(y) is str and y == 'None':
        y = np.zeros((X.shape[0], 1))
    else:
        y = np.asarray(y)
    if type(centroids) is not str:
        centroids = np.asarray(centroids)
    K = np.size(np.unique(cls))
    C = np.size(np.unique(y))
    ncolors = np.max([C, K])

    # plot data points color-coded by class, cluster markers and centroids
    # hold(True)
    colors = [0] * ncolors
    for color in range(ncolors):
        colors[color] = plt.cm.jet(color / (ncolors - 1))[:3]
    for i, cs in enumerate(np.unique(y)):
        plt.plot(X[(y == cs).ravel(), 0], X[(y == cs).ravel(), 1], 'o', markeredgecolor='k', markerfacecolor=colors[i],
                 markersize=6, zorder=2)
    for i, cr in enumerate(np.unique(cls)):
        plt.plot(X[(cls == cr).ravel(), 0], X[(cls == cr).ravel(), 1], 'o', markersize=12, markeredgecolor=colors[i],
                 markerfacecolor='None', markeredgewidth=3, zorder=1)
    if type(centroids) is not str:
        for cd in range(centroids.shape[0]):
            plt.plot(centroids[cd, 0], centroids[cd, 1], '*', markersize=22, markeredgecolor='k',
                     markerfacecolor=colors[cd], markeredgewidth=2, zorder=3)
    # plot cluster shapes:
    if type(covars) is not str:
        for cd in range(centroids.shape[0]):
            x1, x2 = gauss_2d(centroids[cd], covars[cd, :, :])
            plt.plot(x1, x2, '-', color=colors[cd], linewidth=3, zorder=5)
    plt.rcParams["figure.figsize"] = (20, 10)
    # hold(False)

    # create legend
    legend_items = np.unique(y).tolist() + np.unique(cls).tolist() + np.unique(cls).tolist()
    for i in range(len(legend_items)):
        if i < C:
            legend_items[i] = 'Class: {0}'.format(legend_items[i]);
        elif i < C + K:
            legend_items[i] = 'Cluster: {0}'.format(legend_items[i]);
        else:
            legend_items[i] = 'Centroid: {0}'.format(legend_items[i]);
    plt.legend(legend_items, numpoints=1, markerscale=.75, prop={'size': 9})


def clusterval(y, clusterid):
    '''
    CLUSTERVAL Estimate cluster validity using Entropy, Purity, Rand Statistic,
    and Jaccard coefficient.

    Usage:
      Entropy, Purity, Rand, Jaccard = clusterval(y, clusterid);

    Input:
       y         N-by-1 vector of class labels
       clusterid N-by-1 vector of cluster indices

    Output:
      Entropy    Entropy measure.
      Purity     Purity measure.
      Rand       Rand index.
      Jaccard    Jaccard coefficient.
    '''
    NMI = cluster_metrics.supervised.normalized_mutual_info_score(y, clusterid)

    # y = np.asarray(y).ravel(); clusterid = np.asarray(clusterid).ravel()
    C = np.unique(y).size;
    K = np.unique(clusterid).size;
    N = y.shape[0]
    EPS = 2.22e-16

    p_ij = np.zeros((K, C))  # probability that member of i'th cluster belongs to j'th class
    m_i = np.zeros((K, 1))  # total number of objects in i'th cluster
    for k in range(K):
        m_i[k] = (clusterid == k).sum()
        yk = y[clusterid == k]
        for c in range(C):
            m_ij = (yk == c).sum()  # number of objects of j'th class in i'th cluster
            p_ij[k, c] = m_ij.astype(float) / m_i[k]
    entropy = ((1 - (p_ij * np.log2(p_ij + EPS)).sum(axis=1)) * m_i.T).sum() / (N * K)
    purity = (p_ij.max(axis=1)).sum() / K

    f00 = 0;
    f01 = 0;
    f10 = 0;
    f11 = 0
    for i in range(N):
        for j in range(i):
            if y[i] != y[j] and clusterid[i] != clusterid[j]:
                f00 += 1;  # different class, different cluster
            elif y[i] == y[j] and clusterid[i] == clusterid[j]:
                f11 += 1;  # same class, same cluster
            elif y[i] == y[j] and clusterid[i] != clusterid[j]:
                f10 += 1;  # same class, different cluster
            else:
                f01 += 1;  # different class, same cluster
    rand = np.float(f00 + f11) / (f00 + f01 + f10 + f11)
    jaccard = np.float(f11) / (f01 + f10 + f11)

    return rand, jaccard, NMI
from matplotlib.pyplot import figure, show
from scipy.io import loadmat
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram


attributeNames = X.columns
N, M = X_norm.shape


# Perform hierarchical/agglomerative clustering on data matrix
#Method = 'single'
Method = 'complete'
Metric = 'euclidean'

Z = linkage(X_norm, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = 3
cls = fcluster(Z, criterion='maxclust', t=Maxclust) # t=opt_clust  -> daha sonra print edeersek diye
plt.title("Hierarchical Clustering", fontsize=20)
figure(1,figsize=(20,10))
clusterplot(X, cls.reshape(cls.shape[0],1), y=y)

# Display dendrogram
max_display_levels=6
figure(2,figsize=(20,10))
plt.title("Dendogram (Linkage Function)", fontsize=20)
dendrogram(Z, truncate_mode='level', p=max_display_levels)

show()


from matplotlib.pyplot import figure, plot, legend, xlabel, show
import numpy as np
from scipy.io import loadmat
from sklearn.mixture import GaussianMixture
from sklearn import model_selection

KRange = range(1, 11)
T = len(KRange)

covar_type = 'full'  # you can try out 'diag' as well
reps = 3  # number of fits with different initalizations, best result will be kept
init_procedure = 'kmeans'  # 'kmeans' or 'random'

# Allocate variables
BIC = np.zeros((T,))
AIC = np.zeros((T,))
CVE = np.zeros((T,))

# K-fold crossvalidation
CV = model_selection.KFold(n_splits=10, shuffle=True)

for t, K in enumerate(KRange):
    print('Fitting model for K={0}'.format(K))

    # Fit Gaussian mixture model
    gmm = GaussianMixture(n_components=K, covariance_type=covar_type,
                          n_init=reps, init_params=init_procedure,
                          tol=1e-6, reg_covar=1e-6).fit(X_norm)

    # Get BIC and AIC
    BIC[t,] = gmm.bic(X_norm)
    AIC[t,] = gmm.aic(X_norm)

    # For each crossvalidation fold
    for train_index, test_index in CV.split(X_norm):
        # extract training and test set for current CV fold
        X_train = X_norm[train_index]
        X_test = X_norm[test_index]

        # Fit Gaussian mixture model to X_train
        gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X_train)

        # compute negative log likelihood of X_test
        CVE[t] += -gmm.score_samples(X_test).sum()

opt_clust = KRange[CVE.argmin()]

# Plot results
figure(1, figsize=(20, 10));
plot(KRange, BIC,'-*b')
plot(KRange, AIC,'-xr')
plot(KRange, 2*CVE,'-ok')
legend(['BIC', 'AIC', 'Crossvalidation'])
plt.title("Comparison of BIC, AIC and Crossvalidation",fontsize = 22)
xlabel('K', fontsize = 18)
show()
opt_clust

# Evaluate the GMM and hierarchical clustering model in terms of the labels
gmm = GaussianMixture(n_components=opt_clust, covariance_type=covar_type,
                          n_init=reps, init_params=init_procedure,
                          tol=1e-6, reg_covar=1e-6).fit(X_norm)

gmm_cls = gmm.predict(X_norm)
gmm_cds = gmm.means_
gmm_covs = gmm.covariances_

figure(1, figsize=(20,10))
idx = [0, 1] # There are 8 attributes
plt.title("Cluster Centers Visualized (as stars)", fontsize =20)
clusterplot(X_norm[:, idx],  clusterid=gmm_cls, centroids=gmm_cds[:, idx], y=y, covars=gmm_covs[:,idx,:][:,:,idx])
#clusterplot(X_norm, clusterid=gmm_cls, centroids=gmm_cds, y=y, covars=gmm_covs)

hierarchical_cls = fcluster(Z, criterion='maxclust', t=opt_clust)
figure(2, figsize=(20,10))
plt.title("Hierarchical Clustering with 5 Clusters and Centroids", fontsize=20)
clusterplot(X_norm, hierarchical_cls.reshape(hierarchical_cls.shape[0],1), y=y)

show()

measures = np.zeros((2, 3))
measures[0] = np.array(clusterval(y, gmm_cls))
measures[1] = np.array(clusterval(y, hierarchical_cls))
measures_df = pd.DataFrame(measures, columns=['Rand', 'Jaccard', 'NMI'], index=['GMM', 'Hierarchical'])
measures_df


labels = ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"]
Frame = pd.DataFrame(gmm_cds, columns = X.columns, index=labels)
Frame


def gausKernelDensity(X, width):
    '''
    GAUSKERNELDENSITY Calculate efficiently leave-one-out Gaussian Kernel Density estimate
    Input:
      X        N x M data matrix
      width    variance of the Gaussian kernel

    Output:
      density        vector of estimated densities
      log_density    vector of estimated log_densities
    '''
    X = np.mat(np.asarray(X))
    N, M = X.shape

    # Calculate squared euclidean distance between data points
    # given by ||x_i-x_j||_F^2=||x_i||_F^2-2x_i^Tx_j+||x_i||_F^2 efficiently
    x2 = np.square(X).sum(axis=1)
    D = x2[:, [0] * N] - 2 * X.dot(X.T) + x2[:, [0] * N].T

    # Evaluate densities to each observation
    Q = np.exp(-1 / (2.0 * width) * D)
    # do not take density generated from the data point itself into account
    Q[np.diag_indices_from(Q)] = 0
    sQ = Q.sum(axis=1)

    density = 1 / ((N - 1) * np.sqrt(2 * np.pi * width) ** M + 1e-100) * sQ
    log_density = -np.log(N - 1) - M / 2 * np.log(2 * np.pi * width) + np.log(sQ)
    return np.asarray(density), np.asarray(log_density)

# Estimate the optimal kernel density width, by leave-one-out cross-validation
import numpy as np
from matplotlib.pyplot import figure, bar, title, plot, show,ylabel, xlabel


wwidths = X_norm.var(axis=0).max() * (2.0**np.arange(-10,3))
logP = np.zeros(np.size(widths))
for i,w in enumerate(widths):
    f, log_f = gausKernelDensity(X_norm, w)
    logP[i] = log_f.sum()
val = logP.max()
ind = logP.argmax()

width=widths[ind]
print('Optimal estimated width is: {0}'.format(width))

# Estimate density for each observation not including the observation
# itself in the density estimate
density, log_density = gausKernelDensity(X_norm, width)

# Sort the densities
i = (density.argsort(axis=0)).ravel()
density = density[i]

# Display the index of the lowest density data object
print('Lowest density: {0} for data object: {1}'.format(density[0],i[0]))

# Plot density estimate of outlier score
figure(1)
bar(range(20),density[:20].reshape(-1,))
xlabel("20 Data Points with Lowest Density", fontsize = 17)
ylabel("Density Estimates", fontsize = 17)
title('Gaussian Kernel Density Estimates for Lowest 20 Density Data Points', fontsize=20)
figure(2)
plot(logP)
title('Optimal width', fontsize=20)
show()

import numpy as np
from matplotlib.pyplot import figure, subplot, plot, hist, title, show,  xlabel, ylabel
from sklearn.neighbors import NearestNeighbors

# Number of neighbors
K = 5
# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(X_norm)
D, i = knn.kneighbors(X_norm)

print("Compute the density")
# Compute the density
#D, i = knclassifier.kneighbors(np.matrix(xe).T)
knn_density = 1./(D.sum(axis=1)/K)

i = knn_density.argsort()
knn_density = knn_density[i]

# Plot KNN density
print("Plot KNN density")
figure(figsize=(20,10))
title('Data histogram')
xlabel("20 Data Points with Lowest Density", fontsize = 17)
ylabel("KNN Density Estimates", fontsize = 17)
bar(range(20), knn_density[:20])
title('KNN density for Lowest 20 Density Data Points', fontsize = 20)

print("Compute the average relative density")
# Compute the average relative density
DX, iX = knn.kneighbors(X_norm)
knn_densityX = 1./(DX[:,1:].sum(axis=1)/K)
knn_avg_rel_density = knn_densityX/(knn_densityX[iX[:,1:]].sum(axis=1)/K)

i2 = knn_avg_rel_density.argsort()
knn_avg_rel_density = knn_avg_rel_density[i2]



# Plot KNN average relative density
print("Plot KNN average relative density")
figure(figsize=(20,10))
bar(range(20), knn_avg_rel_density[:20])
title('KNN Average Relative Density for Lowest 20 Density Data Points', fontsize = 20)
xlabel("20 Data Points with Lowest Density", fontsize = 17)
ylabel("KNN Average Relative Density Estimates", fontsize = 17)
show()


att_names = df.columns
att_names

dfBin = pd.DataFrame()
for column in df.columns:
    median = df[column].median()
    dfBin['%s_0th-50th'%column] = df[column] <= median
    dfBin['%s_50th-100th'%column] = df[column] > median
dfBin *= 1

att_namesBin = dfBin.columns
dfBin = dfBin.reset_index(drop=True)
dfBin

def mat2transactions(X, labels=[]):
    T = []
    for i in range(X.shape[0]):
        l = np.nonzero(X.iloc[i, :])[0].tolist()
        l = [labels[i] for i in l]
        T.append(l)
    return T

T = mat2transactions(dfBin, labels=att_namesBin)

from apyori import apriori
def print_apriori_rules(rules, min_sup, min_conf):
    frules = []
    for r in rules:
        for o in r.ordered_statistics:
            conf = o.confidence
            supp = r.support
            x = list(o.items_base)
            y = list( o.items_add)
            temp_list = (x, y, supp, conf)
            print("{%s} -> {%s}  (supp: %.3f, conf: %.3f)"%(x,y, supp, conf))
            frules.append(temp_list)
    cols = ["{X} ->", "{Y}", "Support (>%s)"%min_sup, "Confidence (>%s)"%min_conf]
    result = pd.DataFrame(frules, columns=cols).sort_values(by="Confidence (>%s)"%min_conf, ascending=False).reset_index(drop=True)
    return result

rules = apriori(T, min_support=0.30, min_confidence=0.95)
print_apriori_rules(rules, min_sup=0.30 , min_conf=0.95)