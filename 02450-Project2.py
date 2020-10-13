import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import datetime
from numpy.linalg import inv
from sklearn import model_selection
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import torch
from torch import nn
from itertools import combinations
import scipy.stats as stats


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

df.describe()

from sklearn import model_selection

# Since at every model I recreate the data to not to mess it up, I will directly delete "doy" column.
df = pd.read_csv("LA_Ozone.csv")
y = df["ozone"].values
X = df.loc[:, df.columns != "ozone"].values
attributeNames = df.loc[:, df.columns != "ozone"].columns
N, M = X.shape
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = ['Offset', *attributeNames]
M = M+1
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)
lambdas = np.power(10.,range(-5,9))

# Initialize variables
#T = len(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, title, subplot, show, grid, savefig)

k = 0
for train_index, test_index in CV.split(X, y):

    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10

    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train,
                                                                                                      lambdas,
                                                                                                      internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)

    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]

    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train - y_train.mean()).sum(axis=0) / y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train - X_train @ w_rlr[:, k]).sum(axis=0) / y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:, k] = np.linalg.solve(XtX, Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train - X_train @ w_noreg[:, k]).sum(axis=0) / y_train.shape[0]
    Error_test[k] = np.square(y_test - X_test @ w_noreg[:, k]).sum(axis=0) / y_test.shape[0]
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    # m = lm.LinearRegression().fit(X_train, y_train)
    # Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    # Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Display the results for the last cross-validation fold
    if k == K - 1:
        figure(k, figsize=(15, 10))
        # figure().subplots_adjust(wspace = 2)
        subplot(1, 2, 1)
        semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], '.-')  # Don't plot the bias term
        xlabel("Regularization factor (λ's (lambda))")
        ylabel('Mean Coefficient Values (w* values)')
        title('w* (weights) values acc. to λ (lambda) values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner
        # plot, since there are many attributes
        legend(attributeNames[1:], loc='best')
        savefig("Weights.png")

        subplot(1, 2, 2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas, train_err_vs_lambda.T, 'b.-', lambdas, test_err_vs_lambda.T, 'r.-')
        xlabel("Regularization factor (λ's (lambda))")
        ylabel('Squared error (crossvalidation)')
        legend(['Train error', 'Validation error'])
        title("Lambda (λ's) values acc. to Squared Error")
        grid()
        savefig("Lambda.png")

    k += 1

show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum() - Error_train.sum()) / Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum() - Error_test.sum()) / Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format(
    (Error_train_nofeatures.sum() - Error_train_rlr.sum()) / Error_train_nofeatures.sum()))
print(
    '- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum() - Error_test_rlr.sum()) / Error_test_nofeatures.sum()))

weights = pd.DataFrame()
print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m, -1], 2)))
    weights[attributeNames[m]] = np.round(w_rlr[m, -1], 2)
print(weights.T)


def train_neural_net(model, loss_fn, X, y, n_replicates=3, max_iter=10000, tolerance=1e-6):
    import torch
    # Specify maximum number of iterations for training
    logging_frequency = 1000  # display the loss every 1000th iteration
    best_final_loss = 1e100
    for r in range(n_replicates):
        print('\n\tReplicate: {}/{}'.format(r + 1, n_replicates))
        # Make a new net (calling model() makes a new initialization of weights)
        net = model()

        # initialize weights based on limits that scale with number of in- and
        # outputs to the layer, increasing the chance that we converge to
        # a good solution
        torch.nn.init.xavier_uniform_(net[0].weight)
        torch.nn.init.xavier_uniform_(net[2].weight)

        # We can optimize the weights by means of stochastic gradient descent
        # The learning rate, lr, can be adjusted if training doesn't perform as
        # intended try reducing the lr. If the learning curve hasn't converged
        # (i.e. "flattend out"), you can try try increasing the maximum number of
        # iterations, but also potentially increasing the learning rate:
        # optimizer = torch.optim.SGD(net.parameters(), lr = 5e-3)

        # A more complicated optimizer is the Adam-algortihm, which is an extension
        # of SGD to adaptively change the learing rate, which is widely used:
        optimizer = torch.optim.Adam(net.parameters())

        # Train the network while displaying and storing the loss
        print('\t\t{}\t{}\t\t\t{}'.format('Iter', 'Loss', 'Rel. loss'))
        learning_curve = []  # setup storage for loss at each step
        old_loss = 1e6
        for i in range(max_iter):
            y_est = net(X)  # forward pass, predict labels on training set
            loss = loss_fn(y_est, y)  # determine loss
            loss_value = loss.data.numpy()  # get numpy array instead of tensor
            learning_curve.append(loss_value)  # record loss for later display

            # Convergence check, see if the percentual loss decrease is within
            # tolerance:
            p_delta_loss = np.abs(loss_value - old_loss) / old_loss
            if p_delta_loss < tolerance: break
            old_loss = loss_value

            # display loss with some frequency:
            if (i != 0) & ((i + 1) % logging_frequency == 0):
                print_str = '\t\t' + str(i + 1) + '\t' + str(loss_value) + '\t' + str(p_delta_loss)
                print(print_str)
            # do backpropagation of loss and optimize weights
            optimizer.zero_grad();
            loss.backward();
            optimizer.step()

        # display final loss
        print('\t\tFinal loss:')
        print_str = '\t\t' + str(i + 1) + '\t' + str(loss_value) + '\t' + str(p_delta_loss)
        print(print_str)

        if loss_value < best_final_loss:
            best_net = net
            best_final_loss = loss_value
            best_learning_curve = learning_curve

    # Return the best curve along with its final loss and learing curve
    return best_net, best_final_loss, best_learning_curve


class Model(nn.Module):
    def __init__(self, D_in, h, D_out, init_value=1e1):
        super().__init__()
        self.m = nn.Sequential(
            nn.Linear(D_in, h),
            nn.ReLU(),
            nn.Linear(h, D_out),
            nn.ReLU()
        )
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def __getitem__(self, idx):
        return self.m[idx]

    def forward(self, input):
        output = self.m(input)
        return output * self.scale


# Load in the ozone data
df = pd.read_csv('LA_Ozone.csv')

# Normalize the data to achieve standard devation 1 and mean 0
y_df = df.ozone  # Do not normalize the y
y = y_df.values
X_df = df.drop(columns=['ozone', 'doy'])
norm_X_df = (X_df - X_df.mean()) / X_df.std()
# Add the intercept into the X values
X = np.concatenate((np.ones((norm_X_df.shape[0], 1)), norm_X_df.values), 1)

# Outer fold
K1 = 10
k1 = 0
CV1 = model_selection.KFold(n_splits=K1, shuffle=True)
results = pd.DataFrame(np.empty((K1, 5)),
                       columns=['ANN_h[i]*', 'ANN_Test_Err', 'LinR_lambda[i]*', 'LinReg_Test_Err', 'BL_Test_Err'])
p_value = pd.DataFrame(np.empty((K1, 3)), columns=['p_ANNvsLR', 'p_ANNvsBL', 'p_LRvsBL'])
for train_idx1, test_idx1 in CV1.split(X):
    X_train1, y_train1 = X[train_idx1], y[train_idx1]
    X_test1, y_test1 = X[test_idx1], y[test_idx1]
    # Finding lambda
    opt_lamb = None
    min_lamb_err = float('inf')
    for lamb in np.power(10., range(-5, 9)):
        K2 = 10
        k2 = 0
        CV2 = model_selection.KFold(n_splits=K2, shuffle=True)
        test_error = np.empty(K2)
        for train_idx2, test_idx2 in CV2.split(X_train1):
            X_train2, y_train2 = X_train1[train_idx2], y_train1[train_idx2]
            X_test2, y_test2 = X_train1[test_idx2], y_train1[test_idx2]
            # Optimal weights with regularization parameter lambda are given by (X.T@X + lamb@I)^-1 @ (X.T@y)
            w = inv(X_train2.T @ X_train2 + lamb * np.identity(X_train2.shape[1])) @ (X_train2.T @ y_train2)
            y_pred = X_test2 @ w
            # MSE
            test_error[k2] = ((y_pred - y_test2) ** 2).mean()
            k2 += 1
        err = test_error.mean()
        if err < min_lamb_err:
            opt_lamb = lamb
            min_lamb_err = err
    # Finding h
    opt_h = None
    min_h_err = float('inf')
    for h in np.linspace(start=1, stop=20, num=20):
        K2 = 10
        k2 = 0
        CV2 = model_selection.KFold(n_splits=K2, shuffle=True)
        # Train the neural net
        model = lambda: Model(X_train2.shape[1], int(h), 1)
        loss_fn = nn.MSELoss()
        max_iter = 50
        for train_idx2, test_idx2 in CV2.split(X_train1):
            X_train2 = torch.tensor(X_train1[train_idx2], dtype=torch.float)
            y_train2 = torch.tensor(y_train1[train_idx2], dtype=torch.float)
            X_test2 = torch.tensor(X_train1[test_idx2], dtype=torch.float)
            y_test2 = torch.tensor(y_train1[test_idx2], dtype=torch.float)
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train2,
                                                               y=y_train2,
                                                               n_replicates=1,
                                                               max_iter=max_iter)
            y_pred = net(X_test2)
            err = ((y_pred - y_test2) ** 2).mean()
            if err < min_h_err:
                opt_h = h
                min_h_err = err
            k2 += 1
    # Find error of optimal lambda on outer fold
    w = inv(X_train1.T @ X_train1 + opt_lamb * np.identity(X_train1.shape[1])) @ (X_train1.T @ y_train1)
    y_pred = X_test1 @ w
    # MSE
    lamb_err = ((y_pred - y_test1) ** 2).mean()
    lamb_err_all = np.abs(y_pred - y_test1) ** 2
    # Find error of optimal h on outer fold
    # Train the neural net
    model = lambda: Model(X_train2.shape[1], int(h), 1)
    loss_fn = nn.MSELoss()
    max_iter = 50
    X_train_torch = torch.tensor(X_train1, dtype=torch.float)
    y_train_torch = torch.tensor(y_train1, dtype=torch.float)
    X_test_torch = torch.tensor(X_test1, dtype=torch.float)
    y_test_torch = torch.tensor(y_test1, dtype=torch.float)
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train_torch,
                                                       y=y_train_torch,
                                                       n_replicates=1,
                                                       max_iter=max_iter)
    # MSE
    h_err = ((net(X_test_torch) - y_test_torch) ** 2).mean().item()
    h_err_all = list(((y_test_torch - net(X_test_torch)) ** 2))
    # Baseline
    y_mean = y_train1.mean()
    bl_err = ((y_mean * np.ones(len(y_test1)) - y_test1) ** 2).mean()
    bl_err_all = np.abs(y_mean * np.ones(len(y_test1)) - y_test1) ** 2
    # Populate the dataframe
    results.iloc[k1] = np.array([opt_h, h_err, opt_lamb, lamb_err, bl_err])
    k1 += 1

print(results)

models = {'ANN': 'ANN_Test_Err', 'LinReg': 'LinReg_Test_Err', 'Baseline': 'BL_Test_Err'}
comp_df = pd.DataFrame()
for model1, model2 in combinations(models, 2):
    results[[models[model1], models[model2]]].plot(kind='box')
    model1_err = results[models[model1]]
    model2_err = results[models[model2]]
    diff = model1_err.values - model2_err.values
    upper = diff.mean() + stats.t.ppf(1 - 0.025, model1_err.shape[0] - 1) * np.std(diff, ddof=1)
    lower = diff.mean() - stats.t.ppf(1 - 0.025, model1_err.shape[0] - 1) * np.std(diff, ddof=1)
    comp_df['%s vs. %s' % (model1, model2)] = np.array([lower, upper])
comp_df = comp_df.rename(index={0: 'lower_0.025', 1: 'upper_0.975'})
comp_df

df = pd.read_csv('LA_ozone.csv')

# Normalize the data to achieve standard devation 1 and mean 0
y_df = df.ozone # Do not normalize the y
X_df = df.drop(columns=['ozone', 'doy'])
norm_X_df = (X_df - X_df.mean()) / X_df.std()
# Add the intercept into the X values
X = np.concatenate((np.ones((norm_X_df.shape[0], 1)), norm_X_df.values), 1)

# Transform the y values into one-out-of-k encoding
# 3 categories
low = int(y_df.mean() - y_df.std())
high = int(y_df.mean() + y_df.std())
new_data = pd.DataFrame()
new_data['low'] = (y_df <= low).astype('float')
new_data['medium'] = (np.logical_and(y_df > low, y_df <= high)).astype('float')
new_data['high'] = (y_df > high).astype('float')
print(y)
y = new_data @ [0, 1, 2]
y = y.values


import scipy.stats
import scipy.stats as st
def mcnemar(y_true, yhatA, yhatB, alpha=0.05):
    # perform McNemars test
    nn = np.zeros((2,2))
    c1 = yhatA - y_true == 0
    c2 = yhatB - y_true == 0

    nn[0,0] = sum(c1 & c2)
    nn[0,1] = sum(c1 & ~c2)
    nn[1,0] = sum(~c1 & c2)
    nn[1,1] = sum(~c1 & ~c2)

    n = sum(nn.flat);
    n12 = nn[0,1]
    n21 = nn[1,0]

    thetahat = (n12-n21)/n
    Etheta = thetahat

    Q = n**2 * (n+1) * (Etheta+1) * (1-Etheta) / ( (n*(n12+n21) - (n12-n21)**2) )

    p = (Etheta + 1) * (Q-1)
    q = (1-Etheta) * (Q-1)

    CI = tuple(lm * 2 - 1 for lm in scipy.stats.beta.interval(1-alpha, a=p, b=q) )

    p = 2*scipy.stats.binom.cdf(min([n12,n21]), n=n12+n21, p=0.5)
    """print("Result of McNemars test using alpha=", alpha)
    print("Comparison matrix n")
    print(nn)
    if n12+n21 <= 10:
        print("Warning, n12+n21 is low: n12+n21=",(n12+n21))

    print("Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] = ", CI)
    print("p-value for two-sided test A and B have same accuracy (exact binomial test): p=", p)

    thetahat = 2*thetahat-1"""
    return thetahat, CI, p


import sklearn.linear_model as lm
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
from sklearn import model_selection
from itertools import combinations

K1 = 5
k1 = 0
CV1 = model_selection.KFold(n_splits=K1, shuffle=True)
results = pd.DataFrame(np.empty((K1, 5)),
                       columns=['LogR_Lambda[i]*', 'LogR_Test_Err', 'DT_Max_Depth[i]*', 'DT_Test_Err', 'BL_Test_Err'])

p_values = pd.DataFrame(columns=['LRvsDT', 'LRvsBL', 'DTvsBL'])
p_values_last = pd.DataFrame(columns=['LRvsDT', 'LRvsBL', 'DTvsBL'])

for train_idx1, test_idx1 in CV1.split(X):
    X_train1, y_train1 = X[train_idx1], y[train_idx1]
    X_test1, y_test1 = X[test_idx1], y[test_idx1]
    # Finding optimal lambda
    opt_lamb = None
    min_lamb_err = float('inf')
    for lamb in np.power(10., range(-5, 9)):
        K2 = 10
        k2 = 0
        CV2 = model_selection.KFold(n_splits=K2, shuffle=True)
        test_error = np.empty(K2)
        for train_idx2, test_idx2 in CV2.split(X_train1):
            X_train2, y_train2 = X_train1[train_idx2], y_train1[train_idx2]
            X_test2, y_test2 = X_train1[test_idx2], y_train1[test_idx2]
            # Logistic regression
            logreg = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', C=1 / lamb, max_iter=1000)
            logreg.fit(X_train2, y_train2)
            y_pred = logreg.predict(X_test2)

            # YPRED 1

            # Percent wrong
            test_error[k2] = (y_pred != y_test2).sum() / y_pred.shape[0]
            k2 += 1
        err = test_error.mean()
        if err < min_lamb_err:
            opt_lamb = lamb
            min_lamb_err = err
    # Finding max depth
    opt_md = None
    min_md_err = float('inf')
    for md in np.arange(3, 25, 1):
        K2 = 10
        k2 = 0
        CV2 = model_selection.KFold(n_splits=K2, shuffle=True)
        for train_idx2, test_idx2 in CV2.split(X_train1):
            X_train2, y_train2 = X_train1[train_idx2], y_train1[train_idx2]
            X_test2, y_test2 = X_train1[test_idx2], y_train1[test_idx2]
            dtc = DecisionTreeClassifier(criterion='gini', max_depth=md)
            dtc = dtc.fit(X_train2, y_train2)
            y_pred = dtc.predict(X_test2)

            # YPRED 2

            # Percent wrong
            test_error[k2] = (y_pred != y_test2).sum() / y_pred.shape[0]
            k2 += 1
        err = test_error.mean()
        if err < min_md_err:
            opt_md = md
            min_md_err = err

    # Find error of optimal lambda on outer fold
    logreg = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', C=1 / opt_lamb, max_iter=1000)
    logreg.fit(X_train1, y_train1)
    y_pred_LR = logreg.predict(X_test1)

    # YPRED 1

    # Percent wrong
    lamb_err = (y_pred_LR != y_test1).sum() / y_pred.shape[0]

    # Find error of optimal max depth on outer fold
    dtc = DecisionTreeClassifier(criterion='gini', max_depth=opt_md)
    dtc = dtc.fit(X_train1, y_train1)
    y_pred_DT = dtc.predict(X_test1)

    # YPRED 2

    md_err = (y_pred_DT != y_test1).sum() / y_pred.shape[0]
    # Baseline
    y_mode = stats.mode(y_train1)

    # YPRED 3
    y_pred_BL = np.full(len(y_test1), y_mode.mode)

    bl_err = (y_test1 != np.full(len(y_test1), y_mode.mode)).sum() / y_test1.shape[0]

    DF = pd.DataFrame({'LogReg': y_pred_LR, 'DecisionTree': y_pred_DT, 'Baseline': y_pred_BL, 'TestData': y_test1})

    alpha = 0.05
    [thetahat_1, CI_1, p_1] = mcnemar(DF.TestData, DF.LogReg, DF.DecisionTree, alpha=alpha)
    [thetahat_2, CI_2, p_2] = mcnemar(DF.TestData, DF.LogReg, DF.Baseline, alpha=alpha)
    [thetahat_3, CI_3, p_3] = mcnemar(DF.TestData, DF.DecisionTree, DF.Baseline, alpha=alpha)

    p_values = pd.DataFrame(
        {'LRvsDT': [thetahat_1, CI_1, p_1], 'LRvsBL': [thetahat_2, CI_2, p_2], 'DTvsBL': [thetahat_3, CI_3, p_3]})
    p_values_last = p_values_last.append(p_values)
    # Populate the dataframe
    results.iloc[k1] = np.array([opt_lamb, lamb_err, opt_md, md_err, bl_err])
    k1 += 1

results

p_values_last = p_values_last.rename(index={0: 'ThetaHat', 1: 'C.I.', 2:'P-values', 3: 'ThetaHat', 4: 'C.I.', 5:'P-values', 6: 'ThetaHat', 7: 'C.I.', 8:'P-values', 9: 'ThetaHat', 10: 'C.I.', 11:'P-values', 12: 'ThetaHat', 13: 'C.I.', 14:'P-values'})
p_values_last.reset_index(level=0, inplace=True)
p_values_last = p_values_last.rename(columns = {"index" : "Statistics"})
p_values_last = p_values_last.rename(index={0: 'OuterFold-1', 1: 'OuterFold-1', 2:'OuterFold-1', 3: 'OuterFold-2', 4: 'OuterFold-2', 5:'OuterFold-2', 6: 'OuterFold-3', 7: 'OuterFold-3', 8:'OuterFold-3', 9: 'OuterFold-4', 10: 'OuterFold-4', 11:'OuterFold-4', 12: 'OuterFold-5', 13: 'OuterFold-5', 14:'OuterFold-5'})
p_values_last


from numpy.linalg import inv
import sklearn.linear_model as lm
import scipy.spatial.distance as dist

# Load in the ozone data
df = pd.read_csv('LA_ozone.csv')

# Normalize the data to achieve standard devation 1 and mean 0
y_df = df.ozone # Do not normalize the y
y = y_df.values
X_df = df.drop(columns=['ozone', 'doy'])
norm_X_df = (X_df - X_df.mean()) / X_df.std()
# Add the intercept into the X values
X = np.concatenate((np.ones((norm_X_df.shape[0], 1)), norm_X_df.values), 1)
# Optimal weights with regularization parameter lambda are given by (X.T@X + lamb@I)^-1 @ (X.T@y)
lin_reg_lamb = 1e1
lin_reg_w = inv(X.T @ X + lin_reg_lamb * np.identity(X.shape[1])) @ (X.T @ y)

# Transform the y values into one-out-of-k encoding
# 3 categories
low = int(y_df.mean() - y_df.std())
high = int(y_df.mean() + y_df.std())
new_data = pd.DataFrame()
new_data['low'] = (y_df <= low).astype('float')
new_data['medium'] = (np.logical_and(y_df > low, y_df <= high)).astype('float')
new_data['high'] = (y_df > high).astype('float')
y = new_data @ [0, 1, 2]
y = y.values
# Logistic regression
log_reg_lamb = 1e1
log_reg = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', C=1/log_reg_lamb, max_iter=1000)
log_reg = log_reg.fit(X, y)


lin_w = lin_reg_w[1:] # Remove the intercept
lin_w = np.abs(lin_w)
log_w = log_reg.coef_[:, 1:]
log_w = np.abs(log_w)

dist_sum = 0
for idx in range(log_w.shape[0]): # Multinomial
    dist_sum += dist.cosine(log_w[idx], lin_w)
dist_sum / log_w.shape[0]

for cname, val in zip(X_df.columns, lin_w.tolist()):
    print("%s=%.3f"%(cname, val))


log_w[0,:]
Log_W = pd.DataFrame({"LogReg_1": log_w[0,:], "LogReg_2": log_w[1,:], "LogReg_3": log_w[2,:]})
Log_W = Log_W.T
new_names = {
    0 : "vh",
    1 : "ibh",
    2 : "dpg",
    3 : "ibt",
    4 : "vis",
    5 : "wind",
    6 : "humidity",
    7 : "temp"
}
Log_W.rename(columns=new_names, inplace=True)
Log_W



Lin_W = pd.DataFrame({"Lin_Reg": lin_w})
Lin_W = Lin_W.T
new_names = {
    0 : "vh",
    1 : "ibh",
    2 : "dpg",
    3 : "ibt",
    4 : "vis",
    5 : "wind",
    6 : "humidity",
    7 : "temp"
}
Lin_W.rename(columns=new_names, inplace=True)
Lin_W

