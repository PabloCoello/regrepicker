import pandas as pd
import numpy as np
import math
from scipy.stats import f


data = pd.read_excel('../encuestas.xlsx')
y = "PSOE"
x = ["VOX_busc", "Ciudadanos_busc"]


def get_X_matrix(data, x):
        X = data[x].assign(a=1)
        cols = X.columns.tolist()
        cols = cols[-1:]+cols[:-1]
        X = X[cols]
        return(X)


def get_estimators(X, Y):
    '''
    Returns OLS estimators array.
    '''
    B = (np.linalg.inv(np.transpose(X)*X) * 
        np.transpose(X) * 
        np.transpose(np.matrix(Y)))
    return(B)  


def get_estimation(B, X):
    '''
    Returns the values estimated by the model of the explained variable.
    '''
    estimation = [float(np.transpose(B)*np.transpose(line)) for line in X]
    return(estimation)


def get_ssr(estimation, Y):
    '''
    Returns the part of the variance explained by the model.
    '''
    ssr = sum([(estimation[i]-(sum(Y)/len(Y)))**2 for i in range(len(Y))])
    return(ssr)


def get_sse(estimation, Y):
    '''
    Returns the part of the variance that can not be explained by the model.
    '''
    sse = sum([(Y[i]-estimation[i])**2 for i in range(len(Y))])
    return(sse)


def get_sst(Y):
    '''
    Returns total variation of dependent variable.
    '''
    sst = sum([(Y[i]-(sum(Y)/len(Y)))**2 for i in range(len(Y))])
    return(sst)


def get_y_variance(sst, n):
    '''
    Returns y variance.
    '''
    yvariance = sst/(n-1)
    return(yvariance)


def get_y_residual_variance(sse, n, k):
    '''
    Returns y residual variance.
    '''
    residual_yvariance = sse/(n-k-1)
    return(residual_yvariance)


def get_sseh0(Y, B):
        sseh0 = float(sum([(Y[i]-B[0])**2 for i in range(len(Y))]))
        return(sseh0)

def get_sigmasq(ssr, n):
        sigmasq = ssr/(n-2)
        return(sigmasq)

def get_se_estimators(sigmasq, est):
        se = math.sqrt(sigmasq)/ \
                math.sqrt(sum([elem-(sum(est)/len(est) for elem in est)]))
        return(se)

def f_contrast(sse, sseh0, n, k):
    '''
    Returns p-value for F_contrast: H0: b0 = b1 = b2 = ... = bn = 0
    '''
    fvalue = ((sseh0-sse)/(k-1))/(sse/(n-k))
    pvalue = f.pdf(fvalue, k-1, n-k)
    return(pvalue)

#def t_contrast():
        

def r2(sse, sst):
    r2 = 1-(sse/sst)
    return(r2)


Y = data[y]
X = np.matrix(get_X_matrix(data, x))

n = len(Y)
k = len(x)+1

B = get_estimators(X, Y)
estimation = get_estimation(B, X)

ssr = get_ssr(estimation=estimation, Y=Y)
sse = get_sse(Y, estimation)
sst = get_sst(Y)

var = get_y_variance(sst, n)
residual_var = get_y_residual_variance(sst, n, k)

vneh0 = get_sseh0(Y, B)
fvalue = f_contrast(vne, vneh0, n, k)
r2 = r2(sse, sst)

sigmasq = get_sigmasq(ssr, n)
sigma = math.sqrt(sigmasq)
