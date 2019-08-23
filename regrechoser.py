import pandas as pd
import numpy as np
import math
from scipy.stats import f,t


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


def get_sse(estimation, Y):
    '''
    Returns the part of the variance explained by the model.
    '''
    sse = sum([(estimation[i]-(sum(Y)/len(Y)))**2 for i in range(len(Y))])
    return(sse)


def get_ssr(estimation, Y):
    '''
    Returns the part of the variance that can not be explained by the model.
    '''
    ssr = sum([(Y[i]-estimation[i])**2 for i in range(len(Y))])
    return(ssr)


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


def get_y_residual_variance(ssr, n, k):
    '''
    Returns y residual variance.
    '''
    residual_yvariance = ssr/(n-k-1)
    return(residual_yvariance)


def get_ssrh0(Y, B):
        ssrh0 = float(sum([(Y[i]-B[0])**2 for i in range(len(Y))]))
        return(ssrh0)

def get_sigmasq(ssr, n):
        sigmasq = ssr/(n-2)
        return(sigmasq)


def f_contrast(ssr, ssrh0, n, k):
    '''
    Returns p-value for F_contrast: H0: b0 = b1 = b2 = ... = bn = 0
    '''
    fvalue = ((ssrh0-ssr)/(k-1))/(ssr/(n-k))
    pvalue = f.pdf(fvalue, k-1, n-k)
    return(pvalue)

def t_contrast(estimator, se_estimator, n, k):
        tvalue = estimator/se_estimator
        pvalue = t.pdf(tvalue, n-k)
        return(float(pvalue))
        
        
        
def r2(sse, sst):
    r2 = (sse/sst)
    return(r2)


Y = data[y]
X = np.matrix(get_X_matrix(data, x))
var_matrix = np.linalg.inv(np.transpose(X)*X)

n = len(Y)
k = len(x)+1

B = get_estimators(X, Y)
estimation = get_estimation(B, X)

ssr = get_ssr(estimation=estimation, Y=Y)
sse = get_sse(estimation, Y)
sst = get_sst(Y)

var = get_y_variance(sst, n)
residual_var = get_y_residual_variance(sst, n, k)

ssrh0 = get_ssrh0(Y, B)
fvalue = f_contrast(ssr, ssrh0, n, k)
r2 = r2(sse, sst)

sigmasq = get_sigmasq(ssr, n)
sigma = math.sqrt(sigmasq)


var_estimators = [(var_matrix[i,i]*sigmasq) for i in range(len(x)+1)]
se_estimators = [math.sqrt(var) for var in var_estimators]

tvalues = [t_contrast(estimator=B[i], se_estimator=se_estimators[i], n=n, k=k) for i in range(len(se_estimators))]

