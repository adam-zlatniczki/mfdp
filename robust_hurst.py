# -*- coding: utf-8 -*-
"""
Created on Wed Oct 03 14:56:04 2018

@author: adam.zlatniczki
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.covariance import MinCovDet
from scipy.special import comb


def robust_hurst(ts, lags=100, robust_cov=True, plot=False):
    
    minCovDet = MinCovDet(assume_centered=True)
    n = ts.shape[0]
    
    # calculate lagged variances
    var_lags = np.zeros(lags-1)
    
    for lag in range(1,lags):
        lagged_series = ts[lag:]-ts[:-lag]
        
        if robust_cov:
            minCovDet.fit(lagged_series.reshape(-1,1))
            var_lags[lag-1] = np.asscalar(minCovDet.covariance_)
        else:
            var_lags[lag-1] = np.dot(lagged_series, lagged_series) / (n - lag - 1)
    
    # calculate log-log slopes
    slopes = np.zeros(int(comb(lags-2,2)))
    cntr = 0
    for i in range(1,lags-1):
        for j in range(i+1,lags-1):
            slopes[cntr] = np.log(var_lags[j] / var_lags[i]) / (2 * np.log(float(j) / i))
            cntr += 1
    
    H_est = np.median(slopes)
    
    # plot
    if plot:
        plt.figure()
        plt.hist(slopes)
        
        plt.figure()
        plt.plot(np.log(range(1,lags)), np.log(var_lags))
        plt.plot(np.log(range(1,lags)), np.log(range(1,lags))*H_est)
        
    return np.median(slopes)