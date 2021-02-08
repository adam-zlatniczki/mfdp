# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 14:36:51 2018

@author: adam.zlatniczki
"""

import numpy as np
import pandas as pd

from sklearn.covariance import MinCovDet
import cvxopt
from cvxopt.solvers import qp
from cvxopt import matrix


df = pd.read_csv("sp100.csv", index_col=0, parse_dates=True, header=0)


# set random generator for repeatable results
np.random.seed(0)

df.drop("GOOG", axis=1, inplace=True)
df.drop("FOX", axis=1, inplace=True)

# drop columns with missing values
prices = df.iloc[:,].dropna(axis=1)

returns = (prices.shift() / prices)[1:]


train_window = 1000
test_window = 500

train_data_start = 250
train_data_end = train_data_start + train_window
test_data_end = train_data_end + test_window


""" Analyze """
from robust_hurst import robust_hurst
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

cvxopt.solvers.options["show_progress"] = False


### Markowitz long-only
N = returns.shape[1]

p = matrix(np.zeros((N,1)))
G = -matrix(np.eye(N))
h = matrix(np.zeros((N,1)))
A = matrix(np.ones((1,N)))
b = matrix(1.0)


results_df = pd.DataFrame()

for train_window in range(150, 1000 + 1, 10):
    for train_data_start in range(0, returns.shape[0] - train_window + 1, 50):
        train_data_end = train_data_start + train_window

        R_train = np.asarray(returns.iloc[train_data_start:train_data_end,:])        
        
        n = R_train.shape[0]
        N = R_train.shape[1]
        
        mcd = MinCovDet()
        mcd.fit(R_train)
        S = mcd.covariance_
        
        
        """ Markowitz long-only """
        Q = matrix(2*S)
        
        sol = qp(Q, p, G, h, A, b)
        w_opt_M = np.reshape(sol["x"], N)
        
        """ Calculate statistics """
        # Train
        cum_returns_train = np.asarray((prices.iloc[train_data_start:train_data_end,:] - prices.iloc[train_data_start,:])[1:])
        train_returns = np.dot(cum_returns_train, w_opt_M)
        
        H_train = robust_hurst(train_returns)
        lr.fit(np.arange(train_returns.shape[0]).reshape(-1,1), train_returns.reshape(-1,1))
        y = lr.predict(np.arange(train_returns.shape[0]).reshape(-1,1))
        r2_train = r2_score(train_returns.reshape(-1,1), y)
        
        std_train = np.std(train_returns[1:] / train_returns[:-1])
    
        # Test
        for test_window in range(train_window // 10, train_window // 2 + 1, 10):
            print(train_window, train_data_start, test_window)
            
            test_data_end = train_data_end + test_window
            
            R_test = np.asarray(returns.iloc[train_data_end:test_data_end,:])
            cum_returns_test = np.asarray((prices.iloc[train_data_end:test_data_end,:] - prices.iloc[train_data_end,:])[1:])
            test_returns = np.dot(cum_returns_test, w_opt_M)
        
        
            H_test = robust_hurst(test_returns, test_returns.shape[0] // 2)
        
            lr.fit(np.arange(test_returns.shape[0]).reshape(-1,1), test_returns.reshape(-1,1))
            y = lr.predict(np.arange(test_returns.shape[0]).reshape(-1,1))
            r2_test = r2_score(test_returns.reshape(-1,1), y)
            
            std_test = np.std(test_returns[1:] / test_returns[:-1])
            
            row = pd.DataFrame([[train_window, train_data_start, test_window, H_train, r2_train, std_train, H_test, r2_test, std_test]])
            
            results_df = pd.concat([results_df, row])

 
results_df.columns = ["train_window", "train_data_start", "test_window", "H_train", "r2_train", "std_train", "H_test", "r2_test", "std_test"]
results_df.index = range(results_df.shape[0])
results_df.to_csv("results_longonly.csv")


### Markowitz long-short
results_df = pd.DataFrame()


for train_window in range(150, 1000 + 1, 10):
    for train_data_start in range(0, returns.shape[0] - train_window + 1, 50):
        train_data_end = train_data_start + train_window

        R_train = np.asarray(returns.iloc[train_data_start:train_data_end,:])        
        
        n = R_train.shape[0]
        N = R_train.shape[1]
        
        mcd = MinCovDet()
        mcd.fit(R_train)
        S = mcd.covariance_

        
        """ Markowitz long-short """
        A = np.zeros((N+1,N+1))
        A[:N,:N] = 2*S
        A[:N,-1] = np.ones(N)
        A[-1,:N] = np.ones(N)
        
        b = np.zeros((N+1,1))
        b[-1,0] = 1
        
        x = np.dot(np.linalg.inv(A), b)
        w_opt_M = x[:N].reshape(N)

        
        """ Calculate statistics """
        # Train
        cum_returns_train = np.asarray((prices.iloc[train_data_start:train_data_end,:] - prices.iloc[train_data_start,:])[1:])
        train_returns = np.dot(cum_returns_train, w_opt_M)
        
        H_train = robust_hurst(train_returns)
        lr.fit(np.arange(train_returns.shape[0]).reshape(-1,1), train_returns.reshape(-1,1))
        y = lr.predict(np.arange(train_returns.shape[0]).reshape(-1,1))
        r2_train = r2_score(train_returns.reshape(-1,1), y)
        
        std_train = np.std(train_returns[1:] / train_returns[:-1])
    
        # Test
        for test_window in range(train_window // 10, train_window // 2 + 1, 10):
            print(train_window, train_data_start, test_window)
            
            test_data_end = train_data_end + test_window
            
            R_test = np.asarray(returns.iloc[train_data_end:test_data_end,:])
            cum_returns_test = np.asarray((prices.iloc[train_data_end:test_data_end,:] - prices.iloc[train_data_end,:])[1:])
            test_returns = np.dot(cum_returns_test, w_opt_M)
        
        
            H_test = robust_hurst(test_returns, test_returns.shape[0] // 2)
        
            lr.fit(np.arange(test_returns.shape[0]).reshape(-1,1), test_returns.reshape(-1,1))
            y = lr.predict(np.arange(test_returns.shape[0]).reshape(-1,1))
            r2_test = r2_score(test_returns.reshape(-1,1), y)
            
            std_test = np.std(test_returns[1:] / test_returns[:-1])
            
            row = pd.DataFrame([[train_window, train_data_start, test_window, H_train, r2_train, std_train, H_test, r2_test, std_test]])
            
            results_df = pd.concat([results_df, row])

 
results_df.columns = ["train_window", "train_data_start", "test_window", "H_train", "r2_train", "std_train", "H_test", "r2_test", "std_test"]
results_df.index = range(results_df.shape[0])
results_df.to_csv("results_longshort.csv")