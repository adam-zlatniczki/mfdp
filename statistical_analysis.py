# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 22:12:46 2018

@author: adam.zlatniczki
"""

"""
Statistical Analysis
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

results_df_ls = pd.read_csv(r"results_longshort.csv", index_col=0)
results_df_lo = pd.read_csv(r"results_longonly.csv", index_col=0)


""" Trim outliers """

### Long-short
mask = np.logical_and(np.percentile(results_df_ls.std_train, 5) < results_df_ls.std_train, results_df_ls.std_train < np.percentile(results_df_ls.std_train, 95))
mask = np.logical_and(mask, 0 < results_df_ls.H_train)
mask = np.logical_and(mask, results_df_ls.H_train < 1)

mask = np.logical_and(mask, np.percentile(results_df_ls.std_test, 5) < results_df_ls.std_test)
mask = np.logical_and(mask, results_df_ls.std_test < np.percentile(results_df_ls.std_test, 95))
mask = np.logical_and(mask, 0 < results_df_ls.H_test)
mask = np.logical_and(mask, results_df_ls.H_test < 1)

trimmed_df_ls = results_df_ls[mask]


### Long-only
mask = np.logical_and(np.quantile(results_df_lo.std_train, 0.05) < results_df_lo.std_train, results_df_lo.std_train < np.quantile(results_df_lo.std_train, 0.95))
mask = np.logical_and(mask, 0 < results_df_lo.H_train)
mask = np.logical_and(mask, results_df_lo.H_train < 1)

mask = np.logical_and(mask, np.quantile(results_df_lo.std_test, 0.05) < results_df_lo.std_test)
mask = np.logical_and(mask, results_df_lo.std_test < np.quantile(results_df_lo.std_test, 0.95))
mask = np.logical_and(mask, 0 < results_df_lo.H_test)
mask = np.logical_and(mask, results_df_lo.H_test < 1)

trimmed_df_lo = results_df_lo[mask]



""" Hypothesis tests """

from scipy.stats import pearsonr, ttest_rel

### Long-short

# how things depend on each other overall
aggr_df = trimmed_df_ls.groupby(["train_window", "test_window"]).mean().reset_index()
C = aggr_df.corr(method="pearson")

pearsonr(aggr_df["H_train"], aggr_df["r2_train"])
pearsonr(aggr_df["H_train"], aggr_df["std_train"])
pearsonr(aggr_df["r2_train"], aggr_df["std_train"])


pearsonr(aggr_df["H_test"], aggr_df["r2_test"])
pearsonr(aggr_df["H_test"], aggr_df["std_test"])
pearsonr(aggr_df["r2_test"], aggr_df["std_test"])


pearsonr(aggr_df["std_train"], aggr_df["H_test"])
pearsonr(aggr_df["std_train"], aggr_df["r2_test"])

# How things depend on the train window
aggr_df = trimmed_df_ls.groupby(["train_window"]).mean().reset_index()
C = aggr_df.corr(method="pearson")

pearsonr(aggr_df["train_window"], aggr_df["H_test"])
pearsonr(aggr_df["train_window"], aggr_df["r2_test"])


### Long-only


# how things depend on each other overall
aggr_df = trimmed_df_lo.groupby(["train_window", "test_window"]).mean().reset_index()
C = aggr_df.corr(method="pearson")


pearsonr(aggr_df["H_test"], aggr_df["r2_test"])
pearsonr(aggr_df["H_test"], aggr_df["std_test"])
pearsonr(aggr_df["r2_test"], aggr_df["std_test"])


pearsonr(aggr_df["std_train"], aggr_df["H_test"])
pearsonr(aggr_df["std_train"], aggr_df["r2_test"])


# How things depend on the train window
aggr_df = trimmed_df_lo.groupby(["train_window"]).mean().reset_index()
C = aggr_df.corr(method="pearson")

pearsonr(aggr_df["train_window"], aggr_df["H_test"])
pearsonr(aggr_df["train_window"], aggr_df["r2_test"])


# LS vs LO
aggr_df_ls = trimmed_df_ls.groupby(["train_window", "test_window"]).mean().reset_index()
aggr_df_lo = trimmed_df_lo.groupby(["train_window", "test_window"]).mean().reset_index()

aggr_df_ls.H_test.mean() - aggr_df_lo.H_test.mean()
ttest_rel(aggr_df_ls.H_test, aggr_df_lo.H_test)

aggr_df_ls.r2_test.mean() - aggr_df_lo.r2_test.mean()
ttest_rel(aggr_df_ls.r2_test, aggr_df_lo.r2_test)


""" Figures """
import seaborn as sns
import matplotlib.gridspec as gridspec

# Long-short

aggr_df = trimmed_df_ls.groupby(["train_window", "test_window"]).mean().reset_index()

ax = sns.jointplot(x=aggr_df.H_test, y=aggr_df.r2_test, kind='reg')
ax.set_axis_labels("H", "$R^2$")
ax.savefig("H_R2_test_ls.png", dpi=300)

# Long-only

aggr_df = trimmed_df_lo.groupby(["train_window", "test_window"]).mean().reset_index()

ax = sns.jointplot(x=aggr_df.H_test, y=aggr_df.r2_test, kind='reg')
ax.set_axis_labels("H", "$R^2$")
ax.savefig("H_R2_test_lo.png", dpi=300)