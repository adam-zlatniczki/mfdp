# mfdp
Scripts and data that support the article "On the relationship of minimum variance and minimum fractal dimension portfolios".

- SP100.csv: daily adjusted close prices between 2007-07-31 and 2017-07-31 of assets that were listed in the S\&P100 index on 2017-07-31, obtained from Yahoo! Finance
- markowitz_memory.py: the script that trains and tests the Markowitz portfolios, produces results_longshort.csv and results_longonly.csv
- statistical_analysis.py: processes results_longshort.csv and results_longonly.csv, producing the figures and the statistical tests
- robust_hurst.py: an implementation of Hurst exponent calculation, with robust statistics
