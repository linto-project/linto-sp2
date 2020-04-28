"""stats module for computing statistic metrics.

This module contains functions for computing statistic metrics for features and signal aggregation and modeling.

"""


import numpy as np
from scipy import stats
import speechTools as st


def computePercentiles( values ):
	percentilesPositions = [0,25,50,75,100]
	min, q1, median, q3, max = np.percentile( values, percentilesPositions)
	return min, q1, median, q3, max

def computeDistributionParams( values ):
	mean = np.mean( values )
	standardDeviation = np.std( values)
	mad = stats.median_absolute_deviation( values)
	return mean, standardDeviation, mad


def computeDistributionShape( signal ):
	kurtosis = stats.kurtosis( signal, fisher=True)
	skewness = stats.skew( signal)
	return kurtosis, skewness


