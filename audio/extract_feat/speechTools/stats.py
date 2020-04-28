"""stats module for computing statistic metrics.

This module contains functions for computing statistic metrics for features and signal aggregation and modeling.

"""


import numpy as np
from scipy import stats
import speechTools as st


def computePercentiles( values ):
	"""Compute percentiles of a set of values

	This function computes 5 percentile metrics of a set of values
	The metrics are respectively: minimum(min), first quintile(q1), median(med), third quintile(q3), maximum(max).

	Args:
		values (list): The set of values

	Returns:
		float, float, float, float, float: Min, q1, med, q3, max

	"""

	percentilesPositions = [0,25,50,75,100]
	min, q1, median, q3, max = np.percentile( values, percentilesPositions)
	return min, q1, median, q3, max

def computeDistributionParams( values ):
	"""Compute respectively the mean, standard deviation and median absolute deviation of a set of values

	Args:
		values (list): The set of values

	Returns:
	float, float, float: Mean, STD, MAD

	"""

	mean = np.mean( values )
	standardDeviation = np.std( values)
	mad = stats.median_absolute_deviation( values)
	return mean, standardDeviation, mad


def computeDistributionShape( values ):
	"""Compute respectively the Fisher kurtosis and  skewness of a set of values

	Args:
		values (list): The set of values

	Returns:
	float, float: kurtosis, skewness

	"""

	kurtosis = stats.kurtosis( values, fisher=True)
	skewness = stats.skew( values)
	return kurtosis, skewness


