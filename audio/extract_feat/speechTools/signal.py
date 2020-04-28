"""signal modul for signal processing.

This module contains audio DSP functions for processing low level audio signals from wav files. It include as well general purpose functions for signal modeling or aggregation.

"""


import numpy as np
import scipy as sp
import speechTools as st


def quantify ( value, scale ):
	"""Choos the closest option to the value in the provided scale

	This function quantifies a continuous domain value by transforming it on the closest value in a provided discrete scale. If the initial value is out of  scale range, any transformation is applied.

	Args:
		value (float): The value to quantify
		scale (list): The discrete scale where the rectified value should be found

	Returns:
		float: The quantified value

	"""


	if value < scale[0] or value > scale[-1]: return value
	for i in range (1, len(scale)):
		if value <= scale[i]:
			if scale[i] - value > value - scale[i-1]: return scale[i-1]
			else: return scale[i]


def quantifyValues( values, scale):
	"""Rectify a values sequence  by quantifying each single value using the provided scale

	This function quantifies the continuous domain values of a signal by transforming each one of them on the closest value in a provided discrete scale. If the initial value is out of  scale range, any transformation is applied.

	Args:
		values (numpy.array): The values to quantify
		scale (list): The discrete scale where the rectified values should be found

	Returns:
		numpy.array: The rectified values (same object as entry is returned )

	"""

	for v in range(len(values)):
		values[v] = st.quantify( values[v], scale)
	return values

def computeSignalVariability( signal, sampleRate, order=1):
	"""Compute the variability indicator of a signal

	This function computes the signal variability defined as: SV = mean ( absolute ( K_derivative (signal))): with K the derivative order.
	This indicator showes how the signal is smooth or hardly zigzagging.

	Args:
		signal (numpy.array): The 1d signal
		sampleRate (int): The signal sample rate
		order (int): The derivative order (1 for velocity, 2 for acceleration)

	Returns:
		float: The variability indicator

	"""

	sampleSpacing = 1 / sampleRate
	absoluteDerivative = abs(np.diff( signal, n=order) / sampleSpacing)
	signalVariability = np.mean( absoluteDerivative )
	return signalVariability


def getSignalReader( signal, sampleRate=16000, windowWidth=0.032, step=0.01, withWindowIndex=False):
	"""Provide a generator for looping on a signal  through consecutive windows

	This function returns a python generator for looping on a signal. For each loop a single window is yielded. This way, any processing could be applied to the signal for each window through a simple for-loop.
	If the  withWindowIndex param is setted to True, each window is yielded with its position index.

	Args:
		signal (numpy.array): The 1d signal to loop on
		sampleRate (int): The signal sample rate
		windowWidth (float): The duration in second for each yielded window
		step (float): The duration in second between 2 consecutive windows (overlaping is aloud)
		withWindowIndex (bool): The flagg for indicating if the window index should be yielded

	Yields:
		numpy.array, int: The signal windo for each loop, the windo index (yielded only if withWindowIndex == true )

	"""




	windowWidth = int( windowWidth * sampleRate)
	step = int( step * sampleRate)
	nbWindows = int((signal.size - windowWidth) // step ) +1
	if withWindowIndex:
		for i in range(nbWindows):
			startIndex = step * i
			endIndex = startIndex + windowWidth
			window = signal[startIndex:endIndex]
			yield window, i
	else:
		for i in range(nbWindows):
			startIndex = step * i
			endIndex = startIndex + windowWidth
			window = signal[startIndex:endIndex]
			yield window


def resample(signal, sampleRate, newSampleRate):
	"""Change the sample rate of a signal

	Args:
		Signal (np.array): The 1D signal
		sampleRate (int): The existing signal sample rate
		newSampleRate (int): The required signal sample rate

	Returns:
		numpy.array: The signal with the new sample rate

	"""

	nbSamples = int(newSampleRate*signal.size/sampleRate)
	signal = sp.signal.resample( signal, nbSamples)
	signal = np.int16(signal)
	return signal


def equalizeShapes( signal1, signal2):
	"""Append the shorter provided signal so that its size equals the second signal size

	Args:
		signal1 (numpy.array): The first 1d signal
		signal2 (numpy.array): The second 1d signal

	Returns:
		numpy.array, numpy.array: The 2 modified signals in de same order as entry

	"""

	if signal1.size < signal2.size: signal1 = np.append( signal1, [0] * (signal2.size-signal1.size))
	elif signal1.size > signal2.size: signal2 = np.append( signal2, [0] *(signal1.size - signal2.size))
	return signal1, signal2

