"""signal modul for signal processing.

This module contains audio DSP functions for processing low level audio signals from wav files. It include as well general purpose functions for signal modeling or aggregation.

"""


import numpy as np
import scipy as sp
import speechTools as st


def quantify ( value, scale ):
	if value < scale[0] or value > scale[-1]: return value
	for i in range (1, len(scale)):
		if value <= scale[i]:
			if scale[i] - value > value - scale[i-1]: return scale[i-1]
			else: return scale[i]


def quantifyValues( values, scale):
	for v in range(len(values)):
		values[v] = st.quantify( values[v], scale)
	return values

def computeSignalVariability( signal, sampleRate, order=1):
	sampleSpacing = 1 / sampleRate
	absoluteDerivative = abs(np.diff( signal, n=order) / sampleSpacing)
	signalVariability = np.mean( absoluteDerivative )
	return signalVariability


def getSignalReader( signal, sampleRate=16000, windowWidth=0.032, step=0.01, withWindowIndex=False):
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
	nbSamples = int(newSampleRate*signal.size/sampleRate)
	signal = sp.signal.resample( signal, nbSamples)
	signal = np.int16(signal)
	return signal


def equalizeShapes( signal1, signal2):
	if signal1.size < signal2.size: signal1 = np.append( signal1, [0] * (signal2.size-signal1.size))
	elif signal1.size > signal2.size: signal2 = np.append( signal2, [0] *(signal1.size - signal2.size))
	return signal1, signal2

