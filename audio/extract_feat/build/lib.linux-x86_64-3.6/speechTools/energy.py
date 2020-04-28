"""energy module for  measuring time and spectral energy metrics

This module contains functions for computing energy related metrics. Those metrics describe low level signal and speech sub features.
"""


import numpy as np
import librosa
from math import log10
import speechTools as st


def collectEnergies ( signal, sampleRate=16000, windowWidth=0.03, step=0.01):
	"""Return the short terme energy values of each window of an audio signal

	This function reads an audio signal through consecutive windows. For each window the short terme energy is computed as the mean of the squared window amplitudes.

	Args:
		signal (numpy.array): The mono audio signal
		sampleRate (int): the signal sample rate
		windowWidth (float): The duration of the window in second
		step (float): the duration between 2 consecutive windows

	Returns:
		numpy.array: The resulting energy values vector

	"""

	energies = []
	signalReader = st.getSignalReader( signal, sampleRate, windowWidth, step)
	for window in signalReader:
		energies.append(np.mean(np.float64(window)**2))
	energyVector = np.array( energies)
	return energyVector


def getPowerSpectralDensity( signal, sampleRate):
	"""Compute the power spectral density of an audio signal

	This function computes the power spectral density of an audio signal. The entry signal is windowed through a Hann window, and its spectrum is extracted using a real FFT. The power density is the time normalised power2 absolute spectrum.

	Args:
		signal (numpy.array): The mono audio signal(usualy a short terme window E.G 30ms)
		sampleRate (int): the signal sample rate

	Returns:
		numpy.array: The power spectral density(half size of the entry signal)

	"""

	windowedSignal = signal *np.hanning( signal.size )
	spectrum =  np.fft.rfft( windowedSignal )
	absoluteSpectrum = abs(spectrum)
	energySpectralDensity =  absoluteSpectrum **2
	powerSpectralDensity = energySpectralDensity/ signal.size
	return powerSpectralDensity


def getSpectralBase ( windowSize, sampleRate ):
	"""Return the centers of frequency beans for a FFT setting

	This function returns a specteral base: a vector containing the central frequencies (in hz) for each spectral bean corresponding to the provided FFT setting.
	The FFT setting are the size of the signal window that should be passed through the FFT, and the sample rate of this Signal. The first center frequency is 0 and the last one is equal to the the half of the sample rate.

	Args:
		windowSize (int): The number of samples in the signal window
		sampleRate (int): the signal samplerate

	Returns:
		numpy.array: The spectral base (half size of the windo )

	"""

	spectralBase = np.fft.rfftfreq( windowSize, 1/sampleRate)
	return spectralBase


def getFrequencyBean ( frequency, spectralBase ):
	"""Find the nearest frequency bean in the spectral base for the given frequency

	This function returns the index of the spectral base where the frequency bean center is just higher then the given frequency. If all the bean centers are lower then the given frequency, the spectral base size is returned.

	Args:
		frequency (int): The frequency in hz
		spectralBase (numpy.array): The vector containing the ordered frequency bean centers in hz (as defined in the speechTools.energy.getSpectralBase function)

	Returns:
		int: The nearest bean center index

	"""

	allBeansKept = True
	for bean in range( spectralBase.size):
		if spectralBase[bean] >= frequency:
			allBeansKept = False
			break
	if allBeansKept: bean = spectralBase.size
	return bean


def computeSpectrumPower( spectrum, firstBean=None, excludedBean=None ):
	"""Compute the power sum of the specified beans in a spectrum

	This function sums up the power contained in the spectrum between the first specified bean (included index) and the specified excluded bean (excluded index). If one of those beans is not specified, any boundary is used at the corresponding side of the spectrum.

	Args:
		spectrum (numpy.array): The power spectrum
		firstBean (int): The index of the first bean to include in the sum
		excludedBean (int): the index of the first bean to exclude

	Returns:
		float: The sum of the selected spectrum beans

	"""

	power = np.sum( spectrum[firstBean:excludedBean] )
	return power


def computeRec(window, sampleRate, firstLFBean = 9, firstHFBean=32, lastBean=179):
	"""Compute REC the reduced energy cumulating value  for an audio signal window

	This function compute the reduced energy cumulating value. This value is a ratio evaluating the the proportion of low and high frequencies in the signal. In the speech context a high REC value indicate that the window represents a vowel sound. The vowel sounds are indeed concentrated in the low spectrum. At the other side consonant sounds contain more high frequencies due to the front oral activity.
	For more details check up francois pellegrino and  jerome farinas works on vowel and pseudo syllabes detection.

	Args:
		window (numpy.array): The mono audio signal short terme windo
		sampleRate (int) the signal sample rate
		firstLFBean (int): The window spectrum index of the first low frequency bean
		firstHFBean (int): The window spectrum index of the first high frequency bean
		lastBean (int): The window spectrum index of the last frequency bean

	Returns:
		float: The REC value

	"""

	spectrum= getPowerSpectralDensity ( window, sampleRate)
	lfPower = st.computeSpectrumPower( spectrum, firstBean=firstLFBean, excludedBean=firstHFBean)
	power = st.computeSpectrumPower( spectrum, firstBean=firstLFBean, excludedBean=lastBean +1)
	rec = lfPower / power * sum(abs(spectrum - np.mean(spectrum)))
	return rec


def getRecCurve ( signal, sampleRate=16000, firstLFFrequency=300, firstHFFrequency=1000, lastFrequency=5600, windowDuration= 0.032, stepDuration = 0.01):
	"""Return the REC reduced energy cumulating curve of an audio signal

	This function reads an audio signal through consecutive windows. for each window a REC value is computed using the speechTools.energy.computeRec function that needs to pass a frequency setting for distinguishing low and high frequencies.
	The REC curve is a good tool for detecting vowel positions by finding large positive peaks in the curve. To select those peaks the function computes as well a REC threshold sett to the curve  median value.

	Args:
		signal (numpy.array): The mono audio signal
		sampleRate (int): The signal sample rate
		firstLFFrequency (int): The first low frequency in hz
		firstHFFrequency (int): The first high frequency in hz
		lastFrequency (int): The last frequency in hz
		windowDuration (float): The window duration in second
		stepDuration (float): The duration between 2 consecutive windows

	Returns:
		numpy.array: The REC curve time ordered values

	"""

	windowSize = int(windowDuration * sampleRate)
	windowStep = stepDuration *  sampleRate
	spectralBase = st.getSpectralBase ( windowSize, sampleRate)
	firstLFBean = getFrequencyBean ( firstLFFrequency, spectralBase)
	firstHFBean = getFrequencyBean ( firstHFFrequency, spectralBase)
	lastBean = getFrequencyBean( lastFrequency, spectralBase)
	nbRecs = int((signal.size - windowSize) // windowStep) +1
	recCurve = np.zeros(nbRecs)
	signalReader = st.getSignalReader( signal, sampleRate, windowDuration, stepDuration, withWindowIndex=True)
	for window, windowIndex in signalReader:
		if np.any(window): recCurve[windowIndex] = st.computeRec ( window, sampleRate, firstLFBean = firstLFBean, firstHFBean=firstHFBean, lastBean=lastBean)
	#for i in range(1, nbRecs - 1): recCurve[i] = np.mean(recCurve[i-1:i+2])
	recThreshold = np.median(recCurve) 
	return recCurve, recThreshold
