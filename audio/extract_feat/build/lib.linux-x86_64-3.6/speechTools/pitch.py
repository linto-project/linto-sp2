"""pitch module for computing voice pitch signal.

This module contains functions for extracting and modeling pitch signal through various approaches.

"""


import os
import tempfile
import numpy as np
import speechTools as st


def getPitches ( signal, sampleRate=16000, method="reaper", blackBoard=None):
	"""Compute pitch (fundamental frequency or f0) measures of an audio signal containing speech

	This function extracts the voice pitch of an audio signal. It could use 2 methods: yin or Reaper.
	The yin algorithm is fast but quite precise. 
	At the other side, the Reaper algorithme is more reliable but slower and based on an external program that needs temporary IO files.
	The function is all so designed for multi processing strategy and could use a shared list (param blackBoard) for hosting results.

	Args:
		signal (numpy.array): mono int16 audio signal
		sampleRate (int): audio signal sample rate(should be equal to 16000 for avoiding troubles)
		method (str): the extraction method to use "yin" or "reaper"
		blackBoard (multiprocessing.managers.ListProxy): If defined a shared list for appending the resulting pitch signal

	Returns:
		np.array: The pitch measures with a sample rate of 100 (a value each 10ms)

	"""

	if method == "yin":
		times, pitches, harmonicity, argmin = st.getYinFeatures( signal, sampleRate=16000, harmonicThreshold=0.25)
	elif method == "reaper": pitches = st.getReaperPitches( signal, sampleRate)
	if blackBoard == None: return pitches
	blackBoard.append( pitches )


def getYinFeatures(  signal, sampleRate=16000, windowLen = 512, windowStep = 160, pitchMin = 70, pitchMax = 450, harmonicThreshold = 0.15):
	"""Extract the Yin algorithm features of an audio signal containing speech

	This function aplies a yin algorithm on a audio signal for extracting pitch related features, and uses a custom filtering on pitch for rectifying Yin errors.
	The features are respectively: The time stemps, the pitch values, the harmonicity(quantity of harmonics in the signal between 0 and 1 with 1 as a wite noise), the argMin value(un controled pitch values without using harmonicity thresholding).

	Args:
		signal (numpy.array): The mono audio int16 signal
		sampleRate (int): The audio signal sample rate (should be equal to 16000 for avoiding troubles)
		windowLen (int): The number of signal samples in a yin window (the maximum reachable time periode )
		windowStep (int): The number of samples between 2 yin windows (overlaping is aloud)
		pitchMin(int): The lowest reachable fundamental frequency in hz
		pitchMax (int): The highest reachable fundamental frequency
		harmonicThreshold(float): The maximum harmonicity for validating a pitch measure

	Returns:
		numpy.array, numpy.array, numpy.array, numpy.array: Time stemps, pitches, harmonicity, argMins

	"""

	times, pitches, harmonicRates, argMins =  st.compute_yin(signal, sampleRate, windowLen, windowStep, pitchMin, pitchMax, harmonicThreshold)
	pitches = filterPitches ( pitches, pitchMin, pitchMax, 0.15)
	times = np.array(times)
	harmonicRates = np.array(harmonicRates)
	argMins = np.array(argMins) 
	return times, pitches, harmonicRates, argMins


def getReaperPitches( signal=[], sampleRate=16000, wavFile=None):
	"""Call the external Reaper tool for extracting pitch from an audio signal or file

	This function uses the Reaper tool for extracting pitch. The entry could bi  an audio signal with its sample rate or an audio wav file. If the wavFile is provided the signal param should keep its default value.
	The Reaper tool is based on autocorrelation with several preprocessing and postprocessing tasks. It only takes as entry a mono int16 wav file that should be provided directly or writen from signal in a temporary location. Here, calling Reaper on large signals or files may has an important time cost but still more acurate compared to Yin based approach .
	The function output is a pitch signal containing pitch measures sampled to 100 (a measure each 10ms)

	Args:
		signal (numpy.array): If defined the mono audio int16 signal
		sampleRate (int): The audio signal sample rate (should be equal to 16000 for avoiding troubles)
		wavFile (str): The path to the entry wav file

	Returns:
	numpy.array: The extracted pitch values

	"""

	if signal != []:
		tempWavFile = tempfile.NamedTemporaryFile()
		st.writeWav( tempWavFile.name, signal, sampleRate)
		wavFile = tempWavFile.name
	tempPitchFile = tempfile.NamedTemporaryFile()
	os.system( "REAPER/build/reaper -i " + wavFile + " -f " + tempPitchFile.name + " -a")
	reaperResults = np.loadtxt( tempPitchFile, skiprows=7)
	pitches = reaperResults[:,2]
	trustFactors  = reaperResults[:,1]
	if pitches.size % 2 == 1:
		pitches = pitches[:-1]
		trustFactors = trustFactors [:-1]
	pitches = pitches.reshape((-1, 2))[:,0]
	trustFactors  = trustFactors .reshape((-1, 2))[:,0]
	pitches *= trustFactors
	return pitches


def filterPitches ( pitches, minBound=70, maxBound=450, delta=0.15 ):
	"""Rectify errors in the pitch signal

	This function Is specialy designed for rectifying yin pitches. It uses a local refference pitch value and a relative amplitude threshold for correcting outliers. The refference value on a point is computed using a short terme preceding window.

	Args:
		minBound (int): The minimal aloud frequency in hz
		maxBound (int): the maximal aloud frequency in hz
		delta (float): The normalised maximum amplitude difference between 2 consecutive points  

	Returns:
		numpy.array: The rectified pitch signal

	"""

	pitches = np.array (pitches)
	pitches[:4] = [0,0, 0,0]
	for f in range(3, len(pitches) -1):
		if pitches[f +1] == minBound or pitches[f +1] == maxBound: pitches[f +1] = pitches[f]
	for i in range(4, len(pitches) -3):
		if pitches[i+1] == 0: continue
		if pitches[i] == 0 and pitches[i+1] != 0 and pitches[i+2] == 0:
			pitches[i+1] = 0
			continue
		if pitches[i] == 0:
			localBase = pitches[i-4:i+1]
			nonZero = np.nonzero (localBase)[0]
			if len(nonZero) > 2:
				localBase = localBase[ nonZero]
				localPitch = np.median( localBase )
				localThreshold = localPitch + ( localPitch * delta* 3)
				if  pitches[i+1] > localThreshold:
					if pitches[i+1] /2.0 <= localPitch + delta :
						for backStep in [1,2]:
							if pitches [i-backStep] > 0:
								pitches[i+1] = pitches[i-backStep]
								break
					else: pitches[i+1] = 0
			else: pitches [i+1] = np.min( pitches[i+1:i+3])
			continue
		currentThreshold = pitches[i] + (pitches[i] * delta)
		if pitches[i+1] > currentThreshold:
			if pitches[i+1] / 2.0 <= currentThreshold: pitches[i+1] /= 2.0
			else: pitches[i+1] = pitches[i]
	return pitches


def getSemitoneScale ( key = 440, step = 1.05946, scaleWidth = 72):
	"""Build a tonal scale centered on a refference frequency

	This function Builts a tonal scale, by default the chromatic scale. It uses a refference frequency and constructs the  others scale components by a recurcive transposition: multiplying by a transposition factor (>1) for higher frequencies and dividing by this same factor for lower frequencies.

	Args:
		key (int): The scale center/refference frequency in hz
		step (float): The transposition  step (should be > 1) by default the chromatic one
		scaleWidth (int): The number of semitones in the scale

	Returns:
		numpy.array: The semitone scale in hz

	"""

	scale = np.zeros( scaleWidth )
	keyIndex = scaleWidth //2
	scale [keyIndex] = key
	for i in range (1, keyIndex):
		transpositionFactor = step ** i
		scale [keyIndex + i] = key * transpositionFactor
		scale [keyIndex - i] = key / transpositionFactor
	scale[0] = scale[1] / step
	scale[-1] = scale[-2] * step
	return scale


def pitches2semitones ( pitches, scale):
	"""Convert pitch signal to discrete semitones

	This function quantifies a pitch signal by converting it on a semitone sequence. Each pitch measure is rectified and replaced by the closest semitone. The closest semitone is chosen in the provided tonal scale. In the context of speech, this operation keeps only the macro melodic component in the pitch signal..

	Args:
		pitches (numpy.array): The pitch signal with values in hz
		scale (numpy.array): The frequency ordered semitone values in hz

	Returns:
		numpy.array: The semitone sequence

	"""

	semitones = np.zeros(pitches.size)
	for p in range(pitches.size):
		for s in range(scale.size):
			if pitches[p] == scale[s]:
				semitones[p] = s+1
				break
	return semitones

