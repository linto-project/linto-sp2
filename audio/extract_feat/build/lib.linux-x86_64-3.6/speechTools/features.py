"""features module for processing speech signal descriptors.

This module contains functions for manipulating high level speech features like pitch speech rate... It contain as well functions for preprocessing features before the clustering step. Here, features are matrixes where each row representes a sample and each column a single metric.
This could be considered as a major module in the package.

"""


import librosa
import pickle
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import speechTools as st


def getFeaturesCorrelation ( features ):
	"""Return correlation matrix computed on given features

	This function compute the correlation matrix exposing perwise relationship between features. CorMatrix[I,J] is the correlation between the Ist and Jst features in the features matrix 

	Args:
		features (numpy.ndarray): The features matrix

	Returns:
		numpy.ndarray: The correlation matrix

	"""

	correlationMatrix = np.corrcoef(features, rowvar=False)
	return correlationMatrix


def scaleFeatures( features, scaler=None):
	"""Return scaled features using default or provided sklearn scaler

	This function scale features using the provided scaler. This is particularly relevant when the features have different definition ranges. if any scaler is specified, the sklearn robustScaler is used.

	Args:
		features (numpy.ndarray): The features matrix to scale
		scaler (ani sklearn scaler or none): The scaler to use

	Returns:
		numpy.ndarray: the scaled features matrix

	"""

	if len(features.shape) == 1: features = features.reshape( -1, 1)
	if not scaler:
		scaler = RobustScaler()
		scaler.fit(features)
	scaledFeatures = scaler.transform(features)
	return scaledFeatures, scaler


def reduceFeatures( features, pca=None, varianceThreshold=.90):
	"""Aply PCA reduction on given features

	This function aplies PCA reduction on the provided features matrix. This reduction uses a sklearn PCA operator that could be passed with the pca arg. The reduction needs to specify a remaining variance.

	Args:
		features (numpy.ndarray): The features matrix
		pca (sklearn.decomposition.PCA): The PCA operator(if None an new operator is created)
		varianceThreshold (float): The remaining variance to keep defined in [0...1] with 1 as the maximum existing variance

	Returns:
		numpy.ndarray, sklearn.decomposition.PCA: The column reduced features matrix, The used PCA operator

	"""

	if not pca:
		pca = PCA(varianceThreshold)
		pca.fit(features)
	reducedFeatures  = pca.transform(features)
	return reducedFeatures, pca


def computeLocalFeatures ( window, sampleRate=100, forbiddenValue=0):
	"""Return time aggregation features for an given features window/sequence 

	This function Takes as entry a features windo containing consecutive single features values.
	It computes a short representation of the entry using the 11 folowing statistic metrics: min(minimum), Q1(first quartile), med(median), Q3(third quartile), max(maximum), mean, std(standard deviation), mad(median absolute deviation), kurtosis, skewness, variability(time normalised absolute derivative)
	A forbidden value could be specified for skeeping null values or other not representative value. If all values in window are rejected, a 11 size zero array is returned  

	Args:
		window (numpy.array): The time consecutive single feature values
		sampleRate (int): the feature values sampling rate for variability time normalisation
		forbiddenValue (Float, None): The value to exclude from computation

	Returns:
		numpy.array: the 11 size vector containing the aggregation metrics

	"""

	validWindow = window[ window != forbiddenValue]
	if validWindow.size == 0: return np.zeros(11)
	min, q1, median, q3, max = st.computePercentiles( validWindow)
	mean, std, mad = st.computeDistributionParams ( window)
	kurtosis, skewness = st.computeDistributionShape( validWindow)
	variability = st.computeSignalVariability( window, sampleRate)
	return np.array([min, q1, median, q3, max, mean, std, mad, kurtosis, skewness, variability])


def mergeFeatures ( * featureMatrixes):
	"""Merge features matrixes in one single matrix

	This function merges all features matrixes in entry in one single matrix. This merging is applied to each matrix row. The final number of rows is equal to the minimal number of rows in the entry matrixes, so that merged rows are defined in all entry matrixes. Other rows are ignored.

	Args:
		featureMatrixes (tuple): The entry features matrixes with the usage >>>mergedFeatures = speechTools.mergeFeatures(matrix1, matrix2,... matrixN)

	Returns:
		numpy.ndarray: The merged single features matrix

	"""

	featureMatrixes = list(featureMatrixes)
	minLen = np.min( [len(matrix) for matrix in featureMatrixes])
	if minLen == 0: return []
	for m in range(len(featureMatrixes)):
		matrixLen =len(featureMatrixes[m])
		if matrixLen > minLen: featureMatrixes[m] = featureMatrixes[m][:minLen-matrixLen]
	mergedFeatures = np.concatenate( featureMatrixes, axis=1)
	return mergedFeatures

def loadFeatures( ftFile):
	"""Load features matrix from a pickle .ft file

	This function opens a .ft file, a simple pickle file containing a precomputed features matrix, and returns this matrix

	Args:
		ftFile (str): The path to the specified .ft file

	Returns:
		numpy.ndarray: The precomputed features matrix

	"""

	features = None
	with open( ftFile, "rb") as input: features = pickle.load( input,)
	return features


def saveFeatures( features, ftFile="features.ft"):
	"""save features matrix in a pickle .ft file

	This function opens a .ft file, a simple pickle file and uses it to save the given features matrix.

	Args:
		features (numpy.ndarray): The  features matrix
		ftFile (str): The path to the specified .ft file

	"""

	with open(ftFile, "wb") as output: pickle.dump( features, output)


def extractPitchFeatures( signal, sampleRate=16000, pitchesFile=None, windowSize=30, windowStep=10 ):
	"""Extract pitch features for each audio window 

	This function construct a pitch signal that is readed through consecutive windows. For each window, 11 representative metrics are computed. Those metrics are defined in the speechTools.features.computeLocalFeatures function. The final result is a features matrix where each time ordered row refers to a window, and each column represents a single metric. if The param pitchFile is specified, the pitch signal is not constructed  from the given audio signal but directly loaded from a .f0 file.

	Args:
		signal (numpy.array): a mono PCM16 audio signal
		sampleRate (int): The audio signal sample rate(values different then 16000 may cose troubles)
		pitchesFile (str): The path to the precomputed pitch .f0 file
		windowSize (float): The size of the aggregative windo in second
		windowStep (float): the duration between 2 consecutive windows(overlapping is aloud)

	Returns:
		numpy.ndarray: The pitch features matrix

	"""

	if pitchesFile:
		pitches = None
		with open( pitchesFile, "rb") as pitchData: pitches = pickle.load( pitchData)
	else: pitches = st.getPitches (signal, sampleRate )
	nonZeros = np.nonzero( pitches)[0]
	vadMask = st.getVadMask ( signal, sampleRate)
	pitches, vadMask = st.equalizeShapes( pitches, vadMask)
	pitches[nonZeros] *= vadMask[nonZeros]
	nonZeros = np.nonzero( pitches)[0]
	semitoneScale = st.getSemitoneScale()
	pitches [nonZeros] = st.quantifyValues( pitches[nonZeros], semitoneScale)
	semitones = np.zeros( pitches.size)
	semitones[nonZeros] = st.pitches2semitones( pitches[nonZeros], semitoneScale)
	pitchFeatures = []
	semitoneReader = st.getSignalReader( semitones, 100, windowSize, windowStep)
	for window in semitoneReader:
		localFeatures = computeLocalFeatures( window )
		pitchFeatures.append( localFeatures)
	pitcheFeatures = np.array(pitchFeatures)
	return pitchFeatures


def extractEnergyFeatures ( signal, sampleRate=16000, windowSize=30,windowStep=10):
	"""Extract energy features for each audio window 

	This function construct a energy signal that is readed through consecutive windows. For each window, 11 representative metrics are computed. Those metrics are defined in the speechTools.features.computeLocalFeatures function. The final result is a features matrix where each time ordered row refers to a window, and each column represents a single metric.

	Args:
		signal (numpy.array): a mono PCM16 audio signal
		sampleRate (int): The audio signal sample rate(values different then 16000 may cose troubles)
		windowSize (float): The size of the aggregative windo in second
		windowStep (float): the duration between 2 consecutive windows(overlapping is aloud)

	Returns:
		numpy.ndarray: The energy features matrix

	"""


	energies = st.collectEnergies ( signal )
	vadMask = st.getVadMask(signal, sampleRate)
	energies *= vadMask
	energyFeatures = []
	energyReader = st.getSignalReader( energies, 100, windowSize, windowStep)
	for window in energyReader:
		localFeatures = computeLocalFeatures( window )
		energyFeatures.append( localFeatures)
	energyFeatures = np.array(energyFeatures)
	return energyFeatures


def extractSnrFeatures ( signal, sampleRate=16000, windowSize=30,windowStep=10 ):
	"""Extract SNR features for each audio window

	This function construct a SNR signal that is readed through consecutive windows. For each window, 11 representative metrics are computed. Those metrics are defined in the speechTools.features.computeLocalFeatures function. The final result is a features matrix where each time ordered row refers to a window, and each column represents a single metric. In this case the SNR refers to a speech signal to noise ratio. It distinguish the main voice signal from other background audio signal with the WADA algorithm definition.

	Args:
		signal (numpy.array): a mono PCM16 audio signal
		sampleRate (int): The audio signal sample rate(values different then 16000 may cose troubles)
		windowSize (float): The size of the aggregative windo in second
		windowStep (float): the duration between 2 consecutive windows(overlapping is aloud)

	Returns:
		numpy.ndarray: The SNR features matrix

	"""


	snrs = st.getSnrs ( signal, sampleRate)
	snrIndex = 0
	minSnr = np.min(snrs)
	vadMask = st.getVadMask(signal, sampleRate)
	vadMaskReader = st.getSignalReader( vadMask, 100, 1, 0.5)
	for window in vadMaskReader:
		if np.mean(window) <=  0.5: snrs[snrIndex] = minSnr
		snrIndex +=1
	snrFeatures = []
	snrReader = st.getSignalReader( snrs, 2, windowSize, windowStep)
	for window in snrReader:
		localFeatures = computeLocalFeatures( window , 2, minSnr)
		snrFeatures.append( localFeatures)
	snrFeatures = np.array(snrFeatures)
	return snrFeatures


def extractRhythmFeatures( signal=None, sampleRate=None, sylFile=None, windowSize=30,windowStep=10):
	"""Extract rythm features for each audio window 

	This function constructs 3 rhythm signals that are readed through consecutive windows. For each window and each signal, 11 representative metrics are computed. Those metrics are defined in the speechTools.features.computeLocalFeatures function. The final result is 3 features matrixes where each time ordered row refers to a window, and each column represents a single metric. The 3 signals represent respectively: syllabic rate(number of syllables per second), syllabic durations (mean syllable duration for each second), vowel durations(mean vowel durations for each second). If The param sylFile is specified, the syllabic detection is not performed but those are loaded from a precompputed .syl file. In this case the params signal and sampleRate are ignored.

	Args:
		signal (numpy.array): a mono PCM16 audio signal
		sampleRate (int): The audio signal sample rate(values different then 16000 may cose troubles)
		sylFile (str): The path to the precomputed syllables .syl file
		windowSize (float): The size of the aggregative windo in second
		windowStep (float): the duration between 2 consecutive windows(overlapping is aloud)

	Returns:
		numpy.ndarray, numpy.ndarray, numpy.ndarray: The syllabic rate features matrix, the syllabic duration features matrix, the vowel duration features matrix

	"""


	if sylFile:
		syllables = None
		with open( sylFile, "rb") as sylData: syllables = pickle.load( sylData)
	else: syllables = st.getSyllables2( signal, sampleRate)
	syllabicRates = st.getSyllabicRates( syllables  )
	syllabicRateFeatures = []
	syllabicRateReader = st.getSignalReader( syllabicRates, 1, windowSize, windowStep)
	for window in syllabicRateReader:
		localFeatures = computeLocalFeatures( window, 1 )
		syllabicRateFeatures.append( localFeatures)
	syllabicRateFeatures = np.array(syllabicRateFeatures)
	syllabicDurations = st.getSyllabicDurations( syllables )
	syllabicDurationFeatures = []
	syllabicDurationReader = st.getSignalReader( syllabicDurations, 1, windowSize, windowStep)
	for window in syllabicDurationReader:
		localFeatures = computeLocalFeatures( window, 1 )
		syllabicDurationFeatures.append( localFeatures)
	syllabicDurationFeatures = np.array(syllabicDurationFeatures)
	vowelDurations = st.getVowelDurations( syllables )
	vowelDurationFeatures = []
	vowelDurationReader = st.getSignalReader( vowelDurations, 1, windowSize, windowStep)
	for window in vowelDurationReader:
		localFeatures = computeLocalFeatures( window, 1 )
		vowelDurationFeatures.append(localFeatures)
	vowelDurationFeatures = np.array(vowelDurationFeatures)
	return syllabicRateFeatures, syllabicDurationFeatures, vowelDurationFeatures


def extractSpectralFeatures ( signal, sampleRate, windowSize=30,windowStep=10):
	"""Extract spectral features for each audio window

	This function constructs 2 spectral signals that are readed through consecutive windows. For each window and each signal, 11 representative metrics are computed. Those metrics are defined in the speechTools.features.computeLocalFeatures function. The final result is 2 features matrixes where each time ordered row refers to a window, and each column represents a single metric. The 2 signals represent respectively: spectral centroid, spectral flatness.

	Args:
		signal (numpy.array): a mono PCM16 audio signal
		sampleRate (int): The audio signal sample rate(values different then 16000 may cose troubles)
		windowSize (float): The size of the aggregative windo in second
		windowStep (float): the duration between 2 consecutive windows(overlapping is aloud)

	Returns:
		numpy.ndarray, numpy.ndarray: The spectral centroid features matrix, the spectral flatness features matrix

	"""


	floatSignal = np.float32(signal)
	spectralCentroids = librosa.feature.spectral_centroid( floatSignal, sampleRate, n_fft=512, hop_length=160, center=False)
	spectralCentroids = spectralCentroids.flatten()
	vadMask = st.getVadMask( signal, sampleRate)
	spectralCentroids, vadMask = st.equalizeShapes( spectralCentroids, vadMask)
	spectralCentroids *= vadMask
	nonZeros = np.nonzero(spectralCentroids)[0]
	semitoneScale = st.getSemitoneScale()
	spectralCentroids[nonZeros] = st.quantifyValues( spectralCentroids[nonZeros], semitoneScale)
	semitones = np.zeros( spectralCentroids.size)
	semitones[nonZeros]= st.pitches2semitones( spectralCentroids[nonZeros], semitoneScale)
	semitoneFeatures = []
	semitoneReader = st.getSignalReader( semitones, 100, windowSize, windowStep)
	for window in semitoneReader:
		localFeatures = computeLocalFeatures( window )
		semitoneFeatures.append( localFeatures)
	semitoneFeatures = np.array(semitoneFeatures)
	spectralCentroidFeatures = semitoneFeatures
	spectralFlatnesses = librosa.feature.spectral_flatness ( floatSignal, n_fft=512, hop_length=160, center=False)
	spectralFlatnesses = spectralFlatnesses.flatten()
	spectralFlatnesses, vadMask = st.equalizeShapes( spectralFlatnesses, vadMask)
	spectralFlatnesses *= vadMask
	spectralFlatnessFeatures = []
	spectralFlatnessReader = st.getSignalReader( spectralFlatnesses, 100, windowSize, windowStep)
	for window in spectralFlatnessReader:
		localFeatures = computeLocalFeatures( window )
		spectralFlatnessFeatures.append( localFeatures)
	spectralFlatnessFeatures = np.array(spectralFlatnessFeatures)
	return spectralCentroidFeatures, spectralFlatnessFeatures

