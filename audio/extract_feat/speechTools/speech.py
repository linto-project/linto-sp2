"""speech module for high level speech signal processing.

This module contains functions for extracting and modeling speech signal. It includes speech focused tools for describing speech stiles.
This could be considered as a major module in the package.

"""


import pickle
import multiprocessing as mp
import numpy as np
import webrtcvad
import speechTools as st
from speechTools.diverg import segment
from speechTools.WADA import compute_wada_snr_seg


def getSyllabicDurations( syllables):
	"""Compute the syllabic consecutive durations using pseudo-syllables units

	This function takes as entry a pseudo-syllables sequence and compute the mean syllabic durations for each second.

	Args:
		syllables (list): The pseudo-syllable sequence

	Returns:
		numpy.array: The syllabic durations as a signal sampled to one value per second

	"""

	#the time centered syllable positions

	times = []
	#the syllable durations
	durations = []
	for syllable in syllables:
		syllableStart = syllable[1]
		syllableEnd = syllable[2]
		syllabicDuration = syllableEnd - syllableStart
		times.append( (syllableEnd + syllableStart) /2)
		durations.append (   syllabicDuration)
	#the rounded speech start and end times in second
	startTime = int(syllables[0][1])
	endTime = int(syllables[-1][2]) + 1
	#the syllable durations grouped together for each second during speech
	cumulativeDurations = [ [] for i in range (startTime, endTime)]
	for i in range(len(times)):
		#the correct position to insert the current syllable duration in the cumulativeDurations list regarding speech start time ofset
		cumulativePosition = int(times[i]) - startTime
		cumulativeDurations[cumulativePosition].append( durations[i])
	#the averaged syllable durations for each second
	#start by adding the potential null  values before speech start time
	alignedDurations = [0] * startTime
	for i in range (len(cumulativeDurations)):
		if cumulativeDurations[i] == []: alignedDuration = 0
		else: alignedDuration = np.mean(cumulativeDurations[i])
		alignedDurations.append(alignedDuration)
	alignedDurations = np.asarray(alignedDurations)
	return alignedDurations


def getVowelDurations( syllables):
	"""Compute the vowel consecutive durations using pseudo-syllables units

	This function takes as entry a pseudo-syllables sequence and compute the mean vowel durations for each second.

	Args:
		syllables (list): The pseudo-syllable sequence

	Returns:
		numpy.array: The vowel durations as a signal sampled to one value per second

	"""

	#the time centered vowel positions
	times = []
	durations = []
	for syllable in syllables:
		vowelEnd = syllable[2]
		vowelDuration = syllable[4]
		vowelStart = vowelEnd - vowelDuration
		times .append(  (vowelEnd + vowelStart) /2)
		durations.append (  vowelDuration)
	#the rounded speech start and end times in second
	startTime = int(syllables[0][1])
	endTime = int(syllables[-1][2]) + 1
	#the vowel durations grouped together for each second during speech
	cumulativeDurations = [ [] for i in range (startTime, endTime)]
	for i in range(len(times)):
		#the correct position to insert the current vowel duration in the cumulativeDurations list regarding speech start time ofset
		cumulativePosition = int(times[i]) - startTime
		cumulativeDurations[cumulativePosition].append( durations[i])
	#the averaged syllable durations for each second
	#start by adding the potential null  values before speech start time
	alignedDurations = [0] * startTime
	for i in range (len(cumulativeDurations)):
		if cumulativeDurations[i] == []: alignedDuration = 0
		else: alignedDuration = np.mean(cumulativeDurations[i])
		alignedDurations.append(alignedDuration)
	alignedDurations = np.asarray(alignedDurations)
	return alignedDurations


def getSyllabicRates ( syllables):
	"""Compute the the number of pseudo-syllable units per second

	This function takes as entry a pseudo-syllables sequence and compute the number  of pseudo-syllables per second.

	Args:
		syllables (list): The pseudo-syllable sequence

	Returns:
		numpy.array: The syllabic rate as a signal sampled to one value per second

	"""

	#rounded speech start end end times in second
	startTime = int(syllables[0][1])
	endTime = int(syllables[-1][2]) + 1
	#syllabic rate sampled  to each second (sample rate = 1) including the first null values before starting speech
	rates = [0] * startTime
	#the times of each second during speech e.g. [5,6,7,8] if speech starts just after 5s and ends just before 9s
	#each second e.g. {5-6} is a time zone where the syllabic rate is computed counting the syllables in this zone
	#if a long syllable is  starting in a zone and ending in another (overlaping), only the proportion effectively in the zone is counted
	#this proportion is computed by dividing the effective sub duration in the zone by the overal syllabic duration
	times = np.linspace( startTime, endTime, endTime - startTime, False)
	#index of the next syllable to process
	nextSyllableIndex = 0
	#remaining duration of a syllable beyond its starting zone
	overload = 0
	#true if overload > 0
	overlaping = False
	syllabicDuration = 0
	# for each zone during speech
	for t in range (times.size):
		rate = 0
		#if the current syllable has already started during the preceding loop
		if overlaping:
			# if multiple overlaping
			if overload >= 1: 
				#rate is equal to 1s / the overal syllable duration for using relative  syllable proportion only
				rate = 1 / syllabicDuration
				overload -= 1
				rates.append(rate)
				continue
			else:
				#the relative proportion is used : a half syllable means a rate of 0.5
				rate = overload / syllabicDuration
				overload = 0
				overlaping = False
			nextSyllableIndex +=1
		nextTime = times[t] + 1
		for syllable in syllables[nextSyllableIndex:]:
			syllableStart = syllable[1]
			syllableEnd = syllable[2]
			#if the syllable is all included in the current time zone 
			if syllableEnd <= nextTime:
				rate += 1
				nextSyllableIndex +=1
			#if the syllable is starting in the current time zone and ending  beyond we have an overlaping
			elif syllableStart < nextTime:
				overlaping = True
				syllabicDuration = syllableEnd-syllableStart
				subDuration = nextTime - syllableStart
				rate += subDuration/ syllabicDuration
				overload = syllabicDuration - subDuration
				break
			#if no syllable  in the current time zone go to next one
			else : break
		rates.append(rate)
	rates = np.asarray(rates)
	return rates


def getFBSegments(signal, sampleRate, order=2):
	"""Returns the audio stable segments ofthe given signal

	This function uses the forward-backward algorithm for segmenting an audio signal on stable segments. 
	The stable segments are phonologicaly supra segmantal units, smaller than phonems and not  differentiated as those. Each segment is considered  as stable in refference to the low variability of its variance. This characteristic indicates that the segment is the representation of a single sound.
	Here, the resulting sound could be very short. In instance, multiple segments could represente a writen letter.
	For more detail on the algorithm, the implementation is hosted in the speechTools/diverg/diverg.c module.

	Args:
		signal (numpy.array): mono int16 audio signal
		sampleRate (int): audio signal sample rate(should be equal to 16000 for avoiding troubles)
		order (int): The order of the forward-backward algorithm

	Returns:
		np.ndarray: The list of stable segments with the forme [...[start time in second, end time in second]...]

	"""

	signal = list( signal )
	boundaries = [ b for b in segment(signal, sampleRate, order, 0.01)]
	fBSegments = np.array(list(zip(boundaries[:-1],boundaries[1:])))
	return fBSegments


def getVadMask ( signal, sampleRate=16000, vad=None, aggressiveness=3, windowWidth=0.03, windowStep=0.01):
	"""Construct a boolean mask for identifying speech zones in a audio signal

	This function returns a boolean mask for detecting speech zones in an audio signal. The audio signal is readed through short terme windows, and for each windo, a boolean value is raised(1 if the window contains speech 0 otherwise ). The result is a mask or a vector containing a decision value for each window. This way, the mask could bi multiplied by a higher level signal (with the same step spacing) , for speech filtering. An other usage is to compute a speech quantity ratio of a audio segment through the mean or median value of the boolean mask, for detection thresholding tasks.

	Args:
		signal (numpy.array): mono int16 audio signal
		sampleRate (int): audio signal sample rate(should be equal to 16000 for avoiding troubles)
		vad (webrtcvad.Vad): if specified an existing configured web RTC VAD, otherwise an new instance is created using the aggressiveness param
		aggressiveness (int): The aggressiveness of the Vad instance, could only be [1,2,3] with 3 the most aggressive value
		windowWidth (float): The size of the audio window in second, could only be [0.01,0.02,0.03]
		windowStep (float): Duration between 2 consecutive windows (overlapping is aloud)

	Returns:
		numpy.array: the mask or boolean vector using the specified window step as value spacing

	"""

	vadMask = []
	if not vad: vad = webrtcvad.Vad( aggressiveness)
	signalReader = st.getSignalReader( signal, sampleRate, windowWidth=windowWidth, step = windowStep)
	for window in signalReader:
		bytesFrame = window.tobytes()
		vadMask.append ( vad.is_speech(bytesFrame, sampleRate))
	vadMask = np.array ( vadMask)
	return vadMask


def detectVocalActivity (  signal, sampleRate, segments, aggressiveness=3, aloudError=0.25, reductionFactor = 0.8):
	"""Find respectively speech, plosives and silence segments in a sequence of audio stable segments

	This function reads a sequence of stable audio segments and returns separately  and respectively the segments corresponding to speech, plosives and silence. Each segment is a list with the forme : [start time in second, end time in second].
	The affectations use an web RTC VAD object with an given aggressiveness.
	The stable segments are phonologicaly supra segmantal units, smaller than phonems and not  differentiated as those. The stable segments are provided by the speechTools.speech.getFBSegments function, using the forward-backward algorithm.
	The result is 3 lists representing  the segments containing speech, those containing plosives ( less than 150ms silences like in the sounds "P, T, C"), and long silence segments containing any speech.

	Args:
		signal (numpy.array): mono int16 audio signal
		sampleRate (int): audio signal sample rate(should be equal to 16000 for avoiding troubles)
		segments (numpy.ndarray): the time ordered stable audio segments
		aggressiveness (int): The aggressiveness of the Vad instance, could only be [1,2,3] with 3 the most aggressive value
		aloudError (float): The aloud error  between 0 and 1 for quantifying speech quantity in each segment
		reductionFactor (float): The symmetrical length reduction applied to each segment for avoiding segment transition perturbations

	Returns:
		numpy.array, numpy.array, numpy.array: List of speech segments, list of plosive segments, list of silence segments

	"""

	speechSegments = []
	plosiveSegments = []
	silenceSegments = []
	vad = webrtcvad.Vad( aggressiveness)
	for segmentStart, segmentEnd in segments:
		#find the segment corresponding signal window
		windowStart = int(segmentStart * sampleRate)
		windowEnd = int ( segmentEnd * sampleRate)
		windowSize = windowEnd - windowStart
		#find the reduced window which is a troncated window for excluding   the  parts of the window that could be impacted by the neighbour windows 
		reducedWindowSize = int( windowSize * reductionFactor)
		reducedWindowStart = int( windowStart + (windowSize * ( 1 - reductionFactor) * 0.5))
		reducedWindowEnd = reducedWindowStart + reducedWindowSize
		reducedWindow = signal[reducedWindowStart:reducedWindowEnd]
		#get the VAD mask for the reduced window
		vadMask = st.getVadMask( reducedWindow, sampleRate, vad=vad, windowWidth=0.01, windowStep=0.005)
		#compute the speech quantity i.e. the proportion of [1] in the mask 
		speechQuantity = 1
		if vadMask.size > 0: speechQuantity = vadMask[vadMask==1].size / vadMask.size
		# if there is enough speech: add the segment to the speech segments
		#else if it is a very short silence  add it to plosives 
		#else add it to silences
		if speechQuantity > (1- aloudError) : speechSegments.append([segmentStart, segmentEnd])
		elif (segmentEnd - segmentStart) < 0.15: plosiveSegments.append([segmentStart, segmentEnd])
		else: silenceSegments.append( [segmentStart, segmentEnd])
	speechSegments = np.array(speechSegments)
	plosiveSegments = np.array(plosiveSegments)
	silenceSegments = np.array(silenceSegments)
	return speechSegments, plosiveSegments, silenceSegments


def findVocalicSegments(speechSegments, pitches, recCurve, recSampleRate, recThreshold,recPeaks, voicingFactor = 0.85):
	"""Find respectively vocalic and non vocalic segments in a sequence of audio stable segments

	This function reads a sequence of stable speech segments and returns separately  and respectively the vocalic and non vocalic segments . Each segment is a list with the forme : [start time in second, end time in second].
	The stable speech segments are supra segmantal units, smaller than phonems and not  differentiated as those. The stable speech segments are provided by the speechTools.speech.detectVocalActivity function
	The vocalic segments correspond to vowel sounds. They should contain a voice fundamental frequency  or pitch, produced by the vocal cords), and do not have a strong high frequencies component . In the other side, non vocalic segments, by default, correspond to consonant sounds. they do not  contain voice fundamental frequency and have a strong high frequencies componentcosed by frontal oral articulations.

	Args:
		speechSegments (numpy.ndarray): The time ordered list of stable speech segments
		pitches (numpy.array): The pitch signal representing the sampled pitch measures  
		recCurve (np.array): The reduced energy cumulating curve representing the high/low frequencies ratio in the audio signal(provided by the speechTools.energy.getRecCurve function)
		recSampleRate (int): The sample rate of the REC curve (should be the same as the pitch signal sample rate)
		recThreshold (float): The threshold applied to the REC curve for validating a vocalic segment (higher values are more selective)
		recPeaks(np.array): The index positions of amplitude peaks in the REC curve
		voicingFactor (float): The minimal voicing ratio in a vocalic segment between 0 and 1 and representing the  required proportion  of fundamental frequency in the segment

	Returns:
		numpy.ndarray, numpy.ndarray: The time ordered vocalic segments, the time ordered non vocalic segments

	"""

	vocalicSegments = []
	nonVocalicSegments = []
	nbPeaks = len(recPeaks)
	#the first REC peak index of the remaining REC peaks
	peakIndex = 0
	for segmentStart, segmentEnd in speechSegments:
		segmentIsVocalic = False
		#jump to the segment corresponding position in the REC curve
		firstRecCurveIndex = int((segmentStart * recSampleRate) + 1)
		lastRecCurveIndex  = int( segmentEnd * recSampleRate)
		#for each remaining REC peak before segment end
		while peakIndex < nbPeaks and recPeaks[peakIndex] < lastRecCurveIndex:
			recCurveIndex = recPeaks[peakIndex]
			#the REC value on the peak
			recValue = recCurve[ recCurveIndex]
			peakIndex += 1
			#if the peak is not in the segment or its value lower than REC threshold continue
			#else the peak indicates a low frequency concentration in the segment: a requirement for vocalic detection
			if recCurveIndex < firstRecCurveIndex or recValue <= recThreshold: continue
			#a good REC value is not enough 
			#confirm voicing by verifying pitch using a reduced pitch mask
			#the pitch mask is a binary vector indicating for each pitch sample if there is voicing
			#the pitch signal and the REC curve chould be sampled the same way
			firstPitchIndex, lastPitchIndex = firstRecCurveIndex, lastRecCurveIndex
			reductionFactor = 0.8
			pitchSegmentSize = lastPitchIndex - firstPitchIndex
			#the pitch mask is shorter than the segment for avoiding border effect
			pitchMaskSize = int( pitchSegmentSize * reductionFactor)
			pitchMaskStart = int( firstPitchIndex + ((pitchSegmentSize - pitchMaskSize) / 2))
			pitchMaskEnd = pitchMaskStart + pitchMaskSize
			pitchMask = pitches[pitchMaskStart:pitchMaskEnd]
			pitchMask[pitchMask > 0] = 1
			#verify if there is enough voicing in the mask using the voicing factor
			#if verified add the segment to the vocalic segments
			if np.mean( pitchMask) > voicingFactor: segmentIsVocalic = True
			break
		if segmentIsVocalic: vocalicSegments.append([ segmentStart, segmentEnd ])
		else: nonVocalicSegments.append([ segmentStart, segmentEnd])
	vocalicSegments = np.array(vocalicSegments)
	nonVocalicSegments= np.array(nonVocalicSegments)
	return vocalicSegments, nonVocalicSegments


def getLetters( vocalicSegments, nonVocalicSegments, plosiveSegments, silenceSegments):
	"""Return a pseudo letters after aligning and merging characterised audio stable segments

	This function takes as entry vocalic, non vocalic, plosive and silence segments, and returns the aligned coresponding sub letters.
	The stable segments are phonologicaly supra segmantal units, smaller than phonems. The stable segments are provided by the speechTools.speech.getFBSegments function, using the forward-backward algorithm. Here, the segments are  characterised according to their sounding nature through the speechTools.speech.findVocalicSegments function.
	The letters are in reality pseudo letters that should not be considered as the traditional writen letters. Here the pseudo letters are simply labeled stable segments. the used labels are: 'V'  (vowel), 'C' (consonant), 'P' (plosive), 'S' (silence).

	Args:
		vocalicSegments (numpy.ndarray): The  time ordered vocalic segments
		nonVocalicSegments (numpy.ndarray): The time ordered non vocalic segments
		plosiveSegments (numpy.ndarray): The time ordered plosive segments
		silenceSegments (numpy.ndarray): The time ordered silence segments

	Returns:
		list: The liste of pseudo letters with the form [...( label, [startTime in second, end Time in second])...]

	"""

	#aligne the segments in a single time ordered list adding labels for keeping category information
	#we call those labeled segments subLetters
	subLetters = []
	#the subLetters labels refering respectively to Vowel, Consonant, Plosive, Silence
	labels = ["V", "C", "P", "S"]
	comparingTable = [ vocalicSegments, nonVocalicSegments, plosiveSegments, silenceSegments]
	indexes= [0,0,0, 0]
	nbCandidates = 4
	selector = 0
	while selector < nbCandidates:
		if len(comparingTable[selector]) > 0 : selector +=1
		else:
			nbCandidates -=1
			del labels[selector]
			del comparingTable[selector]
			del indexes[selector]
	while nbCandidates > 0:
		nextSegmentStarts = []
		for i in range( len( comparingTable)): nextSegmentStarts.append( comparingTable[i][indexes[i]][0])
		selector = np.argmin( nextSegmentStarts)
		label = labels[selector]
		segment = comparingTable[selector][indexes[selector]]
		subLetters.append( [ label, segment])
		if indexes[selector] + 1  == len(comparingTable[selector]):
			nbCandidates -=1
			del labels[selector]
			del comparingTable[selector]
			del indexes[selector]
			continue
		indexes[selector] +=1
	#rectify subLetters sequence by merging segments if particular label paterns are found for correcting vocalic detection missings
	for i in range (len(subLetters) -2):
		firstLabel, secondLabel, thirdLabel = subLetters[i][0], subLetters[i+1][0], subLetters[i+2][0]
		if (firstLabel, secondLabel, thirdLabel) == ("V", "C", "V"):
			secondSegmentDuration = subLetters[i+1][1][1] - subLetters[i+1][1][0]
			#if a consonant is to short and between 2 vowels: it's probably a missing vowel
			if secondSegmentDuration < 0.25: subLetters[i+1][0] = "V"
	#convert subLetters on letters by merging splited vowels
	letters = [subLetters[0]]
	for label, segment in subLetters[1 :] :
		lastLabel, lastsegment = letters[-1]
		labelCouple = ( lastLabel, label)
		#if 2 vowels ar consecutive: it's probably a single vowel
		if labelCouple == ("V", "V"):
			segment = (lastsegment[0], segment[1])
			letters.pop()
		letters.append((label, segment))
	#pop all the last letters that are not vowels
	nbLetters = len(letters)
	for i in range( 1, nbLetters +1) :
		if letters[- i][0] == "V" :
			letters = letters[ : nbLetters -i +1]
			break
	return letters


def letters2Syllables( letters ):
	"""Concatenate a sequence of pseudo letters on a new sequence of pseudo syllables

	This function converts a time ordered sequence of pseudo letters on a sequence of pseudo syllables.
	The pseudo letters should not be considered as the traditional writen letters. Here the pseudo letters are simply labeled stable segments. the used labels are: 'V'  (vowel), 'C' (consonant), 'P' (plosive), 'S' (silence).
	The pseudo syllables should not be considered as language based writen syllable. Here the pseudo syllable is a universal concatenation of letters (as described in the function speechTools.speech.getLetters),with the patern 'CC...CCV', : a undefined number of consonants folowed by a vowel. The pseudo syllable could be considered as a elementary unit of the speech signal, being a generic alternative to phonems or words.
	Each pseudo syllable is a vector with the form [number of consonants, start time in second, end time in second, vowel start time in second, consonant duration, vowel duration]

	Args:
		letters (list): The time ordered sequence of letters ( labeled audio stable segments )

	Returns:
		list: The time ordered sequence of syllables 

	"""

	syllables = []
	syllableSegments = []
	nbConsonants = 0
	consonantDuration = 0
	vowelDuration = 0
	for label, segment in letters :
		#skip all plosives
		if label == "P": continue
		syllableSegments.append(segment)
		#if consonant
		if label == "C" :
			nbConsonants += 1
			consonantDuration += segment[1] - segment[0]
			continue
		#if vowel
		if label == "V":
			syllableStart = syllableSegments[0][0]
			syllableEnd = syllableSegments[-1][1]
			vowelDuration = segment[1] - segment[0]
			vowelStart = segment[0]
			syllables.append([nbConsonants, syllableStart, syllableEnd, vowelStart, consonantDuration, vowelDuration])
		syllableSegments = []
		nbConsonants = 0
		consonantDuration = 0
		vowelDuration = 0
	return syllables


def getSyllables ( signal, sampleRate, pitchesFile=None):
	"""Extract pseudo syllables from an audio signal containing speech

	This function takes as entry an audio signal and finds pseudo syllabe units.
	The pseudo syllables should not be considered as language based writen syllable. Here the pseudo syllable is a universal concatenation of letters (as described in the function speechTools.speech.getLetters),with the patern 'CC...CCV', : a undefined number of consonants folowed by a vowel. The pseudo syllable could be considered as a elementary unit of the speech signal, being a generic alternative to phonems or words.
	Each pseudo syllable is a vector with the form [number of consonants, start time in second, end time in second, vowel start time in second, consonant duration, vowel duration]
	The used algorithm is an upgraded verssion of the  pseudo syllabe extractor presented in Francois pellegrino's PHD works. It folowes the processing steps below:
	1. Pitch extraction for voicing (presence of fundamental frequency) detection, using The Reaper external tool;
	2. Forward-backward segmentation for speech signal discretization using the diverg external module;
	3. Speech / non spitch  segments partitioning, using the Web RTC VAD external tool;
	4. Reduced energy cumulating curve computation for detecting low frequency regions ;
	5. Vocalic segments detection by detecting low frequency voiced segments;
	6. Building pseudo letters through vowel'V'  (a vocalic segment), or  consonant'C' (a non vocalic segment) labelisation   ;
	7. Building pseudo syllables by concatenating pseudo letters in longer units with the patern  'CC...CV'.

	Args:
		signal (numpy.array): mono int16 audio signal
		sampleRate (int): audio signal sample rate(should be equal to 16000 for avoiding troubles)
		pitches (str): If defined the path of the .f0 file containing precomputed pitch values for speeding up the extraction

	Returns:
		list: The time ordered pseudo syllables sequence

	"""

	#if precomputed pitches are available: use a pitches file
	#else: compute pitches in a paralel process
	if pitchesFile:
		pitches = None
		with open( pitchesFile, "rb") as pitchData: pitches = pickle.load( pitchData)
	else:
		processManager = mp.Manager()
		pitchExtractionMethod = "reaper"
		blackBoard = processManager.list()
		pitchExtractionProcess = mp.Process ( target = st.getPitches, args = ( signal, sampleRate, pitchExtractionMethod, blackBoard))
		pitchExtractionProcess.start()
	segments = st.getFBSegments( signal, sampleRate)
	speechSegments, plosiveSegments, silenceSegments = st.detectVocalActivity(signal, sampleRate, segments, aggressiveness=2)
	recCurve, recThreshold = st.getRecCurve( signal, sampleRate)
	recPeaks = st.detect_peaks(recCurve)
	if not pitchesFile:
		pitchExtractionProcess.join()
		pitches = blackBoard.pop()
	vocalicSegments, nonVocalicSegments = st.findVocalicSegments( speechSegments, pitches, recCurve, 100, recThreshold , recPeaks, voicingFactor=0.7)
	letters = st.getLetters( vocalicSegments, nonVocalicSegments, plosiveSegments, silenceSegments)
	syllables = st.letters2Syllables( letters )
	return syllables


def getSnrs ( signal, sampleRate):
	"""Return the voice signal to noise ratio measures of an audio signal

	This function reads an audio signal and compute the voice signal to noise ratio using the wave forme based WADA algorithme, that is implemented in the external WADA module. This type of SNR measures the level of a single voice signal reported to the other background sounds considered as noise. Here, the used WADA algorithm is not short term designed. It canot be used on short signal windows (I.E 30ms) but on longer zones (I.E 1s) the measures  are 50% overlaped, so that we optain a measure each 500ms, and a sample rate of 2.

	Args:
		signal (numpy.array): mono int16 audio signal
		sampleRate (int): audio signal sample rate(should be equal to 16000 for avoiding troubles)

	Returns:
		numpy.array: the Time ordered SNR measures

	"""

	normalizedSignal = st.WADA.pcm2float( signal, "float32")
	snrs = []
	signalReader = st.getSignalReader ( normalizedSignal, sampleRate, 1, 0.5)
	for window in signalReader:
		snrs.append(st.WADA.compute_snr_on_signal( window ))
	snrs =np.array(snrs)
	return snrs



