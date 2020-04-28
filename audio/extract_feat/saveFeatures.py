"""Save speech features of multiple wav files

This script illustrates the approach for extracting and saving speech features of multiple wav files.
This script uses the Reaper pitch extractor that may have a significant time cost when extracting features of a large data set
During the extraction the pitch and syllable temporary files are saved as well for an eventual  futur use

"""


import pickle
import numpy as np
from os.path import exists
import speechTools as st

#The location of entry data files, should be modified according to  execution contextt
#wav files (.wav) location
wavDir = "data/wavs"
#pitch  files(.f0) location 
pitchesDir = "data/pitches"
#syllables files (.syl) location
sylDir = "data/syls"
#resulting feature files (.ft) location where results will be stored
featureDir = "data/features"

#get the directory reader for geting path bairs  for the wav entry file and the temporary pitch  file
dirReader = st.getDirectoryReader (wavDir,   iExtentions="wav",   outputDir=pitchesDir,  oExtention="f0", openFiles=False)
for wavFile, pitchesFile  in dirReader:
	#overwriting not aloud
	if exists ( pitchesFile): continue
	pitches = st.getReaperPitches( wavFile=wavFile)
	with open(pitchesFile, "wb") as output: pickle.dump( pitches, output)


#similar  approach for extracting temporary syllable files
dirReader = st.getDirectoryReader (wavDir,   iExtentions="wav",   outputDir=sylDir,  oExtention="syl", openFiles=False)
for wavFile, sylFile  in dirReader:
	#get the radicale name, excluding location and extention
	#this radical is shared by all the files refering to the same audio record
	_, fileName, _ = st.parsePath( wavFile )
	#get the corresponding pitch files path
	pitchesFile = st.joinPath( pitchesDir, fileName, "f0")
	#continue if missing pitch file or already existing syllable file
	if exists ( sylFile) or not exists(pitchesFile): continue
	#load audio signal
	signal, sampleRate = st.loadWav( wavFile)
	#extract syllables using the precomputed pitch
	syllables = st.getSyllables( signal, sampleRate, pitchesFile)
	with open(sylFile, "wb") as output: pickle.dump( syllables, output)


#get the directory reader for geting bairs of pathes of the wav entry file and the feature resulting file
dirReader = st.getDirectoryReader (wavDir,   iExtentions="wav",   outputDir=featureDir,  oExtention="ft", openFiles=False)
#In this exemple the extraction is performed on 10s windows overlaped with a 1s step
windowSize  = 10
windowStep = 1
# for each input/ output pair 
for wavFile, ftFile in dirReader:
	#get the radicale name, excluding location and extention
	#this radical is shared by all the files refering to the same audio record
	_, fileName, _ = st.parsePath( wavFile )
	#get the corresponding pitch and syllable files pathes
	pitchesFile = st.joinPath( pitchesDir, fileName, "f0")
	sylFile = st.joinPath( sylDir, fileName, "syl")
	#continue if the result file already exists or the temporary files are missing
	if not exists( pitchesFile) or not exists(sylFile) or exists (ftFile): continue
	#load the audio signal
	signal, sampleRate = st.loadWav( wavFile)
	#run specialised extractions
	pitchFeatures = st.extractPitchFeatures ( signal, sampleRate, pitchesFile=pitchesFile, windowSize=windowSize, windowStep=windowStep)
	energyFeatures = st.extractEnergyFeatures( signal, sampleRate, windowSize=windowSize, windowStep=windowStep)
	snrFeatures = st.extractSnrFeatures ( signal, sampleRate , windowSize=windowSize, windowStep=windowStep)
	syllabicRateFeatures, syllabicDurationFeatures, vowelDurationFeatures = st.extractRhythmFeatures( sylFile=sylFile, windowSize=windowSize, windowStep=windowStep)
	spectralCentroidFeatures, spectralFlatnessFeatures = st.extractSpectralFeatures( signal, sampleRate, windowSize=windowSize, windowStep=windowStep)
	#merge all features in a single matrix
	allFeatures = st.mergeFeatures( pitchFeatures, energyFeatures, snrFeatures, syllabicRateFeatures, syllabicDurationFeatures, vowelDurationFeatures, spectralCentroidFeatures, spectralFlatnessFeatures)
	#if the merged matrix is  empty due to a wav file shorter than the extraction window don't save
	if allFeatures != []: st.saveFeatures( allFeatures, ftFile)

