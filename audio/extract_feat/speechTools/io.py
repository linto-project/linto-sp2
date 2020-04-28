"""Io module for file processing.

This module contains functions for processing, reading, writing and searching data files.

"""


import numpy as np
from scipy.io import wavfile
import librosa
import sys
from os import listdir
from os.path import isfile, join, dirname, basename, exists
import speechTools as st



def getDirectoryReader (inputDir,  iMode = "rb", iExtentions=None, outputDir=None, oMode = "wb", oExtention=None, openFiles=True):
	"""Provide a manager for easily pipelining input and output files throug a transformation process

	This function provides a python generator for looping on pairs of input and output files. For each loop a pair of input and output files (pathes or opened files) is yielded.
	The files could be filtered using input and output locations, and input and output extentions.
	The openFiles param chooses if pathes or opened files should be yielded. In the case of opened files, the opening mode could be specified.
	Using the generator helps for integrating all the files processing in a simple for-loop without handeling directly the file management.

	Args:
		inputDir (str): The input files directory
		iMode (str): The file opening mode
		iExtentions (str or list ): The required input files extentions (a string or list of strings without the dot)
		outputDir (str): The outputFiles directory (should already exists )
		oMode (str): The output files opening Mode
		oExtention (str): The output files extention 
		openFiles (bool): If false the pair of files is not opened and only paths are yielded

	Yields:
		file, file: The input file, the output file (on not opening mode only a pair of pathes is yielded)

	"""

	if iExtentions:
		if type(iExtentions) != list: iExtentions = [iExtentions]
	if not oExtention: oExtention = ""
	inputFiles = [fileName for fileName in listdir( inputDir ) if isfile(join(inputDir, fileName)) and not fileName[0] == "."]
	for fileName in inputFiles:
		dotPosition = -1
		for i in range(len(fileName)):
			if fileName[i] == ".": dotPosition = i
		if dotPosition == -1:
			if iExtentions: continue
			if outputDir: outputFile = fileName + "." + oExtention
		else:
			extention = fileName[dotPosition+1:]
			if extention not in iExtentions: continue
			if outputDir: outputFile = fileName[:dotPosition+1] + oExtention
		inputFile = join(inputDir , fileName)
		if outputDir:
			outputFile = join( outputDir, outputFile)
			if exists( outputFile): continue
		if openFiles:
			with open( inputFile, iMode) as input:
				if outputDir:
					with open ( outputFile, oMode) as output:
						yield input, output
				else: yield input
		else:
			if outputDir: yield inputFile, outputFile
			else: yield inputFile




def findFiles (location,  extentions=[]):
	"""Return  all the files in the specified location and containing the listed extentions

	Args:
		location (str): The directory where to search the files
		extentions (str or list ): The required files extentions (a string or list of strings without the dot)

	Returns:
		list: The list of pathes of the found files

	"""

	selectedFiles = []
	if  type(extentions) != list: extentions = [extentions]
	elements = [fileName for fileName in listdir( location ) if  not fileName[0] == "."]
	for fileName in elements:
		currentPath = join(location, fileName)
		if not isfile( currentPath):
			selectedFiles.extend( findFiles( currentPath, extentions))
			continue
		dir, name, extention = parsePath( currentPath )
		if extentions and (not extention or extention not in extentions): continue
		selectedFiles.append( currentPath)
	return selectedFiles


def parsePath ( path ):
	"""Parse a path on 3 strings: location, radical name, extention

	Args:
		Path (str): The path to parse

	Returns:
		str, str, str: The directory path, the radical name, the extention without the dot

	"""

	dir = dirname( path )
	fileName = basename( path)
	extention = None
	dotPosition = -1
	for i in range(1, len(fileName)):
		if fileName[i] == ".": dotPosition = i
	if dotPosition != -1:
		extention =fileName[dotPosition+1:]
		fileName = fileName[:dotPosition]
	return dir, fileName, extention


def joinPath( dir, fileName, extention=None):
	"""Join a directory path, a radical name, and a extention  within a formated Unix path

	Args:
		dir (str): The directory path
		fileName (str): The radical name
		extention (str): The file extention without the dot (optional)

	Returns:
		str: The joined path

	"""

	if extention: path =  join(dir, fileName + "." + extention)
	else: path = join(dir, fileName)
	return path


def loadWav( fileName, compress=True):
	"""Open a .wav file and extract the audio signal

	Args:
	fileName (str): The .wav file path
		compress (bool): If true force sampling to 16khz and convert stereo to mono

	Returns:
		numpy.array, int: The non normalised int16 audio signal, the signal sample rate

	"""

	if compress: signal, sampleRate = librosa.load( fileName, sr=16000, mono=True, dtype=np.float32, res_type="kaiser_fast")
	else: signal, sampleRate = librosa.load( fileName, sr=None)
	signal = np.int16(signal * (2**15))
	return signal, sampleRate

def writeWav ( fileName, signal, sampleRate=16000):
	"""Write an audio signal in a .wav file

	Args:
		fileName (str): The destination file path
		signal (numpy.array): The audio signal
		sampleRate (int): The signal sample rate

	"""

	if signal.dtype != np.int16: signal = np.int16( signal )
	wavfile.write(fileName, sampleRate, signal)


