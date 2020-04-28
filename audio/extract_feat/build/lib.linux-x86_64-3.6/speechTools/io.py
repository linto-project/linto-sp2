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
	if extention: path =  join(dir, fileName + "." + extention)
	else: path = join(dir, fileName)
	return path


def loadWav( fileName, compress=True):
	if compress: signal, sampleRate = librosa.load( fileName, sr=16000, mono=True, dtype=np.float32, res_type="kaiser_fast")
	else: signal, sampleRate = librosa.load( fileName, sr=None)
	signal = np.int16(signal * (2**15))
	return signal, sampleRate

def writeWav ( fileName, signal, sampleRate=16000):
	if signal.dtype != np.int16: signal = np.int16( signal )
	wavfile.write(fileName, sampleRate, signal)


