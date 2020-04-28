"""trans module for reading transcription files.

This module contains functions for loading transcription information from TRS files.

"""


from lxml import etree
import speechTools as st


def getTrans( fileName ):
	trans = etree.parse( fileName)
	return trans


def getSpeakers ( trans ):
	speakers = {}
	for speaker in trans.xpath( "/Trans/Speakers/Speaker"):
		speakerId = speaker.get("id")
		speakerName = speaker.get( "name")
		speakers[speakerId] = speakerName
	return speakers


def getTurns ( trans ):
	turns = []
	for turn in trans.xpath("/Trans/Episode/Section/Turn[@speaker]"):
		startTime = float(turn.get("startTime"))
		endTime = float(turn.get("endTime"))
		speaker = turn.get("speaker")
		turns.append( [startTime, endTime, speaker])
	return turns


def getTurnsDistribution ( speakers, turns):
	turnsDistribution = {}
	for speakerId in speakers: turnsDistribution [speakerId] = []
	for turn in turns:
		turnsDistribution[turn[h2]].append( [turn[0], turn[1]])
	return turnsDistribution


def getSilenceSegments ( trans ):
	turns = []
	for turn in trans.xpath("/Trans/Episode/Section/Turn[not(@speaker)]"):
		startTime = float(turn.get("startTime"))
		endTime = float(turn.get("endTime"))
		turns.append( [startTime, endTime])
	return turns



