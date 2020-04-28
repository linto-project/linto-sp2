"""trans module for reading transcription files.

This module contains functions for loading transcription information from TRS files.

"""


from lxml import etree
import speechTools as st


def getTrans( fileName ):
	"""Return the element tree represented by a trs transcription file

	Args:
		fileName (str): The trs file path

	Returns:
		lxml.etree._ElementTree: The parsed data tree

	"""

	trans = etree.parse( fileName)
	return trans


def getSpeakers ( trans ):
	"""Find all the speakers in a transcription tree

	Args:
		trans (lxml.etree._ElementTree): The transcription data tree

	Returns:
		dict: The dictionary containing speakers information with the form {..."speakerID": "speakerName",...}

	"""

	speakers = {}
	for speaker in trans.xpath( "/Trans/Speakers/Speaker"):
		speakerId = speaker.get("id")
		speakerName = speaker.get( "name")
		speakers[speakerId] = speakerName
	return speakers


def getTurns ( trans ):
	"""Find all the turns in a transcription tree

	Args:
		trans (lxml.etree._ElementTree): The transcription data tree

	Returns:
		list: The found turns with the form [...[startTime, endTime, speakerID],...]

	"""

	turns = []
	for turn in trans.xpath("/Trans/Episode/Section/Turn[@speaker]"):
		startTime = float(turn.get("startTime"))
		endTime = float(turn.get("endTime"))
		speaker = turn.get("speaker")
		turns.append( [startTime, endTime, speaker])
	return turns


def getTurnsDistribution ( speakers, turns):
	"""allocate each turn to its speaker

	Args:
		speakers (dict): The speakers data as returned by speechTools.trans.getSpeakers
		turns (list): The turns data as returned by speechTools.trans.getTurns

	Returns:
		dict: The turns distribution with the form {..."userId":[...[startTime, endTime],...],...}

	"""

	turnsDistribution = {}
	for speakerId in speakers: turnsDistribution [speakerId] = []
	for turn in turns:
		turnsDistribution[turn[h2]].append( [turn[0], turn[1]])
	return turnsDistribution


def getSilences ( trans ):
	"""Find all the silences in a transcription tree defined as turns without speaker

	Args:
		trans (lxml.etree._ElementTree): The transcription data tree

	Returns:
		list: The found turns with the form [...[startTime, endTime],...]

	"""

	turns = []
	for turn in trans.xpath("/Trans/Episode/Section/Turn[not(@speaker)]"):
		startTime = float(turn.get("startTime"))
		endTime = float(turn.get("endTime"))
		turns.append( [startTime, endTime])
	return turns



