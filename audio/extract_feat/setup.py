from setuptools import setup, Extension, find_packages


setup (
	name = "speechTools",
	version = 0.1,
	packages = find_packages(),
	author = "Yassir Bouiry",
	description = "Automatic speech analysis in meeting context",
	long_description = open("README.md").read(),
	install_requires = ["scipy",
	"numpy" ],
	include_package_data = True,
	classifiers = [
		"Prosody",
		"natural speech",
		"voice pitch f0",
		"syllable extraction"],
	ext_modules=[
		Extension('speechTools.diverg', ['speechTools/diverg/diverg.c', 'speechTools/diverg/subdiv.c'])],
	entry_points = {"console_scripts":["getSpeechFeatures = speechTools.core:getSpeechFeatures"]},
	license = "IRIT/SAMoVA",
	zip_safe = True)

