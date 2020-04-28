"""clustering module containing all clustering functions.

This module contains all clustering related functions. It exposes several unsupervised clustering approaches and various evaluation tools.
This could be considered as a major module in the package.

"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import speechTools as st


def getElbowCurve( features, maxNbClusters=12):
	"""Compute elbow curve to find best number of clusters for KMeans.

	This function runs KMeans clustering for several numbers of clusters. It computes respective inertias for each number of clusters.

	Args:
		features (numpy.ndarray): features for KMeans models fitting
		maxNbClusters (int): Maximum number of clusters to run on(optional, by default = 12)

	Returns:
		list, list, list: Numbers of clusters from 2 to maxNbClusters, respective inertias, respective fitted models

	"""

	inertias = []
	models = []
	nbClustersList = [  i +2 for i in range(maxNbClusters-1)]
	for k in nbClustersList:
		kMeansModel=st.getFittedKMeansModel( features, k)
		inertias.append( kMeansModel.inertia_)
		models.append( kMeansModel)
	return nbClustersList, inertias, models


def plotElbowCurve( nbClustersList, inertias, imgFile=None, show=True):
	"""Plot elbow curve

	This function plots the elbow curve or save the curve on a PNG image.

	Args:
		nbClustersList (list): Number of clusters range
		inertias (list): Respective inertias for the specified numbers of clusters
		imgFile (str, None): If defined, image path for saving (optional)
		show (bool): if True show curve plot(optional)

	"""

	plt.plot( nbClustersList, inertias, "bx-")
	plt.xlabel('Number of clusters')
	plt.ylabel('Inertia')
	plt.title("Elbow curve For finding Optimal number of clusters")
	if imgFile: plt.savefig( imgFile )
	if show: plt.show()

def evaluateClusters( features, labels):
	"""Evaluate clusters quality using silhouette score

	This function uses sklearn sylhouette score for evaluating clusters. Those are represented by a set of features and clustering labels.

	args:
		features (numpy.ndarray): features matrix representing each point in the clustering space
		labels (list): list of respective clustering affectations for each features sample

	returns:
		float: silhouette score of the given clusters

"""

	return silhouette_score( features, labels)


def getNbClusters( model):
	"""Return the number of clusters contained in a fitted clustering model

	This function returns the number of clusters counting the number of different labels provided by the model. This is relevant when the model don't contain a number of cluster attribute.

	Args:
		model(any sklearn clustering model with an labels_ attribute): The used fitted model

	Returns:
		int: the number of clusters in the model

	"""

	labels = model.labels_
	labelValues = []
	for label in labels:
		if label not in labelValues and label != -1: labelValues.append(label)
	nbClusters = len( labelValues)
	return nbClusters


def findBestNbClusters ( nbClustersList, inertias, kMeansModels ):
	"""Find best number of clusters for KMeans

	This function finds the best number of clusters for KMeans using elbow method.
	It finds the number maximizing elbow curve second derivative(acceleration)

	args:
		nbClustersList (list): Number of clusters range
		inertias (list): Respective inertias for the specified numbers of clusters
		kMeansModels (list): respective fitted KMeans models

	returns:
		int, float, sklearn.cluster.KMeans: Best number of clusters, relative inertia, relative KMeans model

	"""
	inertiaAccelerationCurve = np.diff( inertias, 2)
	print(" inertia acceleration")
	for acc in inertiaAccelerationCurve: print (round(acc/1000000000))
	maxAccelerationIndex = np.argmax( inertiaAccelerationCurve)
	bestNbClusters = nbClustersList [maxAccelerationIndex + 1]
	bestInertia = inertias[maxAccelerationIndex + 1]
	bestKMeansModel = kMeansModels[ maxAccelerationIndex +1]
	return bestNbClusters, bestInertia, bestKMeansModel


def getFittedKMeansModel ( features, nbClusters, nbInitialisations = 1000, maxNbIterations=10000):
	"""Returns a feature fitted KMeans model

	This function constructs a KMeans model and fits it on given features

	Args:
		features (numpy.ndarray): features matrix
		nbClusters (int): Number of clusters to find
		nbInitialisations (int): Number of model initialisations (optional)
		maxNbIterations (int): Maximum iterations before convergence (optional)

	Returns:
		sklearn.cluster.KMeans: The fitted model

	"""

	kMeansModel=KMeans(n_clusters=nbClusters, n_init=nbInitialisations, max_iter=maxNbIterations, n_jobs=-1)
	kMeansModel.fit(features)
	return kMeansModel


def getFittedSpectralModel ( features, nbClusters, nbInitialisations=1000):
	"""Returns a feature fitted spectral clustering model

	This function constructs a spectral clustering model and fits it on given features

	Args:
		features (numpy.ndarray): features matrix
		nbClusters (int): Number of clusters to find
		nbInitialisations (int): Number of model initialisations (optional)

	Returns:
		sklearn.cluster.SpectralClustering: The fitted model

	"""

	spectralModel = SpectralClustering( n_clusters=nbClusters, n_init=nbInitialisations, affinity="rbf", assign_labels="discretize")
	spectralModel.fit( features )
	return spectralModel


def getFittedDBScanModel( features, maxDistance, minNbSamples):
	"""Returns a feature fitted DBScan model

	This function constructs a DBScan model and fits it on given features

	Args:
		features (numpy.ndarray): features matrix
		maxDistance (float): Maximum distance DBScan param 
		 minNbSamples (int): The DBScan minimum number of samples param

	Returns:
		sklearn.cluster.DBSCAN: The fitted model

	"""

	dbscanModel = DBSCAN( eps=maxDistance, min_samples=minNbSamples)
	dbscanModel.fit( features )
	return dbscanModel


def getFittedAgglomerativeModel(  features, nbClusters, linkage="average"):
	"""Returns a feature fitted agglomerative model

	This function constructs a agglomerative model and fits it on given features

	Args:
		features (numpy.ndarray): features matrix
		nbClusters (int): Number of clusters to find
linkage (str): sklearn agglomerative model linkage param. could be:"average", "ward", "complete", "single"
	Returns:
		sklearn.cluster.AgglomerativeClustering: The fitted model

	"""

	agglomerativeModel = AgglomerativeClustering( n_clusters=nbClusters, linkage=linkage)
	agglomerativeModel.fit( features)
	return agglomerativeModel


def getBestAgglomerativeModel( features, maxNbClusters=10 ):
	"""Evaluate various agglomerative clustering configuration

	This function runs multiples agglomerative models and returns de best one after printing its main params	

	Args:
		features (numpy.ndarray): Features matrix
		maxNbClusters (int): Maximum number of clusters find during clustering

	Returns:
		sklearn.clustering.AgglomerativeClustering, float: The best fitted model, the best silhouette scor. Returns (False, -1) if all evaluated models are rejected

	"""

	print("Agglomerative model")
	models = []
	linkageMetrics = ["ward", "complete", "average", "single"]
	for nbClusters in range(2, maxNbClusters+1):
		for metric in linkageMetrics: models.append( st.getFittedAgglomerativeModel( features, nbClusters, metric))
	bestModel, bestScore = st.getBestFittedModel( models, features)
	if not bestModel:
		print("Regected all models")
		return False, -1
	print("Score:", bestScore)
	print("Number of clusters:", st.getNbClusters(bestModel))
	print("Linkage:", bestModel.get_params()["linkage"])
	return bestModel, bestScore


def getBestKMeansModel( features, maxNbClusters=10):
	"""Evaluate various KMeans  configuration

	This function runs multiples KMeans models and returns de best one after printing its main params	

	Args:
		features (numpy.ndarray): Features matrix
		maxNbClusters (int): Maximum number of clusters to find during clustering

	Returns:
		sklearn.clustering.KMeans, float: The best fitted model, the best silhouette scor. Returns (False, -1) if all evaluated models are rejected

	"""

	print("KMeans model")
	models = []
	for nbClusters in range(2, maxNbClusters+1):
		models.append( st.getFittedKMeansModel( features, nbClusters))
	bestModel, bestScore = st.getBestFittedModel( models, features)
	if not bestModel:
		print("Regected all models")
		return False, -1
	print("Score:", bestScore)
	print("Number of clusters:", st.getNbClusters(bestModel))
	return bestModel, bestScore


def getBestSpectralModel( features, maxNbClusters=7):
	"""Evaluate various agglomerative clustering configuration

	This function runs multiples spectral models and returns de best one after printing its main params	

	Args:
		features (numpy.ndarray): Features matrix
		maxNbClusters (int): Maximum number of clusters to find during clustering

	Returns:
		sklearn.clustering.SpectraleClustering, float: The best fitted model, the best silhouette scor. Returns (False, -1) if all evaluated models are rejected

	"""

	print("Spectral Model")
	models = []
	for nbClusters in range(2, maxNbClusters+1):
		models.append( st.getFittedSpectralModel( features, nbClusters))
	bestModel, bestScore = st.getBestFittedModel( models, features)
	if not bestModel:
		print("Regected all models")
		return False, -1
	print("Score:", bestScore)
	print("Number of clusters:", st.getNbClusters(bestModel))
	return bestModel, bestScore

def getBestDBScanModel ( features):
	"""Evaluate various DBScan clustering configuration

	This function runs multiples DBScan models and returns de best one after printing its main params	

	Args:
		features (numpy.ndarray): Features matrix

	Returns:
		sklearn.clustering.DBSCAN, float: The best fitted model, the best silhouette scor. Returns (False, -1) if all evaluated models are rejected

	"""

	print("DBScan model")
	models = []
	for nbSamples in range( 2, len(features)//4):
		nbSamples *= 2
		for distance in range( 1, 26):
			distance /= 50
			models.append( st.getFittedDBScanModel( features, distance, nbSamples))
	bestModel, bestScore = st.getBestFittedModel( models, features)
	if not bestModel:
		print("Regected all models")
		return False, -1
	print("Score:", bestScore)
	print("Number of clusters:", st.getNbClusters(bestModel))
	print("Max distence:", bestModel.get_params()["eps"])
	print("Min number of samples", bestModel.get_params()["min_samples"])
	return bestModel, bestScore


def getBestFittedModel( models, features ):
	"""Find the best clustering model according to silhouette score

	This function takes a list of clustering models, and select the best fitted model using silhouette score evaluation on given features.

	Args:
		models (list): List of fitted clustering models that contains a label_ atribute
		features (numpy.ndarray): features matrix to evaluate on

	returns:
		sklearn model, float: best fitted clustering model, best score

"""

	validModels = []
	clusteringScores = []
	for model in models:
		#Skip mono cluster models
		if st.getNbClusters( model ) < 2: continue
		validModels.append( model )
		labels = model.labels_
		clusteringScore = evaluateClusters(features, labels)
		clusteringScores.append( clusteringScore)
	if len(clusteringScores) == 0: return False, -1
	bestScoreIndex = np.argmax(clusteringScores)
	return validModels[bestScoreIndex], clusteringScores[bestScoreIndex]


def getClusteringSequences( clusteringLabels, windowSize, windowStep, withOverlaping=False):
	"""Returns the audio sequences or intervals based on clustering labels

	This function concatenates the consecutive clustering labels on audio sequences with a start end end times.

	Args:
		clusteringLabels (list): consecutive and ordered clustering affectations
		windowSize (float): size of audio window each label refers to, in second
		windowStep (float): hop step of 2 consecutive audio windows, in second
		withOverlaping (bool): if true aloud sequence overlaping when windowSize > windowStep

	Returns:
		list: pairs of labels and intervals: [...,[labelID,[startTime, endTime],...]

	"""

	if not withOverlaping: windowSize = windowStep
	sequences = []
	currentSequence = [0, windowSize]
	for l in range( 1, len(clusteringLabels)):
		currentLabel = clusteringLabels[l]
		previousLabel = clusteringLabels[l-1]
		if currentLabel == previousLabel: continue
		currentSequence[1] = ((l-1) * windowStep) + windowSize
		sequences.append([previousLabel, currentSequence])
		currentSequence= [ l * windowStep, (l*windowStep) + windowSize]
	currentSequence[1] = (l*windowStep) + windowSize
	sequences.append( [clusteringLabels[-1], currentSequence])
	return sequences



def rectifyClusteringLabels( labels, rectificationWidth=5):
	"""Supress obvious cluster labeling errors

	This function read the consecutive clustering labels and lookes for paterns [a,a,...,a,a,b,a,a,...,a,a], where "a" and "b" ar different cluster labels. In this case the label "b" is replaced by "a".

	Args:
		labels (list): Consecutive clustering labels
		rectificationWidth (int): Number of labels in the rectification patern, should be an odd number

	Returns: 
		list: The rectified labels returning the same arg object without copying 

	"""

	for l in range( len(labels) - rectificationWidth):
		firstLabelIndex = l
		medianLabelIndex = l + ( rectificationWidth//2)
		lastLabelIndex = l + rectificationWidth -1
		firstLabel = labels[firstLabelIndex]
		medianLabel = labels[ medianLabelIndex]
		lastLabel = labels[lastLabelIndex]
		leftLabels = labels[firstLabelIndex +1 : medianLabelIndex]
		rightLabels = labels[ medianLabelIndex +1 : lastLabelIndex]
		if medianLabel != firstLabel and medianLabel != lastLabel:
			for label in leftLabels:
				if label != firstLabel: continue
			for label in rightLabels:
				if label != lastLabel: continue
			labels[medianLabelIndex] = labels[medianLabelIndex - 1]
	return labels


