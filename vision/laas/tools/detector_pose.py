import cv2
import time
import numpy as np


class poseDetectorAPI:
	def __init__(self, protoFile, weightsFile, nPoints):
		self.protoFile = protoFile
		self.weightsFile = weightsFile
		self.nPoints = nPoints
		self.net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
		self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

		self.detected_keypoints = []

		self.frameWidth = 0
		self.frameHeight = 0

		# COCO Output Format
		self.keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee',
		                         'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

		self.POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
		                   [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
		                   [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
		                   [2, 17], [5, 16]]

		# index of pafs correspoding to the POSE_PAIRS
		# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
		self.mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44],
		               [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
		               [47, 48], [49, 50], [53, 54], [51, 52], [55, 56],
		               [37, 38], [45, 46]]

		self.colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255],
		               [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255],
		               [0, 0, 255], [255, 0, 0], [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]

	def getKeypoints(self, probMap, threshold=0.1):

		mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)

		mapMask = np.uint8(mapSmooth > threshold)
		keypoints = []

		# find the blobs
		_, contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		# for each blob find the maxima
		for cnt in contours:
			blobMask = np.zeros(mapMask.shape)
			blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
			maskedProbMap = mapSmooth * blobMask
			_, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
			keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

		return keypoints

	# Find valid connections between the different joints of a all persons present
	def getValidPairs(self, output):
		valid_pairs = []
		invalid_pairs = []
		n_interp_samples = 10
		paf_score_th = 0.1
		conf_th = 0.7
		# loop for every POSE_PAIR
		for k in range(len(self.mapIdx)):
			# A->B constitute a limb
			pafA = output[0, self.mapIdx[k][0], :, :]
			pafB = output[0, self.mapIdx[k][1], :, :]
			pafA = cv2.resize(pafA, (self.frameWidth, self.frameHeight))
			pafB = cv2.resize(pafB, (self.frameWidth, self.frameHeight))

			# Find the keypoints for the first and second limb
			candA = self.detected_keypoints[self.POSE_PAIRS[k][0]]
			candB = self.detected_keypoints[self.POSE_PAIRS[k][1]]
			nA = len(candA)
			nB = len(candB)

			# If keypoints for the joint-pair is detected
			# check every joint in candA with every joint in candB
			# Calculate the distance vector between the two joints
			# Find the PAF values at a set of interpolated points between the joints
			# Use the above formula to compute a score to mark the connection valid

			if (nA != 0 and nB != 0):
				valid_pair = np.zeros((0, 3))
				for i in range(nA):
					max_j = -1
					maxScore = -1
					found = 0
					for j in range(nB):
						# Find d_ij
						d_ij = np.subtract(candB[j][:2], candA[i][:2])
						norm = np.linalg.norm(d_ij)
						if norm:
							d_ij = d_ij / norm
						else:
							continue
						# Find p(u)
						interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
						                        np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
						# Find L(p(u))
						paf_interp = []
						for k in range(len(interp_coord)):
							paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
							                   pafB[
								                   int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))]])
						# Find E
						paf_scores = np.dot(paf_interp, d_ij)
						avg_paf_score = sum(paf_scores) / len(paf_scores)

						# Check if the connection is valid
						# If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
						if (len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples) > conf_th:
							if avg_paf_score > maxScore:
								max_j = j
								maxScore = avg_paf_score
								found = 1
					# Append the connection to the list
					if found:
						valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

				# Append the detected connections to the global list
				valid_pairs.append(valid_pair)
			else:  # If no keypoints are detected
				# print("No Connection : k = {}".format(k))
				invalid_pairs.append(k)
				valid_pairs.append([])
		return valid_pairs, invalid_pairs

	# This function creates a list of keypoints belonging to each person
	# For each detected valid pair, it assigns the joint(s) to a person
	def getPersonwiseKeypoints(self, valid_pairs, invalid_pairs):
		# the last number in each row is the overall score
		personwiseKeypoints = -1 * np.ones((0, 19))

		for k in range(len(self.mapIdx)):
			if k not in invalid_pairs:
				partAs = valid_pairs[k][:, 0]
				partBs = valid_pairs[k][:, 1]
				indexA, indexB = np.array(self.POSE_PAIRS[k])

				for i in range(len(valid_pairs[k])):
					found = 0
					person_idx = -1
					for j in range(len(personwiseKeypoints)):
						if personwiseKeypoints[j][indexA] == partAs[i]:
							person_idx = j
							found = 1
							break

					if found:
						personwiseKeypoints[person_idx][indexB] = partBs[i]
						personwiseKeypoints[person_idx][-1] += self.keypoints_list[partBs[i].astype(int), 2] + \
						                                       valid_pairs[k][i][2]

					# if find no partA in the subset, create a new subset
					elif not found and k < 17:
						row = -1 * np.ones(19)
						row[indexA] = partAs[i]
						row[indexB] = partBs[i]
						# add the keypoint_scores for the two keypoints and the paf_score
						row[-1] = sum(self.keypoints_list[valid_pairs[k][i, :2].astype(int), 2]) + valid_pairs[k][i][2]
						personwiseKeypoints = np.vstack([personwiseKeypoints, row])
		return personwiseKeypoints

	def processFrame(self, image, threshold=0.1):

		# Init parameters if needed
		if self.frameWidth == 0:
			self.frameWidth = image.shape[1]
			self.frameHeight = image.shape[0]
			# Fix the input Height and get the width according to the Aspect Ratio
			self.inHeight = 368 #120 #368
			self.inWidth = int((self.inHeight / self.frameHeight) * self.frameWidth)

		t = time.time()
		inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (self.inWidth, self.inHeight), (0, 0, 0), swapRB=False,
		                                crop=False)

		self.net.setInput(inpBlob)
		self.output = self.net.forward()
		print("Time Taken in forward pass = {}".format(time.time() - t))

		self.detected_keypoints = []
		self.keypoints_list = np.zeros((0, 3))
		keypoint_id = 0

		for part in range(self.nPoints):
			probMap = self.output[0, part, :, :]
			probMap = cv2.resize(probMap, (image.shape[1], image.shape[0]))
			keypoints = self.getKeypoints(probMap, threshold)
			# print("Keypoints - {} : {}".format(self.keypointsMapping[part], keypoints))
			keypoints_with_id = []
			for i in range(len(keypoints)):
				keypoints_with_id.append(keypoints[i] + (keypoint_id,))
				self.keypoints_list = np.vstack([self.keypoints_list, keypoints[i]])
				keypoint_id += 1

			self.detected_keypoints.append(keypoints_with_id)

		valid_pairs, invalid_pairs = self.getValidPairs(self.output)
		self.personwiseKeypoints = self.getPersonwiseKeypoints(valid_pairs, invalid_pairs)

	def processBoxes(self, image, boxes, threshold=0.1):

		self.detected_keypoints = []
		self.keypoints_list = np.zeros((0, 3))
		keypoint_id = 0

		for b in boxes:

			if b[0] == 0 and b[1] == 0 and b[2] == 0:
				continue
			w = image.shape[1]
			h = image.shape[0]
			y1 = max(b[0],0)
			x1 = max(b[1],0)
			y2 = min(b[2],h)
			x2 = min(b[3],w)
			person = image[y1:y2, x1:x2]

			try:
				self.frameWidth = person.shape[1]
				self.frameHeight = person.shape[0]
				# Fix the input Height and get the width according to the Aspect Ratio
				self.inHeight = 368#368
				self.inWidth = int((self.inHeight / self.frameHeight) * self.frameWidth)

				inpBlob = cv2.dnn.blobFromImage(person, 1.0 / 255, (self.inWidth, self.inHeight), (0, 0, 0), swapRB=False, crop=False)

				self.net.setInput(inpBlob)
				self.output = self.net.forward()
			except:
				continue

			for part in range(self.nPoints):
				probMap = self.output[0, part, :, :]
				probMap = cv2.resize(probMap, (person.shape[1], person.shape[0]))
				keypoints = self.getKeypoints(probMap, threshold)
				# print("Keypoints - {} : {}".format(self.keypointsMapping[part], keypoints))
				keypoints_with_id = []
				for i in range(len(keypoints)):
					keypoints_with_id.append(keypoints[i] + (keypoint_id,))
					self.keypoints_list = np.vstack([self.keypoints_list, keypoints[i]])
					keypoint_id += 1

				self.detected_keypoints.append(keypoints_with_id)

			valid_pairs, invalid_pairs = self.getValidPairs(self.output)
			self.personwiseKeypoints = self.getPersonwiseKeypoints(valid_pairs, invalid_pairs)

			self.draw(person)
			self.detected_keypoints = []
			self.keypoints_list = np.zeros((0, 3))
			keypoint_id = 0

	def draw(self, image):

		if self.detected_keypoints == []:
			return

		for i in range(self.nPoints):
			for j in range(len(self.detected_keypoints[i])):
				cv2.circle(image, self.detected_keypoints[i][j][0:2], 5, self.colors[i], -1, cv2.LINE_AA)



		for i in range(17):
			for n in range(len(self.personwiseKeypoints)):
				index = self.personwiseKeypoints[n][np.array(self.POSE_PAIRS[i])]
				if -1 in index:
					continue
				B = np.int32(self.keypoints_list[index.astype(int), 0])
				A = np.int32(self.keypoints_list[index.astype(int), 1])
				cv2.line(image, (B[0], A[0]), (B[1], A[1]), self.colors[i], 3, cv2.LINE_AA)

	def close(self):
		self.sess.close()
		self.default_graph.close()
