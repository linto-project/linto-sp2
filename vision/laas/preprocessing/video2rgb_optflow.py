import numpy as np
import cv2
import os
import glob
from lxml import etree
import librosa


def video_2_rgb_optflow_audio(video_path, gt_path, max_samples=1600, use_detector=True, start_sec = 0):

	root = video_path if not video_path[-1] == '/' else video_path[:-1]
	count = 0
	optflow = OpticalFlowEstimation()

	if use_detector:
		head_dt = HeadDetector()

	# Iterate over all folders
	for (dirpath, dirnames, filenames) in os.walk(video_path):
		# Check if a folder name video exists
		if 'video' not in dirpath or 'IS1' not in dirpath:
			continue
		# Iterate over each video
		for video_name in filenames:
			# Process only Closeup videos
			if 'Closeup' not in video_name:
				continue

			# Split video name and process only avi files
			vid_idx, cam_idx, suffix = video_name.split('.')
			if not suffix == 'avi':
				continue
			print(dirpath+'/'+video_name)

			save_path_silence  = dirpath[:-5].replace(root, root + '_wholeshort_v2_Silence') + '/'
			save_path_speaking = dirpath[:-5].replace(root, root + '_wholeshort_v2_Speaking') + '/'
			if not os.path.exists(save_path_silence   + '/' + cam_idx + '/' ):
				os.makedirs(save_path_silence   + '/' + cam_idx + '/' )
				os.makedirs(save_path_speaking  + '/' + cam_idx + '/' )
			else:
				continue

			if '1' in cam_idx:
				gt_idx = 'A'
			elif '2' in cam_idx:
				gt_idx = 'B'
			elif '3' in cam_idx:
				gt_idx = 'C'
			else:
				gt_idx = 'D'

			gt_file_path = gt_path + '/' + vid_idx + '.' + gt_idx + '.segments.xml'
			gt_file = etree.parse(gt_file_path)
			gt = getTurns_ami(gt_file)

			video_file = dirpath + '/' + video_name
			cap = cv2.VideoCapture(video_file)
			n_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
			fps = int(cap.get(cv2.CAP_PROP_FPS))
			fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			fps_count = 0
			fps_count_silence  = 0
			fps_count_speaking = 0

			optflow.clean()

			audio_file = dirpath[:-6] + '/audio/' + vid_idx + '.Mix-Headset.wav'
			audio_seq, sr = librosa.load(audio_file, sr=16000)
			audio_step = sr//fps
			start_frame = start_sec*fps
			for i in range(n_f):

				if fps_count_speaking >= max_samples and fps_count_silence >= max_samples:
					break

				print(str(i) + '/' + str(n_f))
				ret, frame = cap.read()
				if not ret:
					break
				if i < start_frame:
					continue

				t_stamp = i/fps
				speaking = within_range(gt, t_stamp, 2.)

				if (speaking and fps_count_speaking >= max_samples) or (not speaking and fps_count_silence >= max_samples):
					continue

				if speaking:
					save_name = save_path_speaking + '/' + cam_idx + '/' + '{:06d}'.format(fps_count_speaking)
				else:
					save_name = save_path_silence + '/' + cam_idx + '/' + '{:06d}'.format(fps_count_silence)

				if use_detector:
					head_dt.apply_detector(fps_count, frame)

					if len(head_dt.tracker.trackers) != 1 or head_dt.tracker.trackers[0].box_head is None:
						continue

					x, y, w, h = head_dt.tracker.trackers[0].box_head

					if w == 0 or h == 0:
						continue

					flow = optflow.apply(frame)

					# x,y,w,h = detector.faces_det[0][0]
					x1 = int(max(x, 0))
					y1 = int(max(y, 0))
					x2 = int(min(x + w, fw))
					y2 = int(min(y + h, fh))
					head = frame[y1:y2, x1:x2]
					head = cv2.resize(head, (224, 224))
					flow_head = flow[y1:y2, x1:x2]
					flow_head = cv2.resize(flow_head, (224, 224))
					cv2.imshow("head", head)
					cv2.imwrite(save_name + '.jpg', head)

					flow_head = cv2.normalize(flow_head, None, 0, 255, cv2.NORM_MINMAX)
					cv2.imwrite(save_name + '_flow.jpg', flow_head)
				else:
					cv2.imwrite(save_name + '.jpg', frame)
					flow = optflow.apply(frame)
					flow = cv2.normalize(flow, None, 0, 255, cv2.NORM_MINMAX)
					cv2.imwrite(save_name + '_flow.jpg', flow)


				if speaking:
					txt = 'Speaking'
					fps_count_speaking += 1
				else:
					txt = 'Silence'
					fps_count_silence += 1

				audio = audio_seq[i*audio_step:(i+1)*audio_step]
				librosa.output.write_wav(save_name + '.wav', audio, sr)

				if use_detector:
					head_dt.tracker.draw_trackers(frame)
				cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
				cv2.imshow("preview", frame)
				key = cv2.waitKey(1)

			if use_detector:
				head_dt.clean()


def video_2_rgb_optflow(video_path, gt_path = None, max_samples=1600, use_detector=True, start_sec = 0):

	root = video_path if not video_path[-1] == '/' else video_path[:-1]
	count = 0
	optflow = OpticalFlowEstimation()

	if use_detector:
		head_dt = HeadDetector()

	# Iterate over all folders
	for (dirpath, dirnames, filenames) in os.walk(video_path):
		# Check if a folder name video exists
		if not any("avi" in s for s in filenames) :
			continue
		# Iterate over each video
		for video_name in filenames:
			# Split video name and process only avi files
			vid_idx, suffix = video_name.split('.')
			if not suffix == 'avi':
				continue
			print(dirpath+'/'+video_name)

			save_path = dirpath.replace('_data', '_rgb_of_heads') + '/'
			if not os.path.exists(save_path  + '/'):
				os.makedirs(save_path + '/' )
			else:
				continue

			video_file = dirpath + '/' + video_name
			cap = cv2.VideoCapture(video_file)
			n_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
			fps = int(cap.get(cv2.CAP_PROP_FPS))
			fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			fps_count = 0

			optflow.clean()

			start_frame = start_sec*fps
			for i in range(n_f):

				if fps_count >= max_samples and fps_count >= max_samples:
					break

				print(str(i) + '/' + str(n_f))
				ret, frame = cap.read()
				if not ret:
					break
				if i < start_frame:
					continue

				t_stamp = i/fps

				save_name = save_path + '/' +  '{:06d}'.format(fps_count)

				if use_detector:
					head_dt.apply_detector(fps_count, frame)

					# if len(head_dt.tracker.trackers) != 1 or head_dt.tracker.trackers[0].box_head is None:
					if head_dt.tracker.trackers[0].box_head is None:
							continue

					x, y, w, h = head_dt.tracker.trackers[0].box_head

					if w == 0 or h == 0:
						continue

					flow = optflow.apply(frame)

					# x,y,w,h = detector.faces_det[0][0]
					x1 = int(max(x, 0))
					y1 = int(max(y, 0))
					x2 = int(min(x + w, fw))
					y2 = int(min(y + h, fh))
					head = frame[y1:y2, x1:x2]
					head = cv2.resize(head, (224, 224))
					flow_head = flow[y1:y2, x1:x2]
					flow_head = cv2.resize(flow_head, (224, 224))
					cv2.imshow("head", head)
					cv2.imwrite(save_name + '.jpg', head)

					flow_head = cv2.normalize(flow_head, None, 0, 255, cv2.NORM_MINMAX)
					cv2.imwrite(save_name + '_flow.jpg', flow_head)
				else:
					cv2.imwrite(save_name + '.jpg', frame)
					flow = optflow.apply(frame)
					flow = cv2.normalize(flow, None, 0, 255, cv2.NORM_MINMAX)
					cv2.imwrite(save_name + '_flow.jpg', flow)

				fps_count += 1

				if use_detector:
					head_dt.tracker.draw_trackers(frame)
				cv2.imshow("preview", frame)
				key = cv2.waitKey(1)

			if use_detector:
				head_dt.clean()


class HeadDetector:
	def __init__(self):

		from tools.linToDetector import linToDetector
		from tools.tracker_cv2 import cvTrackerAPI

		self.detector = linToDetector()
		self.tracker = cvTrackerAPI()
		self.tracker.tracker_remove_thr = 1

	def clean(self):
		self.tracker.clean()
		self.detector.clean()

	def apply_detector(self, fps_count, frame):
		# Detect all elements (person, face, gesture)
		if (fps_count % 1) == 0:
			self.detector.processFrame(frame, detFace=True, detShapes=False, regFace=False, poseEstim=False)
			self.tracker.processFrame(frame, self.detector.people_det[0], self.detector.faces_det[0])
		else:
			self.tracker.processFrame(frame)
			self.detector.clean()

			targets = self.tracker.get_box_p1p2()
			heads = self.tracker.get_box_head_p1p2()
			self.detector.processBoxes(frame, targets, heads, detFace=False, detShapes=False, regFace=False,
								  poseEstim=False)


class OpticalFlowEstimation:
	def __init__(self):
		self.next = None
		self.prvs = None
		self.hsv  = None

	def clean(self):
		self.next = None
		self.prvs = None
		self.hsv  = None

	def apply(self, frame):
		# Apply optical flow
		if self.prvs is not None:
			self.next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			flow = cv2.calcOpticalFlowFarneback(self.prvs, self.next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
			mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
			im_of = np.dstack((flow, mag))
			# im_of *= 16
			im_of = cv2.normalize(im_of, None, -1, 1, cv2.NORM_MINMAX)
			self.hsv[..., 0] = ang * 180 / np.pi / 2
			self.hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
			bgr = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)
			cv2.imshow('frame', frame)
			cv2.imshow('frame op', bgr)
			cv2.imshow('frame new flow', im_of)
			k = cv2.waitKey(1) & 0xff
			# if k == 27:
			# 	break
			self.prvs = self.next
		else:
			self.prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			im_of = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.float32)
			self.hsv = np.zeros_like(frame)
			self.hsv[..., 1] = 255
		return im_of


def getTurns_ami ( trans ):
	turns = []
	for turn in trans.xpath("segment"):
		startTime = float(turn.get("transcriber_start"))
		endTime = float(turn.get("transcriber_end"))
		turns.append( [startTime, endTime])
	return turns


def within_range( sequence, value, min_dist=1.):
	for lower, upper in sequence:
		if value > upper: continue
		if upper - lower > min_dist and (lower - 0.) < value < (upper + 0.):
			return True
		else:
			return False
	return False


if __name__ == '__main__':
	# video_2_rgb_optflow_audio('/data/jfmadrig/amicorpus/5fold_valid/', '/data/jfmadrig/amicorpus/annotations_manual/segments',
	# 					max_samples=960, use_detector=True, start_sec=300)

	video_2_rgb_optflow('/data/jfmadrig/hpedatasets/ict3DHP_data/')

