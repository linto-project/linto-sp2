# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import numpy as np
import cv2
import dlib
from imutils import face_utils
import time


class dlibLandmarkDetectorAPI:
    def __init__(self, landmark_model, scale = 1):
        self.landmark_model = landmark_model
        self.scale = int(scale)

        self.detector  = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(landmark_model)

    def processFrame(self, image):

        # Ask the detector to find the bounding boxes of each face. The # in the
        # second argument indicates that we should upsample the image # time. This
        # will make everything bigger and allow us to detect more faces.
        level = self.scale
        dets = self.detector(image, level)

        boxes_list = []
        shape_list = []

        # loop over the face detections
        for (i, rect) in enumerate(dets):
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)

            boxes_list.append([y, x, y + h, x + w])

            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = self.predictor(image, rect)
            shape = face_utils.shape_to_np(shape)
            shape_list.append(shape)

        return (boxes_list, shape_list)


    def processFrameSeg(self, image, boxes):

        boxes_list = []
        shape_list = []
        level = self.scale

        for (i, b) in enumerate(boxes):

            if b[0] == 0 and b[1] == 0 and b[2] == 0:
                break
            person = image[b[0]:b[2], b[1]:b[3]]
            # Ask the detector to find the bounding boxes of each face. The # in the
            # second argument indicates that we should upsample the image # time. This
            # will make everything bigger and allow us to detect more faces.
            dets = self.detector(person, level)

            # loop over the face detections
            for (j, rect) in enumerate(dets):
                # convert dlib's rectangle to a OpenCV-style bounding box
                # [i.e., (x, y, w, h)], then draw the face bounding box
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                boxes_list.append([y + b[0], x + b[1], y + h + b[0], x + w + b[1]])

                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = self.predictor(person, rect)
                shape = face_utils.shape_to_np(shape)

                for k in range(len(shape)):
                    shape[k][0] = shape[k][0] + b[1]
                    shape[k][1] = shape[k][1] + b[0]
                shape_list.append(shape)

        return (boxes_list, shape_list)

    def findFace(self, image, boxes):
        shape_list = []
        for (i, b) in enumerate(boxes):
            if b == []:
                shape_list.append([])
                continue

            rect = dlib.rectangle(b[1], b[0], b[3], b[2])
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = self.predictor(image, rect)
            shape = face_utils.shape_to_np(shape)
            shape_list.append(shape)

        return (shape_list)

    def close(self):
        self.sess.close()
        self.default_graph.close()

