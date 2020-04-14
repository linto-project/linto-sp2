# Global class to detect: person, face, face landmarks (shapes) and face recognition

from tools.detector_tensorflow import tfDetectorAPI
from tools.detector_cv2 import cvDetectorAPI
from tools.dlib_landmark_detector import dlibLandmarkDetectorAPI
from tools.recognize_face import recognitionFaceAPI
from tools.detector_pose import poseDetectorAPI
import tools.params as params
import cv2
import numpy as np
import time


class linToDetector:
    def __init__(self, outputPath = None, write = False):
        # load our serialized model from disk
        print("[INFO] loading models...")
        self.objDet = tfDetectorAPI(path_to_ckpt=params.person_model_path)
        self.faceDet = cvDetectorAPI(face_prototxt=params.face_prototxt, face_model_path=params.face_model_path)
        self.shapeDet = dlibLandmarkDetectorAPI(landmark_model=params.landmark_model, scale=1.0)
        self.recognizer = recognitionFaceAPI(target_encodings=params.target_encodings)
        self.poseDetector = poseDetectorAPI(protoFile= params.pose_protoFile, weightsFile = params.pose_weightsFile, nPoints = params.pose_nPoints)

        self.threshold = 0.5

        self.rsh = 1000
        self.rsw = 1000
        self.rgbMean = (104.0, 177.0, 123.0)

        self.clean()


    def clean(self):
        self.people_det = []
        self.shapes = []
        self.faces_det = []
        self.recogFaces = []
        # self.recognizer.boxes = []


    def draw_face_det(self, img, color=(255, 0, 0), threshold=0.):
        if self.faces_det is None or self.faces_det==[]:
            return None
        for i in range(len(self.faces_det[0])):
            # Class 1 represents human
            # if people_det[2][i] == 1 and people_det[1][i] > threshold:
            if self.faces_det[1][i] > threshold:
                box = self.faces_det[0][i]
                # p1(x,y) p2(x,y)
                cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), color, 2)



    def draw_face_landmarkds(self, img, color=(0, 255, 0), threshold=0.):
        if hasattr(self, 'shapes') is False:
            return None
        # loop over the face detections
        for (i, shape) in enumerate(self.shapes):
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(img, (x, y), 1, (0, 0, 255), -1)



    def draw_people_det(self, img, color=(255, 0, 0), threshold=0.):
        if self.people_det is None or self.people_det == []:
            return None
        for i in range(len(self.people_det[0])):
            # Class 1 represents human
            # if people_det[2][i] == 1 and people_det[1][i] > threshold:
            if self.people_det[1][i] > threshold:
                box = self.people_det[0][i]
                cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), color, 2)



    def draw_face_recognition(self, img, color=(255, 0, 0), threshold=0.):
        if self.recogFaces is None or self.recogFaces == []:
            return None
        self.recognizer.draw_recognitions(img)


    def draw_pose_estimation(self, img, color=(255, 0, 0), threshold=0.):
        self.poseDetector.draw(img)


    def processFrame(self, img, detFace=True, detShapes=True, regFace=True, poseEstim=False):
        # Detect all objects in the scene
        boxes, scores, classes, num = self.objDet.processFrame(img)
        self.all_det = [boxes, scores, classes, num]
        # filter persons
        self.people_det = [[boxes[x]  for x in range(num) if classes[x] == 1 and scores[x] > 0.5],
                           [scores[x] for x in range(num) if classes[x] == 1 and scores[x] > 0.5]]

        # Process the whole image
        # boxesf, scoresf, detections = cvapi.processFrame(img)
        # Detect faces on ROI
        if detFace:
            boxesf, scoresf = self.faceDet.processFrameSeg(img, self.people_det[0])
            # if self.people_det[0] == []:
                # boxesf, scoresf = self.faceDet.processFrame(img)
            self.faces_det = [boxesf, scoresf]
        else:
            self.faces_det = [None, None]

        if detShapes and detFace:
            # self.shapes = self.shapeDet.processFrame(img)
            # self.shapes = self.shapeDet.processFrameSeg(img, self.people_det[0])
            self.shapes = self.shapeDet.findFace(img, boxesf)

        if regFace:
            # R
            if detFace:
                # self.recogFaces = self.recognizer.processFrame(img) # Boxes(img, self.faces_det[0])
                self.recogFaces = self.recognizer.processFrameBoxes(img, self.faces_det[0])
            else:
                self.recogFaces = self.recognizer.processFrameBoxes(img, self.people_det[0])

        if poseEstim:
            self.poseDetector.processBoxes(img, self.people_det[0])


    def processBoxes(self, img, person_boxes, head_boxes = None, detFace=True, detShapes=True, regFace=True, poseEstim=False):

        if self.people_det == []:
            self.people_det.append(person_boxes)
        else:
            self.people_det[0] = person_boxes
        self.faces_det     = head_boxes

        # Detect faces on ROI
        if detFace:
            boxesf, scoresf = self.faceDet.processFrameSeg(img, person_boxes)
            self.faces_det = [boxesf, scoresf]

        # Detect shapes on ROI
        if detShapes and head_boxes is not None:
            # dlib_dect = dlibapi.processFrameSeg(img, people_det[0])
            self.shapes = self.shapeDet.findFace(img, self.faces_det)


        if regFace:
            # R
            if head_boxes is not None:
                # self.recogFaces = self.recognizer.processFrame(img) # Boxes(img, self.faces_det[0])
                self.recogFaces = self.recognizer.processFrameBoxes(img, head_boxes)
            else:
                self.recogFaces = self.recognizer.processFrameBoxes(img, person_boxes)

        if poseEstim:
            self.poseDetector.processBoxes(img, person_boxes)



    def close(self):
        self.sess.close()
        self.default_graph.close()

