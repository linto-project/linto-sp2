# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import numpy as np
import cv2
import tools.utils
import time


class cvDetectorAPI:
    def __init__(self, face_prototxt, face_model_path):
        self.face_prototxt = face_prototxt
        self.face_model_path = face_model_path
        self.net = cv2.dnn.readNetFromCaffe(face_prototxt, face_model_path)
        self.rsh = 1000
        self.rsw = 1000
        self.rgbMean = (104.0, 177.0, 123.0)


    # Input: rgb image,
    # Output: boxes format:
    # startY, startX, endY, endX
    def processFrame(self, image):
        # grab the frame dimensions and convert it to a blob
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (self.rsh, self.rsw)), 1.0,
                                     (self.rsh, self.rsw), self.rgbMean)


        # pass the blob through the network and obtain the detections and
        # predictions
        self.net.setInput(blob)
        detections = self.net.forward()

        (h, w) = image.shape[:2]
        boxes_list = []
        scores = []
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < 0.01:
                continue
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            lx = endX - startX
            ly = endY - startY
            # filter out detections with unrealistic dimensions
            if ((lx * ly > w * h * 0.01) and confidence < 0.3) or (
                    ((ly / lx < 1) or (ly / lx > 1.5)) and confidence < 0.95):
                continue
            boxes_list.append([startY, startX, endY, endX])
            scores.append(confidence)

        # return boxes_list, scores, detections
        return boxes_list, scores

    # Input: rgb image, boxes format:
    # startY, startX, endY, endX
    def processFrameSeg(self, image, boxes):

        boxes_list = []
        scores = []
        for j, b in enumerate(boxes):
            # b = boxes[i]
            if b[0] == 0 and b[1] == 0 and b[2] == 0:
                break
            person = image[b[0]:b[2], b[1]:b[3]]
            # grab the frame dimensions and convert it to a blob
            # blob = cv2.dnn.blobFromImage(person, 1.0,
            #                          (person.shape[0], person.shape[1]), self.rgbMean)
            cb, cg, cr, c_ = cv2.mean(person)

            blob = cv2.dnn.blobFromImage(cv2.resize(person, (299,299)), 1.0,
                                         (299, 299), (cb, cg, cr))

            # pass the blob through the network and obtain the detections and
            # predictions
            self.net.setInput(blob)
            detections = self.net.forward()

            img_draw = image.copy()

            (h, w) = person.shape[:2]
            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence < 0.3 or any(i > 1 for i in detections[0, 0, i, 3:7])  or any(i <= 0 for i in detections[0, 0, i, 3:7]) :
                    break
                # compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                boxes_list.append([startY + b[0], startX + b[1], endY + b[0], endX + b[1]])
                scores.append(confidence)

                # cv2.rectangle(img_draw, (startX + b[1], startY + b[0]), (endX + b[1], endY + b[0]),
                #               (0, 0, 255), 2)
                # cv2.rectangle(img_draw, (b[1], b[0]), (b[3], b[2]), (255,0,0), 2)
                # cv2.imshow("detection", img_draw)
                # cv2.waitKey(0)
                break

            if len(boxes_list) <= j:
                boxes_list.append([])
                scores.append(-1)


        return boxes_list, scores

    def close(self):
        self.sess.close()
        self.default_graph.close()

