# import the necessary packages
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

class recognitionFaceAPI:
    def __init__(self, target_encodings, detection_method="cnn"):
        self.target_encodings = target_encodings
        self.detection_method = detection_method

        # load the known faces and embeddings
        # print("[INFO] loading encodings...")
        self.data = pickle.loads(open(target_encodings, "rb").read())

    def processFrame(self, image):
        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input frame, then compute
        # the facial embeddings for each face
        start_time = time.time()
        # boxes: (top, right, bottom, left)
        boxes = face_recognition.face_locations(image,
                                                model=self.detection_method)
        encodings = face_recognition.face_encodings(image, boxes)
        names = []
        time_detection = time.time() - start_time

        # loop over the facial embeddings
        start_time = time.time()
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(self.data["encodings"],
                                                     encoding)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = self.data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)
        time_recognition = time.time() - start_time

        # print("Detec: %f \t Recog: %f" % (time_detection, time_recognition))

        self.boxes = boxes
        self.names = names
        return (boxes, names)


    # Input: rgb image, boxes format:
    # startY, startX, endY, endX
    def processFrameBoxes(self, image, boxes):
        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input frame, then compute
        # the facial embeddings for each face
        start_time = time.time()
        # Re-organize boxes to format:
        # (top, right, bottom, left)
        box_list = [[item[0], item[3],item[2], item[1]] for item in boxes if item != [] ]
        encodings = face_recognition.face_encodings(image, box_list)
        names = []
        time_detection = time.time() - start_time

        # loop over the facial embeddings
        start_time = time.time()
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches, matchesDist = face_recognition.compare_faces_distance(self.data["encodings"], encoding, 0.6)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = self.data["names"][i]
                    counts[name] = counts.get(name, 0) + matchesDist[i]

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)
                # name = self.data["names"][list(matchesDist).index(max(matchesDist))]

            # update the list of names
            names.append(name)
        time_recognition = time.time() - start_time

        # print("Detec: %f \t Recog: %f" % (time_detection, time_recognition))

        self.boxes = [item for item in boxes if item != [] ]
        self.names = names
        return (boxes, names)


    def draw_recognitions(self, image, drawBox = True):
        r = 1
        # loop over the recognized faces
        if drawBox and self.names is None:
            for (top, right, bottom, left) in self.boxes:
                # rescale the face coordinates
                top = int(top * r)
                right = int(right * r)
                bottom = int(bottom * r)
                left = int(left * r)

                # draw the predicted face name on the image
                cv2.rectangle(image, (left, top), (right, bottom),
                              (0, 0, 255), 2)
        else:
            for ((top, right, bottom, left), name) in zip(self.boxes, self.names):
                # for (top, right, bottom, left) in boxes:
                # rescale the face coordinates
                top = int(top * r)
                right = int(right * r)
                bottom = int(bottom * r)
                left = int(left * r)

                # draw the predicted face name on the image
                if drawBox:
                    cv2.rectangle(image, (left, top), (right, bottom),
                                  (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)

    def close(self):
        self.sess.close()
        self.default_graph.close()

