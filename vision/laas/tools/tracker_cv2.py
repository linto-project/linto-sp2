from __future__ import print_function
import sys
import cv2
import tools.utils as utils
import time

class singleTracker:

    trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

    def __init__(self, frame, box_body, box_head = None):
        # Specify the tracker type
        self.trackerType = "CSRT"

        self.tracker_body = self.createTrackerByName()
        self.tracker_head = self.createTrackerByName()
        self.md = 0
        self.box_body  = box_body
        self.box_head  = box_head
        self.body_status = True
        self.color = utils.create_colors(1)[0]

        ok = self.tracker_body.init(frame, box_body)
        if box_head is not None:
            ok = self.tracker_head.init(frame, box_head)

    def close(self):
        self.tracker_body = []
        self.tracker_head = []

    def createTrackerByName(self):
        trackerType = self.trackerType
        # Create a tracker based on tracker name
        if trackerType == self.trackerTypes[0]:
            tracker = cv2.TrackerBoosting_create()
        elif trackerType == self.trackerTypes[1]:
            tracker = cv2.TrackerMIL_create()
        elif trackerType == self.trackerTypes[2]:
            tracker = cv2.TrackerKCF_create()
        elif trackerType == self.trackerTypes[3]:
            tracker = cv2.TrackerTLD_create()
        elif trackerType == self.trackerTypes[4]:
            tracker = cv2.TrackerMedianFlow_create()
        elif trackerType == self.trackerTypes[5]:
            tracker = cv2.TrackerGOTURN_create()
        elif trackerType == self.trackerTypes[6]:
            tracker = cv2.TrackerMOSSE_create()
        elif trackerType == self.trackerTypes[7]:
            tracker = cv2.TrackerCSRT_create()
        else:
            tracker = None
            print('Incorrect tracker name')
            print('Available trackers are:')
            for t in self.trackerTypes:
                print(t)
        return tracker

    def update(self, frame, person_detection = None, head_detection = None, threshold = 0.6):
        # If not detection, update tracker
        if person_detection is None:
            success, box = self.tracker_body.update(frame)
            self.box_body = box
            self.body_status = success
            if not success:
                self.md += 1

            if self.box_head is not None:
                success_head, box_head = self.tracker_head.update(frame)
                self.box_head = box_head

        # Re-init tracker if detection is available
        else:
            box = self.box_body
            detected = False
            for i, det in enumerate(person_detection):
                det_box = (det[1], det[0], det[3] - det[1], det[2] - det[0])
                iou = utils.IoU(det_box, box)

                if iou < threshold and head_detection is not None and head_detection[i] != [] and self.box_head is not None:
                    h_det = head_detection[i]
                    det_hd = (h_det[1], h_det[0], h_det[3] - h_det[1], h_det[2] - h_det[0])
                    iou_head = utils.IoU(det_hd, self.box_head)
                else:
                    iou_head = 0

                # Reinitialize
                if iou > threshold or iou_head > threshold:
                    if head_detection is not None and head_detection[i] != []:
                        h_det = head_detection[i]
                        det_hd = (h_det[1], h_det[0], h_det[3] - h_det[1], h_det[2] - h_det[0])
                    elif self.box_head is not None:
                        success_head, box_head = self.tracker_head.update(frame)
                        self.box_head = box_head
                        det_hd = self.box_head
                    else:
                        det_hd = None

                    self.tracker_body = self.createTrackerByName()
                    self.tracker_head = self.createTrackerByName()

                    ok = self.tracker_body.init(frame, det_box)
                    if det_hd is not None:
                        try:
                            ok = self.tracker_head.init(frame, det_hd)
                            self.box_head = det_hd
                        except:
                            self.box_head = None

                    self.md = 0
                    self.box_body = det_box
                    self.body_status = True

                    person_detection.remove(det)
                    detected = True
                    if head_detection is not None:
                        del head_detection[i]
                    break

            if not detected:
                self.md += 5

        return [self.body_status, self.box_body]





class cvTrackerAPI:

    def __init__(self):
        self.colors = utils.create_colors(100)
        self.trackers = []
        self.box_status = []
        self.tracker_remove_thr = 100


    def clean(self):
        self.trackers = []
        self.box_status = []

    def close(self):
        self.trackers = []


    def addTrackers(self, boxes, person_detection, head_detection, frame, threshold = 0.2):

        for i, p_det in enumerate(person_detection):
        # for p_det, h_det in zip(person_detection,head_detection):
            det_box = (p_det[1], p_det[0], p_det[3] - p_det[1], p_det[2] - p_det[0])
            # det_hd  = (h_det[1], h_det[0], h_det[3] - h_det[1], h_det[2] - h_det[0])
            isnew = True
            for box in boxes:
                if utils.IoU(det_box, box) > threshold:
                    isnew = False
                    break

            if isnew:
                if head_detection is not None and head_detection[i] != []:
                    h_det = head_detection[i]
                    det_hd = (h_det[1], h_det[0], h_det[3] - h_det[1], h_det[2] - h_det[0])
                else:
                    det_hd = None
                self.trackers.append(singleTracker(frame, det_box, det_hd))
                boxes.append(det_box)

    def draw_trackers(self, frame):
        # draw tracked objects
        # for tracker, color in zip(self.trackers, self.colors):
        for tracker in self.trackers:
            newbox = tracker.box_body
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, tracker.color, 2, 1)

            newbox = tracker.box_head
            if newbox is None:
                continue
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, tracker.color, 2, 1)

    #
    def get_boundig_boxes(self):
        """
        Get a list of boxes in the format y,x h,w
        :return: Return a list of bouding boxes in format: y,x, h, w , color
        """
        boxes = []
        for i, newbox in enumerate(self.box_list):
            boxes.append([int(newbox[1]), int(newbox[0]), int(newbox[3]), int(newbox[2]), self.trackers[i].color ])
        return  boxes

    # Get a list of boxes in the format y1,x1 y2,x2
    def get_box_p1p2(self):
        boxes = []
        for newbox in self.box_list:
            boxes.append([int(newbox[1]), int(newbox[0]), int(newbox[1] + newbox[3]), int(newbox[0] + newbox[2]) ])
        return  boxes

    # Get a list of boxes in the format y1,x1 y2,x2
    def get_box_head_p1p2(self):
        boxes = []
        for tracker in self.trackers:
            newbox = tracker.box_head
            if newbox is not None:
                boxes.append([int(newbox[1]), int(newbox[0]), int(newbox[1] + newbox[3]), int(newbox[0] + newbox[2]) ])
        return  boxes

    def processFrame(self, frame, person_detection = None, head_detection = None):

        self.box_list =  []

        # get updated location of objects in subsequent frames
        for i, tracker in enumerate(self.trackers):
            # Bool , [x,y,w,h]
            success, box = tracker.update(frame, person_detection, head_detection, 0.5)
            self.box_list.append(box)

        if person_detection is not None:
            # Box format: x,y,w,h
            self.addTrackers(self.box_list, person_detection, head_detection, frame, threshold=0.1)

        self.removeOldTrackers()

        return self.box_list

    def removeOldTrackers(self):

        for i, tracker in enumerate(self.trackers):
            if tracker.md >= self.tracker_remove_thr:
                self.trackers.remove(tracker)
                del self.box_list[i]
