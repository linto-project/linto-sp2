# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import numpy as np
import tensorflow as tf
import tools.utils as utils
import time


class tfDetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image, fusion_threshold = 0.3):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        # start_time = time.time()
        (boxes, scores, classes, num_det) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        # end_time = time.time()
        # print("Elapsed Time:", end_time-start_time)
        im_height, im_width,_ = image.shape

        boxes = boxes[0, 0:int(num_det[0])]
        scores = scores[0, 0:int(num_det[0])]
        classes = classes[0, 0:int(num_det[0])]

        # Box format: startY, startX, endY, endX
        # Rect format: x, y, w, y

        scores  = scores.tolist()
        classes = classes.tolist()

        boxes_list = []
        scores_list = []
        classes_list = []

        for i, box in enumerate(boxes):

            det_box = [int(box[0] * im_height), int(box[1] * im_width),
                       int(box[2] * im_height), int(box[3] * im_width)]
            newbox = True
            rect_det = utils.boxp1p2_to_rect(det_box)

            for j, b in enumerate(boxes_list):
                if classes[i] != classes_list[j]:
                    continue

                rect_list = utils.boxp1p2_to_rect(b)
                iou = utils.IoU(rect_list, rect_det)

                if iou > fusion_threshold:
                    x, y, w, h = utils.union(rect_list, rect_det)
                    boxes_list[j] = [y, x, y+h, x+w]
                    newbox = False
                    break

            if newbox :
                boxes_list.append(det_box)
                scores_list.append(scores[i])
                classes_list.append(classes[i])
        return boxes_list, scores_list, classes_list, len(boxes_list)

        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1] * im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3] * im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num_det[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

