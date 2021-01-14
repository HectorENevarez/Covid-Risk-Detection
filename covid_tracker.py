from Detection.persontracker import PersonTracker
from Detection.detection import detect_persons
from Detection.mask_detection import mask_detect
from scipy.spatial import distance as dist
from tensorflow import keras
import numpy as np
import imutils
import time
import cv2
import os

# argument parser

# Loading in YOLO, coco, and facemask detector
net = cv2.dnn.readNetFromDarknet("models/yolov3.cfg", "models/yolov3.weights") #YOLO

LABELS = open("models/coco.names").read().strip().split("\n")#COCO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#mask_detection = keras.models.load_model("models/mask_detection") #facemask detector
mask_detection = keras.models.load_model("models/mask_detection")

#Unique person tracker
personDetection = PersonTracker() #from Unique ID package

video_path = "street.mp4"
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

while True:
    _, frame = cap.read()

    #frame = imutils.resize(frame, width=1000) #resize to fit screen window
    
    detections = detect_persons(frame, net, ln, pid=LABELS.index("person"))

    violate = set()

    if len(detections) >= 2:
        centroids = np.array([det[2] for det in detections])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        for i in range(0, D.shape[1]):
            for j in range (i + 1, D.shape[1]):
                if D[i,j] < 75:
                    violate.add(i)
                    violate.add(j)
    
    mask_box = []
    box = []
    soc_dist = []
    wearing_mask = [] 
    for (i, (m_bbox, bbox, centroid)) in enumerate(detections):
        (x, y, w, h) = bbox
        box.append(bbox)

        if i in violate:
            soc_dist.append(False)
        else:
            soc_dist.append(True)

        (x_mask, y_mask, w_mask, h_mask) = m_bbox
        mask_box.append([x_mask, y_mask, w_mask, h_mask])

        wearing_mask.append(mask_detect(m_bbox, mask_detection, frame))

    people = personDetection.update(box, soc_dist, mask_box, wearing_mask)
    for (objectID, centroid) in people.items():
        (x, y, w, h) = personDetection.get_bbox(objectID)
        sd = personDetection.get_soc_dist(objectID)
        (x_mask, y_mask, w_mask, h_mask) = personDetection.get_mask_box(objectID)
        wm = personDetection.get_wearing_mask(objectID)
        
        if sd:
            soc_dist_col = (0, 255, 0) #Green
        else:
            soc_dist_col = (0, 0, 255) #Red

        if wm == "mask":
            mask_col = (0, 255, 0) #Green
        elif wm == "nomask":
            mask_col = (0, 0, 255)
        else:
            mask_col = (255, 255, 255)

        cv2.rectangle(frame, (x_mask, int(y_mask - 2 * w_mask / 5)), (x_mask + w_mask, int(y_mask + 3 * w_mask / 5)), mask_col, 2)
        cv2.rectangle(frame, (x, y), (w, h), soc_dist_col, 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)


    cv2.imshow("Frame", frame)
    #out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#out.release()
cap.release()
cv2.destroyAllWindows()