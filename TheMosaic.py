from Preprocessing_mod import Preprocessor, Data, DataSet
from LayerGenerator import *
import Trainer
import cv2
import os
import numpy as np
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("[INFO] Loading Caffe model...")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
print("[INFO] Training model...")
Trainer.train()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
kernel = np.ones((20, 20), np.float32) / 400
print("[INFO] Starting video encoding...")
cap = cv2.VideoCapture('video.mp4')
writer = None
pbar = tqdm(total=int(cv2.VideoCapture.get(cap, cv2.CAP_PROP_FRAME_COUNT)))
face_names = ['other', 'sangjun', 'taekjung', 'taejong', 'yongdae']
fsize = (300, 300)

while cap.isOpened():
    pbar.update()
    _, frame = cap.read()
    if frame is None:
        break

    if writer is None:
        writer = cv2.VideoWriter('video_mosaic.mp4', fourcc, 30, (frame.shape[1], frame.shape[0]))

    # square_frame = cv2.resize(frame, (frame.shape[0], frame.shape[0]), interpolation=cv2.INTER_AREA)

    (h, w) = frame.shape[:2]
    # (ch, cw) = square_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, fsize, interpolation=cv2.INTER_AREA), 1.0, fsize, (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if 0 > detections[0, 0, i, 3] or detections[0, 0, i, 3] > 1 or 0 > detections[0, 0, i, 4] \
                or detections[0, 0, i, 4] > 1 or 0 > 0 or detections[0, 0, i, 5] > 1 \
                or 0 > detections[0, 0, i, 6] or detections[0, 0, i, 6] > 1 or confidence < 0.8:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        # cbox = detections[0, 0, i, 3:7] * np.array([cw, ch, cw, ch])
        (startX, startY, endX, endY) = box.astype("int")
        # (cstartX, cstartY, cendX, cendY) = cbox.astype("int")

        # frame[startY:endY, startX:endX] = cv2.filter2D(frame[startY:endY, startX:endX], -1, kernel)
        roi = frame[startY:endY, startX:endX]
        if not Trainer.predict(roi):
            frame[startY:endY, startX:endX] = cv2.filter2D(frame[startY:endY, startX:endX], -1, kernel)
        prediction = face_names[Trainer.predict(roi)[0]] # [0] [1] [2] [3] [4]
        text = str(prediction)
        cv2.putText(frame, text, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        # cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
    # cv2.imshow('fr', frame)
    writer.write(frame)

writer.release()
cv2.destroyAllWindows()
cap.release()
pbar.close()
