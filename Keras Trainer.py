
# coding: utf-8

# In[1]:


from __future__ import print_function, division
from builtins import range, input

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, BatchNormalization, Dropout, Activation, MaxPooling2D

import matplotlib.pyplot as plt
import numpy as np

from Preprocessing_mod import Preprocessor, Data, DataSet
import pickle

with open('PreprocessorObject.obj', 'rb') as file:
    p = pickle.load(file)
with open('DataObject.data', 'rb') as file:
    data = pickle.load(file)
    
img_shape = p.shape
img_size_flat = img_shape[0]*img_shape[1]*img_shape[2]
IMG_SIZE = (img_shape[0], img_shape[0])
num_classes = len(p.label_ids)
num_channels = p.num_channels
epochs = 10
batch_size = 32


# In[2]:


print(img_shape, img_size_flat, num_classes, num_channels)
print(len(data.train.inputs), len(data.test.inputs), len(data.valid.inputs))


# In[3]:


data_con = np.concatenate((data.train.inputs, data.test.inputs, data.valid.inputs))
label_con = np.concatenate((data.train.labels, data.test.labels, data.valid.labels))
len(data_con)


# In[4]:


def y2indicator(Y):
    K = num_classes
    N = len(Y)
    I = np.zeros((N, K))
    I[np.arange(N), Y] = 1
    return I


# In[5]:


X = data_con.reshape(-1, 32, 32, 3) / 255.0
Y = label_con.astype(np.int32)


# In[6]:


K = len(set(Y))
Y = y2indicator(Y)


# In[7]:


len(X)


# In[8]:


model = Sequential()

model.add(Conv2D(input_shape=(32, 32,3), filters = 32, kernel_size = (3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 32, kernel_size = (3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 32, kernel_size = (3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(units=300))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(units=K))
model.add(Activation('softmax'))


# In[9]:


model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

r = model.fit(X, Y, validation_split=0.33, epochs=epochs, batch_size=batch_size)
print("Returned: ", r)


# In[10]:


print(r.history.keys())


# In[11]:


plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()


# In[12]:


model.predict(X)


# In[13]:


import cv2
import os
from tqdm import tqdm


# In[14]:


BASE_DIR = os.path.dirname(os.path.abspath(''))
print("[INFO] Loading Caffe model...")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
kernel = np.ones((20, 20), np.float32) / 400
print("[INFO] Starting video encoding...")
cap = cv2.VideoCapture('video.mp4')
writer = None
face_names = ['sangjun', 'taekjung', 'taejong', 'yongdae']
fsize = (300, 300)


# In[15]:


while cap.isOpened():
    _, frame = cap.read()
    if frame is None:
        break

    if writer is None:
        writer = cv2.VideoWriter('video_keras.mp4', fourcc, 30, (frame.shape[1], frame.shape[0]))

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, fsize, interpolation=cv2.INTER_AREA), 1.0, fsize, (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if 0 > detections[0, 0, i, 3] or detections[0, 0, i, 3] > 1 or 0 > detections[0, 0, i, 4]                 or detections[0, 0, i, 4] > 1 or 0 > 0 or detections[0, 0, i, 5] > 1                 or 0 > detections[0, 0, i, 6] or detections[0, 0, i, 6] > 1 or confidence < 0.8:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        roi = frame[startY:endY, startX:endX]
        roi = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_AREA)
        roi = np.expand_dims(roi, axis=0)
        text = str(model.predict(roi))
        cv2.putText(frame, text, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    writer.write(frame)

writer.release()
cv2.destroyAllWindows()
cap.release()

