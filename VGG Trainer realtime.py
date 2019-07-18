
# coding: utf-8

# In[1]:


from __future__ import print_function, division
from builtins import range, input

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

train_path = 'images/processed/train'
valid_path = 'images/processed/test'

IMAGE_SIZE = [100, 100]
epochs = 5
batch_size = 32

image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(valid_path+'/*/*.jp*g')

folders = glob(train_path+'/*')

plt.imshow(image.load_img(np.random.choice(image_files)))
plt.show()


# In[2]:


vgg = VGG16(input_shape=IMAGE_SIZE +[3], weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False


# In[3]:


x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)
model = Model(inputs = vgg.input, outputs = prediction)

model.summary()


# In[4]:


model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'rmsprop',
    metrics=['accuracy']
)

gen = ImageDataGenerator(
    rotation_range = 20,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.1,
    zoom_range = 0.2,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function = preprocess_input
)


# In[5]:


test_gen = gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE)
print(test_gen.class_indices)


# In[6]:


labels = [None] * len(test_gen.class_indices)
for k,v in test_gen.class_indices.items():
    labels[v] = k
    
for x,y in test_gen:
    print("min:", x[0].min(), "max:", x[0].max())
    plt.title(labels[np.argmax(y[0])])
    plt.imshow(x[0])
    plt.show()
    break


# In[7]:


train_generator = gen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    shuffle=True,
    batch_size=batch_size
)
valid_generator = gen.flow_from_directory(
    valid_path,
    target_size=IMAGE_SIZE,
    shuffle=True,
    batch_size=batch_size
)


# In[8]:


r = model.fit_generator(
    train_generator,
    validation_data=valid_generator,
    epochs=epochs,
    steps_per_epoch=len(image_files)//batch_size,
    validation_steps=len(valid_image_files)//batch_size
)


# In[9]:


def get_confusion_matrix(data_path, N):
    print("Generation confusion matrix", N)
    predictions = []
    targets = []
    i = 0
    for x, y in gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, shuffle=False, batch_size=batch_size*2):
        i+=1
        if not i%50:
            print(i)
        p = model.predict(x)
        p = np.argmax(p, axis=1)
        y = np.argmax(y, axis=1)
        predictions = np.concatenate((predictions, p))
        targets = np.concatenate((targets, y))
        if len(targets) >= N:
            break
            
    cm = confusion_matrix(targets, predictions)
    return cm

cm = get_confusion_matrix(train_path, len(image_files))
print(cm)
valid_cm = get_confusion_matrix(valid_path, len(valid_image_files))
print(valid_cm)

print(r.history.keys())


# In[10]:


plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()


# In[11]:


import cv2
import os


# In[12]:


BASE_DIR = os.path.dirname(os.path.abspath(''))
print("[INFO] Loading Caffe model...")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
kernel = np.ones((20, 20), np.float32) / 400
print("[INFO] Starting video encoding...")
cap = cv2.VideoCapture(0)
writer = None
# face_names = ['sangjun', 'taekjung', 'taejong', 'yongdae', 'mooh']
face_names = ['sangjun', 'mooh']
fsize = (300, 300)


# In[13]:


while cap.isOpened():
    _, frame = cap.read()

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
        roi = cv2.resize(roi, (100, 100), interpolation=cv2.INTER_AREA)
        roi = np.expand_dims(roi, axis=0)
        text = str(face_names[np.argmax(model.predict(roi), axis=1)[0]])
        cv2.putText(frame, text, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cv2.destroyAllWindows()
cap.release()

