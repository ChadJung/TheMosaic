# receives root directory when creating Preprocessor instance
# folders: image01, image02, image03 ... (or user set names)
# images: 1.ext, 2.ext, 3.ext ... (ext: file extension)

import os
import numpy as np
import cv2
import shutil
import random
import pickle
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
random.seed(20)

class Preprocessor:
    def __init__(self, img_dir,             # name of the directory containing the image folders
                 extension='jpg',           # extension of the image files, (default: jpg)
                 data_size=(32, 32),     # size of single data(image)
                 split_data=True,           # split into train, test, valid?
                 train_rate=0.8,            # if split_data then % of train data size
                 batch_size=None            # size of batch when iterating data
                 ):
        assert train_rate == 0.7 or train_rate == 0.8
        self.img_dir = os.path.join(BASE_DIR, img_dir)
        self.ext = extension
        self.data_size = data_size
        self.num_channels = 3
        self.shape = (data_size[0], data_size[1], self.num_channels)
        self.split_data = split_data
        self.train_rate = train_rate
        self.label_ids = []
        self.batch_size = batch_size

    def preprocess(self):
        x_train = []
        y_labels = []
        n = 0
        net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
        # data_size = 0
        # for root, dirs, files in os.walk(self.img_dir):
        #     curr_dir = root.split('\\')[-1]
        #     if curr_dir not in ['images', 'other_faces']:
        #         data_size += len(files)
        data_size = 500
        for root, dirs, files in os.walk(self.img_dir):
            curr_dir = root.split('\\')[-1]
            data_count = 0
            if curr_dir == 'images':
                continue
            if not curr_dir == 'other_faces':
                n += 1
            print('\n')
            print('processing {}...'.format(curr_dir))
            random.shuffle(files)
            # print(files)
            pbar = tqdm(total=data_size)
            for file in files:
                pbar.update()
                if file.endswith(self.ext):  # walk through all the images.ext in img_dir
                    # print(file)
                    label = 0 if curr_dir == 'other_faces' else n
                    if label not in self.label_ids:
                        self.label_ids.append(label)
                    path = os.path.join(root, file)
                    raw_image = cv2.imread(path)
                    # image = cv2.resize(raw_image, (300, 300))
                    (h, w) = raw_image.shape[:2]
                    blob = cv2.dnn.blobFromImage(cv2.resize(raw_image, (300, 300), interpolation=cv2.INTER_AREA), 1.0, (300, 300)
                                                 # if not curr_dir == 'other_faces' else (270, 300)
                                                 , (104.0, 177.0, 123.0))

                    net.setInput(blob)
                    detections = net.forward()

                    for i in range(0, detections.shape[2]):
                        confidence = detections[0, 0, i, 2]

                        # removes the wrong & out of bound detections
                        if 0 > detections[0, 0, i, 3] or detections[0, 0, i, 3] > 1 or 0 > detections[0, 0, i, 4] \
                                or detections[0, 0, i, 4] > 1 or 0 > 0 or detections[0, 0, i, 5] > 1 \
                                or 0 > detections[0, 0, i, 6] or detections[0, 0, i, 6] > 1 or confidence < 0.8:
                            continue
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        # write the facial images
                        face = raw_image[startY:endY, startX:endX]
                        #face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        face_resize = cv2.resize(face, self.data_size, interpolation=cv2.INTER_AREA)

                        data_flattened = face_resize.flatten()
                        x_train.append(data_flattened)
                        y_labels.append(label)
                        data_count += 1
                        if data_count > data_size:
                            pbar.close()
                            break
                    if data_count > data_size:
                        break
            if data_count > data_size:
                continue

        x_train = np.array(x_train)
        y_labels = np.array(y_labels)
        dsize_flat = self.shape[0] * self.shape[1] * self.shape[2]
        print(y_labels)
        data = Data(x_train, y_labels, self.label_ids, dsize_flat, self.split_data, self.train_rate, self.batch_size)

        return data


class DataSet:
    def __init__(self, inputs, labels, class_num, batch_size=None):
        assert inputs.shape[0] == len(labels)
        self.current_batch = 0
        self.inputs = inputs
        self.labels = labels
        self.class_num = class_num
        if batch_size is None:
            batch_size = inputs.shape[0]
        else:
            batch_size = batch_size

        self.batch_count = inputs.shape[0] // batch_size if batch_size < inputs.shape[0] else 1
        # slice the batches and put the slices in lists
        self.input_batches = []
        self.target_batches = []

        for curr_batch in range(self.batch_count):
            batch_slice = slice(curr_batch * batch_size,
                                (curr_batch + 1) * batch_size)
            self.input_batches.append(inputs[batch_slice])
            self.target_batches.append(labels[batch_slice])

        # shuffle the batches
        dataset = list(zip(self.input_batches, self.target_batches))
        random.shuffle(dataset)
        self.input_batches, self.target_batches = zip(*dataset)
        self.input_batches = np.array(self.input_batches)
        self.target_batches = np.array(self.target_batches)
        # print(self.target_batches)

    def __next__(self):
        if self.current_batch >= self.batch_count:
            self.current_batch = 0
            raise StopIteration()
        target_batch = self.target_batches[self.current_batch]
        input_batch = self.input_batches[self.current_batch]
        # print(type(target_batch))
        targets_one_hot = np.zeros((target_batch.shape[0], self.class_num))
        # print(range(target_batch.shape[0]), target_batch)
        targets_one_hot[range(target_batch.shape[0]), target_batch] = 1
        self.current_batch += 1
        return input_batch, targets_one_hot

    def __iter__(self):
        return self


class Data:
    def __init__(self, inputs, outputs, label_ids, dsize_flat, split_data=True, train_rate=0.8, batch_size=None):

        def shuffle(x, y):
            dataset = list(zip(x, y))
            random.shuffle(dataset)
            x_mod, y_mod = zip(*dataset)
            return np.array(x_mod), np.array(y_mod)

        if split_data:
            current_index = 0
            test_rate = 0.1
            train_inputs = np.empty((0, dsize_flat), np.int32)
            train_outputs = np.empty(0, np.int32)
            test_inputs = np.empty((0, dsize_flat), np.int32)
            test_outputs = np.empty(0, np.int32)
            valid_inputs = np.empty((0, dsize_flat), np.int32)
            valid_outputs = np.empty(0, np.int32)
            for n in range(len(label_ids)):
                count = np.count_nonzero(outputs == n)
                train_index = current_index + int(count * train_rate)
                test_index = train_index + int(count * test_rate)
                label_index = current_index + count
                # print(count, train_index, test_index, label_index)
                # print(train_inputs.shape, inputs[current_index:train_index].shape)
                train_inputs = np.concatenate((train_inputs, inputs[current_index:train_index]))
                train_outputs = np.concatenate((train_outputs, outputs[current_index:train_index]))
                test_inputs = np.concatenate((test_inputs, inputs[train_index:test_index]))
                valid_inputs = np.concatenate((valid_inputs, inputs[test_index:label_index]))
                test_outputs = np.concatenate((test_outputs, outputs[train_index:test_index]))
                valid_outputs = np.concatenate((valid_outputs, outputs[test_index:label_index]))
                current_index = label_index

            # print(train_inputs, train_outputs)
            train_inputs, train_outputs = shuffle(train_inputs, train_outputs)
            test_inputs, test_outputs = shuffle(test_inputs, test_outputs)
            valid_inputs, valid_outputs = shuffle(valid_inputs, valid_outputs)

            print(train_outputs)
            self.train = DataSet(train_inputs, train_outputs, len(label_ids), batch_size)
            self.test = DataSet(test_inputs, test_outputs, len(label_ids), batch_size)
            self.valid = DataSet(valid_inputs, valid_outputs, len(label_ids), batch_size)
        else:
            inputs, outputs = shuffle(inputs, outputs)
            self.train = Dataset(inputs, outputs, len(label_ids), batch_size)


if __name__ == '__main__':
    p = Preprocessor('images', extension='jpg', batch_size=50)  # image root directory can be changed
    d = p.preprocess()

    print('Train data length: {}'.format(d.train.inputs.shape[0]))
    print('Train NonZero: {}'.format(np.count_nonzero(d.train.labels)))
    print('Test data length: {}'.format(d.test.inputs.shape[0]))
    print('Test NonZero: {}'.format(np.count_nonzero(d.test.labels)))
    print('Valid data length: {}'.format(d.valid.inputs.shape[0]))
    print('Valid NonZero: {}'.format(np.count_nonzero(d.valid.labels)))

    with open('PreprocessorObject.obj', 'wb') as f:
        pickle.dump(p, f)
    with open('DataObject.data', 'wb') as f:
        pickle.dump(d, f)
