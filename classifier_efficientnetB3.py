import numpy as np
import cv2
from tqdm import tqdm
import os
import sys
import keras
from sklearn.preprocessing import LabelEncoder


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def get_data(test_path, image_size = 128):
    # resize images and put together Testing folder
    x, y = [], []
    lb = LabelEncoder()

    classes_ = get_classes(test_path)
    for i in classes_:
        path = os.path.join(test_path, i)
        for j in tqdm(os.listdir(path)):
            img = cv2.imread(os.path.join(path, j))
            img = cv2.resize(img, (image_size, image_size))
            x.append(img)
            y.append(i)

    x, y = np.array(x), np.array(y)
    labels_train = lb.fit(y)
    y = lb.transform(y)

    # print(f"X_test: {x.shape}\ny_test: {y.shape}")
    return x, y


def get_classes(path):
    classes_ = []
    for sub_folder in os.listdir(path):
        classes_.append(sub_folder)
    return classes_


if __name__ == '__main__':
    load_path = sys.argv[1]

    # get classifier model
    model = keras.models.load_model('./models/C5_EfficientNetB3.h5')

    # create labels
    classes = get_classes(load_path)

    # get data
    X_test, y_test = get_data(load_path)

    # test data
    loss, accuracy = model.evaluate(X_test, y_test)

    print(f"Accuracy score: {accuracy}\nLoss: {loss}")
