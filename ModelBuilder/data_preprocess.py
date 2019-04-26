import os
import pickle
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

RAW_DATA_DIR = '../res/TrainingData/raw/'

LABEL_NAMES = ['fist', 'open_hand', 'peace_sign', 'claw', 'letter_y', 'letter_k', 'letter_w']
IMG_WIDTH = 120
IMG_HEIGHT = 150

training_data = []
should_show_first = True

# If set to true will mirror the images to double the dataset and allow for better left and right hand classification
should_mirror_images = False


def create_training_data():
    for category in LABEL_NAMES:

        path = os.path.join(RAW_DATA_DIR, category)
        class_num = LABEL_NAMES.index(category)  # get the classification as an integer

        for img in tqdm(os.listdir(path)):  # iterate over each image per category
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))  # resize to normalize data size
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


create_training_data()

print('Total images processed:', len(training_data))

# Shuffle data to avoid over-fitting one particular class
random.shuffle(training_data)

X = []  # Features array
y = []  # Labels array

for feature, label in training_data:
    X.append(feature)
    y.append(label)

X = np.array(X)
np.reshape(X, (-1, IMG_WIDTH, IMG_HEIGHT, 1))

if should_show_first:
    plt.imshow(X.copy()[0], cmap='gray')
    plt.show()
    print('Shown image label:', LABEL_NAMES[y[0]])

X = np.expand_dims(X, axis=3)
print('Feature array shape:', X.shape)

# Save data as compressed pickle file
pickle_out = open('../res/TrainingData/compressed/X.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('../res/TrainingData/compressed/y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()
