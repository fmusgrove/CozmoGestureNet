import os
import pickle
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split

RAW_DATA_DIR = '../res/TrainingData/raw/'

LABEL_NAMES = ['fist', 'open_hand', 'peace_sign', 'ok', 'claw', 'letter_y', 'letter_k', 'letter_w']
IMG_WIDTH = 100
IMG_HEIGHT = 120

should_show_first = True

# If set to true will mirror the images to double the dataset and allow for better left and right hand classification
should_mirror_images = False


def process_image_for_model(img_to_process: np.ndarray):
    # resize the array to the size standard
    resized_array = cv2.resize(img_to_process, (IMG_WIDTH, IMG_HEIGHT))
    # Normalize the pixel values
    resized_array = resized_array / 255.0
    # Expand the dimensions out to what TensorFlow is expecting
    resized_array = np.expand_dims(resized_array, axis=2)
    resized_array = np.expand_dims(resized_array, axis=0)
    return resized_array


def create_training_data():
    features = []
    labels = []
    for category in LABEL_NAMES:
        path = os.path.join(RAW_DATA_DIR, category)
        class_num = LABEL_NAMES.index(category)  # get the classification as an integer

        for img in tqdm(os.listdir(path)):  # iterate over each image per category
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
                features.append(resized_array)
                labels.append(class_num)

                # Flip image across the vertical axis
                if should_mirror_images:
                    flipped_image = cv2.flip(resized_array.copy(), 1)
                    features.append(flipped_image)
                    labels.append(class_num)

                # Invert the images
                # inverted_image = cv2.bitwise_not(resized_array.copy())
                # features.append(inverted_image)
            except Exception as e:
                pass
    return features, labels


def test_predictions():
    """
    Function to quickly test a single prediction made by the classification model
    :return:
    """
    CLASS_TO_TEST = 'peace_sign'
    IMAGE_NUMBER = 340
    MODEL_NAME = 'optimized_classification_graph.model'

    classification_model: tf.keras.models.Model = tf.keras.models.load_model(f'../Models/hand_class_graph/{MODEL_NAME}')

    path = os.path.join(RAW_DATA_DIR, CLASS_TO_TEST)

    file_image = cv2.imread(os.path.join(path, f'{IMAGE_NUMBER}.png'), cv2.IMREAD_GRAYSCALE)

    processed_image = process_image_for_model(file_image)

    print('Processed image shape:', processed_image.shape)
    image_to_show = np.squeeze(processed_image.copy())
    plt.imshow(image_to_show, cmap='gray')
    plt.show()

    prediction = classification_model.predict([processed_image])
    confidence_score = np.amax(prediction[0])
    label_index = int(np.argmax(prediction[0]))
    print('Classification:', LABEL_NAMES[label_index])
    print('Confidence Score:', confidence_score)


def preprocess_main():
    features, labels = create_training_data()

    # Split the data and shuffle it to avoid over-fitting one particular class
    x_train, x_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.3, shuffle=True)

    x_train = np.asarray(x_train)
    x_valid = np.asarray(x_valid)

    # Normalize features array
    x_train = x_train / 255.0
    x_valid = x_valid / 255.0

    # Convert labels to one-hot array
    y_train = tf.keras.utils.to_categorical(y_train)
    y_valid = tf.keras.utils.to_categorical(y_valid)

    if should_show_first:
        image_to_show = x_train.copy()
        image_to_show = np.squeeze(image_to_show)[3]
        plt.imshow(image_to_show, cmap='gray')
        plt.show()
        print('Shown image label:', LABEL_NAMES[int(np.argmax(y_train[3]))])

    x_train = np.expand_dims(x_train, axis=3)
    x_valid = np.expand_dims(x_valid, axis=3)
    print('Feature array shape:', x_train.shape)

    # Save data as compressed pickle file
    pickle_out = open('../res/TrainingData/compressed/x_train.pickle', 'wb')
    pickle.dump(x_train, pickle_out)
    pickle_out.close()

    pickle_out = open('../res/TrainingData/compressed/y_train.pickle', 'wb')
    pickle.dump(y_train, pickle_out)
    pickle_out.close()

    pickle_out = open('../res/TrainingData/compressed/x_valid.pickle', 'wb')
    pickle.dump(x_valid, pickle_out)
    pickle_out.close()

    pickle_out = open('../res/TrainingData/compressed/y_valid.pickle', 'wb')
    pickle.dump(y_valid, pickle_out)
    pickle_out.close()

    print('Total images processed:', len(features))
    print('>>>>>>> Pre-processing complete')


if __name__ == '__main__':
    preprocess_main()
