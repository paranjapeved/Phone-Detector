import keras
import cv2
import numpy as np
import os
from keras.models import Sequential
import sys


# Function to get pixel coordinates of phone in the image
def get_pixel_coordinates(img, label):
    rows, cols, chans = img.shape
    x , y = label[0],label[1]
    return int(x*cols),int(y*rows)

# Resizes image to a different size and scales labels accordingly
def transform_data_target(src_shape,dest_shape,src_label):
    rows_s, cols_s = src_shape[0],src_shape[1]
    rows_d, cols_d = dest_shape[0],dest_shape[1]
    x_s,y_s = src_label[0], src_label[1]
#     print(x_s,y_s)
    x_d = x_s *cols_d / cols_s
#     print(str(x_d) +"="+ str(x_s) +"*"+str(cols_d) +"/"+ str(cols_s))
    y_d = y_s *rows_d / rows_s
    return int(x_d),int(y_d)

# CNN Model defintion
def build_model(input_shape):
    model = Sequential()

    # Each layer downsizes image dimensions by half
    model.add(keras.layers.Conv2D(8, kernel_size=(5, 5),
                                  strides=(2, 2),
                                  activation='relu',
                                  input_shape=input_shape,
                                  padding='valid',
                                  ))

    model.add(keras.layers.MaxPool2D(pool_size=2,
                                     strides=2,
                                     padding='valid'
                                     ))

    model.add(keras.layers.Conv2D(16, kernel_size=(3, 3),
                                  strides=(2, 2),
                                  activation='relu',
                                  padding='valid',
                                  ))

    model.add(keras.layers.MaxPool2D(pool_size=2,
                                     strides=2,
                                     padding='valid'
                                     ))

    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                                  strides=(2, 2),
                                  activation='relu',
                                  padding='valid',
                                  ))
    model.add(keras.layers.MaxPool2D(pool_size=2,
                                     strides=2,
                                     padding='valid'
                                     ))

    model.add(keras.layers.Flatten())

    # Dense layers

    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.7))

    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.7))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(2))

    model.summary()
    model.compile(loss='mean_squared_error',
                  optimizer='Adam',
                  metrics=['mae'])
    return model

# Reads data and labels from path
def read_data(inp_dir_path):
    if inp_dir_path[-1] != '/':
        inp_dir_path += '/'
    images = []
    dic = {}
    labels = []
    with open(inp_dir_path + 'labels.txt') as f:
        for line in f:
            line1 = line.split()
            if len(line1) == 3:
                dic[line1[0]] = [float(line1[1]), float(line1[2])]

    for name in os.listdir(inp_dir_path):
        if name[-3:] == 'jpg':
            img_name = inp_dir_path + name
            img = cv2.imread(img_name, 1)
            images.append(img)
            labels.append(dic[name])

    return images, labels

# Normalize images
def preprocess_images(images):
    return images / 255.0



# Augments dataset by flipping horizontally, vertically and diagonally
def augment_images(img, labels):

    augmented = [img]
    aug_labels = [labels]

    augmented.append(cv2.flip(img, 0))
    aug_labels.append([labels[0], 1.0 - labels[1]])

    augmented.append(cv2.flip(img, 1))
    aug_labels.append([1.0 - labels[0], labels[1]])

    augmented.append(cv2.flip(img, -1))
    aug_labels.append([1.0 - labels[0], 1.0 - labels[1]])
    augmented = np.asarray(augmented)


    return augmented, np.asarray(aug_labels)


if __name__ == '__main__':

    if len(sys.argv) <= 1:
        print("Please enter path of test image")
    else:

        print("\nLoading Data...")
        path = sys.argv[1]
        images, labels = read_data(path)
        print("Data succesfully loaded.")
        images, labels = np.asarray(images), np.asarray(labels)
        images = preprocess_images(images)

        train_images_aug, train_labels_aug = [], []
        for i in range(len(labels)):
            augs, aug_labels = augment_images(images[i], labels[i])
            for j in range(len(aug_labels)):
                train_images_aug.append(augs[j])
                train_labels_aug.append(aug_labels[j])
        train_images_aug, train_labels_aug = np.asarray(train_images_aug), np.asarray(train_labels_aug)
        target_shape = (326, 490, 3)

        print("Data ready for training")

        model1 = build_model(target_shape)

        print("Now training")
        print("train set size")
        print(len(train_images_aug))

        model1.fit(train_images_aug,
                            train_labels_aug,
                            batch_size=16,
                            epochs=100,
                            validation_split=0.1,
                            shuffle=True)


        print("Saving model...")

        model_json = model1.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model1.save_weights("model.h5")
        print("Saved model to disk")

