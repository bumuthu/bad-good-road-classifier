import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from keras.models import Sequential
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, \
    GlobalAveragePooling2D, Concatenate
from keras import applications


class InceptionV3Classifier:

    def __init__(self):
        self.data_dir = '..//data//'
        self.test_ratio = 0.3
        self.ROWS = 139
        self.COLS = 139

        self.prepare_image_path_df()

    def prepare_image_path_df(self):

        paths = []
        y = []

        for bad_pth in os.listdir(self.data_dir + 'BadRoad'):
            paths.append(bad_pth)
            y.append(0)
        for good_pth in os.listdir(self.data_dir + 'GoodRoad'):
            paths.append(good_pth)
            y.append(1)

        paths_train, paths_test, y_train, y_test = train_test_split(paths, y, test_size=self.test_ratio,
                                                                    random_state=42)

        train_path_dict = [{"id": paths_train[i], "label": y_train[i]} for i in range(len(paths_train))]
        test_path_dict = [{"id": paths_test[i], "label": y_test[i]} for i in range(len(paths_test))]

        self.train_df = pd.DataFrame.from_dict(train_path_dict)
        self.test_df = pd.DataFrame.from_dict(test_path_dict)

    def prepare_data_generator(self):

        data_gen = ImageDataGenerator(vertical_flip=True,
                                      horizontal_flip=True,
                                      height_shift_range=0.1,
                                      width_shift_range=0.1,
                                      preprocessing_function=preprocess_input)

        self.train_data_gen = data_gen.flow_from_dataframe(
            dataframe=self.train_df,
            target_size=(self.ROWS, self.COLS),
            batch_size=16)

        self.test_data_gen = data_gen.flow_from_dataframe(
            dataframe=self.test_df,
            target_size=(self.ROWS, self.COLS),
            batch_size=16)

        input_shape = (self.ROWS, self.COLS, 3)
        self.nclass = len(self.train_data_gen.class_indices)

    def make_inceptionv3_model(self):

        base_model = applications.InceptionV3(weights='imagenet',
                                              include_top=False,
                                              input_shape=(self.ROWS, self.COLS, 3))
        base_model.trainable = False

        add_model = Sequential()
        add_model.add(base_model)
        add_model.add(GlobalAveragePooling2D())
        add_model.add(Dropout(0.5))
        add_model.add(Dense(self.nclass,
                            activation='softmax'))

        self.model = add_model
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizers.SGD(lr=1e-4,
                                                    momentum=0.9),
                           metrics=['accuracy'])
        self.model.summary()

    def train_model(self):

        file_path = "weights.best.hdf5"

        checkpoint = ModelCheckpoint(file_path, monitor='acc', verbose=1, save_best_only=True, mode='max')

        early = EarlyStopping(monitor="acc", mode="max", patience=15)

        callbacks_list = [checkpoint, early]  # early

        history = self.model.fit_generator(self.train_data_gen,
                                           epochs=2,
                                           shuffle=True,
                                           verbose=True,
                                           callbacks=callbacks_list)

        self.model.load_weights(file_path)

    def evaluate_model(self):

        predicts = self.model.predict_generator(self.test_data_gen, verbose=True, workers=2)
        predicts = np.argmax(predicts, axis=1)

        print(predicts)

