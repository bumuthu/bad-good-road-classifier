import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, \
    GlobalAveragePooling2D, Concatenate
from keras import applications
from sklearn.metrics import classification_report


class InceptionV3Classifier:

    def __init__(self, train_data_gen, test_data_gen, y_test):

        self.ROWS = 139
        self.COLS = 139
        self.nclass = 2

        self.train_data_gen = train_data_gen
        self.test_data_gen = test_data_gen
        self.y_test = y_test

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

        file_path = "./weights.best.hdf5"

        checkpoint = ModelCheckpoint(filepath=file_path, monitor='acc', verbose=1, save_best_only=True, mode='max')

        early = EarlyStopping(monitor="acc", mode="max", patience=15)

        callbacks_list = [checkpoint, early]  # early

        history = self.model.fit_generator(self.train_data_gen,
                                           epochs=10,
                                           shuffle=True,
                                           verbose=True,
                                           callbacks=callbacks_list)

        # self.model.load_weights(file_path)

    def evaluate_model(self):

        predicts = self.model.predict_generator(self.test_data_gen, verbose=True, workers=2)
        predicts = np.argmax(predicts, axis=1)

        report = classification_report(self.y_test, predicts)
        print(report)


