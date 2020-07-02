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

class DataPreprocessing:

    def __init__(self):
        self.data_dir = '/home/bumuthudilshanhhk/Downloads/dataset/'
        self.test_ratio = 0.3
        self.ROWS = 139
        self.COLS = 139

    def prepare_image_path_df(self):
        paths = []
        y = []

        for bad_pth in os.listdir(self.data_dir + 'BadRoad'):
            paths.append(self.data_dir + 'BadRoad/' + bad_pth)
            y.append(0)
        for good_pth in os.listdir(self.data_dir + 'GoodRoad'):
            paths.append(self.data_dir + 'GoodRoad/' + good_pth)
            y.append(1)

        paths_train, paths_test, y_train, y_test = train_test_split(paths, y, test_size=self.test_ratio,
                                                                    random_state=42)

        train_path_dict = [{"filename": paths_train[i], "class": str(y_train[i])} for i in range(len(paths_train))]
        test_path_dict = [{"filename": paths_test[i], "class": str(y_test[i])} for i in range(len(paths_test))]

        self.train_df = pd.DataFrame.from_dict(train_path_dict)
        self.test_df = pd.DataFrame.from_dict(test_path_dict)
        self.y_test = y_test


    def prepare_data_generator(self):
        data_gen = ImageDataGenerator(vertical_flip=True,
                                      horizontal_flip=True,
                                      height_shift_range=0.1,
                                      width_shift_range=0.1,
                                      preprocessing_function=preprocess_input)

        self.train_data_gen = data_gen.flow_from_dataframe(
            dataframe=self.train_df,
            directory=None,
            class_mode="categorical",
            x_col="filename",
            y_col="class",
            weight_col=None,
            classes=None,
            target_size=(self.ROWS, self.COLS),
            batch_size=16)

        self.test_data_gen = data_gen.flow_from_dataframe(
            dataframe=self.test_df,
            directory=None,
            x_col="filename",
            y_col="class",
            weight_col=None,
            classes=None,
            target_size=(self.ROWS, self.COLS),
            batch_size=64)

        input_shape = (self.ROWS, self.COLS, 3)
        self.nclass = len(self.train_data_gen.class_indices)