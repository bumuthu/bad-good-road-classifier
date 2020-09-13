import numpy as np
import json
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, \
    GlobalAveragePooling2D, Concatenate
from keras import applications
from sklearn.metrics import classification_report, accuracy_score


class InceptionResNetV2Classifier:

    def __init__(self, train_df, test_df, y_test, epochs, batch_size):
        self.ROWS = 299
        self.COLS = 299
        self.batch_size = batch_size
        self.epochs = epochs
        self.file_path = "./weights/weights-inceptionresnetv2.h5"

        self.train_df = train_df
        self.test_df = test_df
        self.y_test = y_test

    def prepare_data_generator(self):
        train_data_gen = ImageDataGenerator(vertical_flip=False,
                                            horizontal_flip=True,
                                            height_shift_range=0.3,
                                            width_shift_range=0.3,
                                            rotation_range=30,
                                            preprocessing_function=preprocess_input)

        test_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

        self.train_data_gen = train_data_gen.flow_from_dataframe(
            dataframe=self.train_df,
            directory=None,
            class_mode="categorical",
            x_col="filename",
            y_col="class",
            shuffle=False,
            weight_col=None,
            classes=['crack', 'pothole', 'good'],
            target_size=(self.ROWS, self.COLS),
            batch_size=self.batch_size)

        self.test_data_gen = test_data_gen.flow_from_dataframe(
            dataframe=self.test_df,
            directory=None,
            class_mode="categorical",
            x_col="filename",
            y_col="class",
            shuffle=False,
            weight_col=None,
            classes=['crack', 'pothole', 'good'],
            target_size=(self.ROWS, self.COLS),
            batch_size=self.batch_size)

    def create_model(self):
        base_model = applications.InceptionResNetV2(weights='imagenet',
                                        include_top=False,
                                        input_shape=(self.ROWS, self.COLS, 3))

        for layer in base_model.layers[:5]:
            layer.trainable = False

        add_model = Sequential()
        add_model.add(base_model)
        add_model.add(Flatten())
        add_model.add(Dropout(0.2))
        add_model.add(Dense(1024, activation='relu'))
        add_model.add(Dropout(0.3))
        add_model.add(Dense(1024, activation='relu'))
        add_model.add(Dense(3, activation='softmax'))

        self.model = add_model

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizers.SGD(lr=1e-3,
                                                    momentum=0.9),
                           metrics=['accuracy'])
        self.model.summary()

    def train_model(self):
        checkpoint = ModelCheckpoint(filepath=self.file_path, monitor='accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        early = EarlyStopping(monitor="accuracy", mode="max", patience=15)

        callbacks_list = [checkpoint, early]  # early

        history = self.model.fit_generator(self.train_data_gen,
                                           epochs=self.epochs,
                                           validation_data=self.test_data_gen,
                                           verbose=True,
                                           callbacks=callbacks_list)

        with open('./history-4/inceptionresnetv2.json', 'w') as f:
            json.dump(history.history, f)

    def evaluate_model(self, noise):
        self.model.load_weights(self.file_path)
        predicts = self.model.predict(self.test_data_gen, verbose=True, batch_size=self.batch_size)
        predicts = np.argmax(predicts, axis=1)

        val_data = {'target': self.y_test, 'prediction': predicts.tolist()}

        with open('./validation-4/inceptionresnetv2' + noise + '.json', 'w') as f:
            json.dump(val_data, f)

        return accuracy_score(self.y_test, predicts)
