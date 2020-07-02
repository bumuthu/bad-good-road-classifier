from models.inceptionv3 import InceptionV3Classifier
from preprocessing import DataPreprocessing

if __name__ == "__main__":

    epochs = 1

    preproc = DataPreprocessing()

    preproc.prepare_image_path_df()

    inceptionv3_model = InceptionV3Classifier(preproc.train_df, preproc.test_df, preproc.y_test, epochs)

    inceptionv3_model.prepare_data_generator()
    inceptionv3_model.make_inceptionv3_model()
    inceptionv3_model.train_model()
    inceptionv3_model.evaluate_model()