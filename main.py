from models.inceptionv3 import InceptionV3Classifier
from preprocessing import DataPreprocessing

if __name__ == "__main__":

    preproc = DataPreprocessing()

    preproc.prepare_image_path_df()
    preproc.prepare_data_generator()

    inceptionv3_model = InceptionV3Classifier(preproc.preproc, preproc.test_data_gen, preproc.y_test)

    inceptionv3_model.make_inceptionv3_model()
    inceptionv3_model.train_model()
    inceptionv3_model.evaluate_model()