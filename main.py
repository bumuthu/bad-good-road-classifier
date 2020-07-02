from models.InceptionV3 import InceptionV3Classifier

if __name__ == "__main__":

    inceptionv3_model = InceptionV3Classifier()

    inceptionv3_model.prepare_image_path_df()
    inceptionv3_model.prepare_data_generator()
    inceptionv3_model.make_inceptionv3_model()
    inceptionv3_model.train_model()
    inceptionv3_model.evaluate_model()