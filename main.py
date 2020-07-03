from models.inceptionv3 import InceptionV3Classifier
from models.vgg16 import VGG16Classifier
from models.vgg19 import VGG19Classifier
from models.resnet50 import ResNet15Classifier
from models.xception import XceptionClassifier
from preprocessing import DataPreprocessing

if __name__ == "__main__":

    epochs = 1
    data_path = '/home/bumuthudilshanhhk/Downloads/dataset/'
    test_ratio = 0.3
    batch_size = 16

    preproc = DataPreprocessing(data_path, test_ratio)
    preproc.prepare_image_path_df()

    # vgg16_model = VGG16Classifier(preproc.train_df, preproc.test_df, preproc.y_test, epochs)
    # vgg16_model.prepare_data_generator()
    # vgg16_model.make_vgg16_model()
    # vgg16_model.train_model()
    # vgg16_model.evaluate_model()

    inceptionv3_model = InceptionV3Classifier(preproc.train_df, preproc.test_df, preproc.y_test, epochs, batch_size)
    inceptionv3_model.prepare_data_generator()
    inceptionv3_model.make_inceptionv3_model()
    inceptionv3_model.train_model()
    inceptionv3_model.evaluate_model()