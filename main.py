from models.inceptionv3 import InceptionV3Classifier
from models.vgg19 import VGG19Classifier
from models.resnet50 import ResNet50Classifier
from models.xception import XceptionClassifier
from models.inceptionresnetv2 import InceptionResNetV2Classifier
from preprocessing import DataPreprocessing

if __name__ == "__main__":

    # number of epochs
    epochs = 1

    # select the path to data directory
    # '<path to dataset>'
    data_path = '/home/bumuthudilshanhhk/Downloads/dataset-new/'

    # data split ratio for training and testing
    test_ratio = 0.3

    # batch size for training
    batch_size = 16

    preproc = DataPreprocessing(data_path, test_ratio)
    preproc.prepare_image_path_df()

    vgg19_model = VGG19Classifier(preproc.train_df, preproc.test_df, preproc.y_test, epochs, batch_size)
    inceptionv3_model = InceptionV3Classifier(preproc.train_df, preproc.test_df, preproc.y_test, epochs, batch_size)
    resnet50_model = ResNet50Classifier(preproc.train_df, preproc.test_df, preproc.y_test, epochs, batch_size)
    xception_model = XceptionClassifier(preproc.train_df, preproc.test_df, preproc.y_test, epochs, batch_size)
    inceptionresnetv2_model = InceptionResNetV2Classifier(preproc.train_df, preproc.test_df, preproc.y_test, epochs, batch_size)

    def train_model(model):
        model.prepare_data_generator()
        model.create_model()
        model.train_model()

    def evaluate_model(model):
        model.prepare_data_generator()
        model.create_model()
        model.evaluate_model()

    # uncomment followings for training
    # train_model(vgg19_model)
    train_model(inceptionresnetv2_model)
    train_model(resnet50_model)
    train_model(xception_model)
    train_model(inceptionv3_model)

    # evaluate_model(vgg19_model)
    evaluate_model(inceptionresnetv2_model)
    evaluate_model(resnet50_model)
    evaluate_model(xception_model)
    evaluate_model(inceptionv3_model)




