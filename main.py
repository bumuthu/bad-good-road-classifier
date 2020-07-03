from models.inceptionv3 import InceptionV3Classifier
from models.vgg16 import VGG16Classifier
from models.vgg19 import VGG19Classifier
from models.resnet50 import ResNet50Classifier
from models.xception import XceptionClassifier
from preprocessing import DataPreprocessing

if __name__ == "__main__":

    epochs = 15
    data_path = '/home/bumuthudilshanhhk/Downloads/dataset/'
    test_ratio = 0.3
    batch_size = 16

    preproc = DataPreprocessing(data_path, test_ratio)
    preproc.prepare_image_path_df()

    vgg16_model = VGG16Classifier(preproc.train_df, preproc.test_df, preproc.y_test, epochs, batch_size)
    vgg19_model = VGG19Classifier(preproc.train_df, preproc.test_df, preproc.y_test, epochs, batch_size)
    inceptionv3_model = InceptionV3Classifier(preproc.train_df, preproc.test_df, preproc.y_test, epochs, batch_size)
    resnet50_model = ResNet50Classifier(preproc.train_df, preproc.test_df, preproc.y_test, epochs, batch_size)
    xception_model = XceptionClassifier(preproc.train_df, preproc.test_df, preproc.y_test, epochs, batch_size)

    def train_model(model):
        model.prepare_data_generator()
        model.create_model()
        model.train_model()

    def evaluate_model(model):
        model.prepare_data_generator()
        model.create_model()
        model.evaluate_model()

    train_model(resnet50_model)
    evaluate_model(resnet50_model)
    train_model(xception_model)
    evaluate_model(xception_model)
    train_model(inceptionv3_model)
    evaluate_model(inceptionv3_model)
    train_model(vgg16_model)
    evaluate_model(vgg16_model)
    train_model(vgg19_model)
    evaluate_model(vgg19_model)

    # evaluate_model(resnet50_model)
    # evaluate_model(xception_model)
    # evaluate_model(inceptionv3_model)
    # evaluate_model(vgg16_model)
    # evaluate_model(vgg19_model)
