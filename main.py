from models.inceptionv3 import InceptionV3Classifier
from models.vgg16 import VGG16Classifier
from models.vgg19 import VGG19Classifier
from models.resnet50 import ResNet50Classifier
from models.xception import XceptionClassifier
from models.inceptionresnetv2 import InceptionResNetV2Classifier
from preprocessing import DataPreprocessing
import json

if __name__ == "__main__":

    # number of epochs
    epochs = 30

    # select the path to data directory
    # '<path to dataset>'
    data_path = '/home/bumuthudilshanhhk/Downloads/dataset/'

    # data split ratio for training and testing
    test_ratio = 0.3

    # batch size for training
    batch_size = 16

    def train_model(model):
        model.prepare_data_generator()
        model.create_model()
        model.train_model()

    def evaluate_model(model, noise):
        model.prepare_data_generator()
        model.create_model()
        return model.evaluate_model(noise)


    noise_levels = [0.05, 0.1, 0.2]


    results = {"resnet": [], "xception": [], "inceptionv3": [], "vgg19":[], "inceptionresnetv2": []}

    for func in ['rand']:
        for level in noise_levels:

            preproc = DataPreprocessing(data_path, test_ratio)
            preproc.prepare_image_path_df(func, level)

            vgg19_model = VGG19Classifier(preproc.train_df, preproc.test_df, preproc.y_test, epochs, batch_size)
            inceptionv3_model = InceptionV3Classifier(preproc.train_df, preproc.test_df, preproc.y_test, epochs, batch_size)
            resnet50_model = ResNet50Classifier(preproc.train_df, preproc.test_df, preproc.y_test, epochs, batch_size)
            xception_model = XceptionClassifier(preproc.train_df, preproc.test_df, preproc.y_test, epochs, batch_size)
            inceptionresnetv2_model = InceptionResNetV2Classifier(preproc.train_df, preproc.test_df, preproc.y_test, epochs, batch_size)

            noise = func + str(level)
            noise_attr = func+ '-level-' + str(level)

            results["resnet"].append({noise_attr : evaluate_model(resnet50_model, noise)})
            results["xception"].append({noise_attr: evaluate_model(xception_model, noise)})
            results["inceptionv3"].append({noise_attr: evaluate_model(inceptionv3_model, noise)})
            results["vgg19"].append({noise_attr: evaluate_model(vgg19_model, noise)})
            results["inceptionresnetv2"].append({noise_attr: evaluate_model(inceptionresnetv2_model, noise)})

            with open('./history/noise.json', 'w') as f:
                json.dump(results, f)

    # uncomment followings for training

    # train_model(resnet50_model)
    # train_model(xception_model)
    # train_model(inceptionv3_model)
    # train_model(vgg19_model)
    # train_model(inceptionresnetv2_model)

