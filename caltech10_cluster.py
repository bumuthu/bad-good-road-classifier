import os
import math
import pickle
import operator
import cv2 as cv
import numpy as np
from scipy import ndimage
from sklearn.utils import shuffle
from itertools import islice
from sklearn.svm import SVC
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

class Caltech10Classifier(object):
    def __init__(self):
        self.n_classes = 10
        self.n_train = 90
        self.n_clusters = 300
        self.data_dir = os.path.join(os.getcwd(),'101_ObjectCategories')
        self.visual_words_path = os.path.join(os.getcwd(),'visual_words.npy')
        self.train_feature_path = os.path.join(os.getcwd(),'train_features')
        self.test_feature_path  = os.path.join(os.getcwd(),'test_features')

    def choose_classes(self):
        class_dict = {}
        for label in os.listdir(self.data_dir):
            label_dir = os.path.join(self.data_dir, label)
            if len(os.listdir(label_dir)) > 100:
                class_dict[label] = len(os.listdir(label_dir))
        class_dict = dict(sorted(class_dict.items(), key=operator.itemgetter(1),reverse=True))
        class_dict = {k: class_dict[k] for k in list(class_dict)[:self.n_classes]}
        return class_dict

    def load_data(self):
        image_dict = {}
        class_dict = self.choose_classes()
        min_class_size = list(class_dict.values())[-1]
        print("Each class has {} images".format(min_class_size))
        for label in list(class_dict.keys()):
            label_dir = os.path.join(self.data_dir, label)
            label_images = []
            for idx,img_name in enumerate(os.listdir(label_dir)):
                img_path = os.path.join(label_dir, img_name)
                img = cv.imread(img_path)
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                if img is not None:
                    label_images.append(img)
                if idx == min_class_size - 1:
                    break
            label_images = shuffle(label_images)
            image_dict[label] = label_images
        return image_dict

    def split_data(self,image_dict):
        self.train_dict = {k: image_dict[k][:self.n_train] for k in list(image_dict)}
        self.test_dict  = {k: image_dict[k][self.n_train:] for k in list(image_dict)}

    def sift_extractor(self, img):
        self.extractor = cv.xfeatures2d.SIFT_create()
        keypoints, descriptors = self.extractor.detectAndCompute(img, None)
        return keypoints, descriptors

    def compute_features(self,data_dict,train_first=True): # pass train_dict, test_dict
        feature_array = []
        label_array = []
        descriptor_list = []
        for label,label_images in data_dict.items():
            for img in label_images:
                keypoints, descriptors = self.sift_extractor(img)
                if descriptors is not None:
                    descriptor_list.extend(descriptors)
                    feature_array.append(descriptors)
                    label_array.append(label)
        descriptor_list = np.array(descriptor_list)
        feature_array, label_array = np.array(feature_array), np.array(label_array)
        feature_array, label_array = shuffle(feature_array, label_array)
        if train_first:
            self.scalar = StandardScaler()
            self.scalar.fit(descriptor_list)
        descriptor_list = self.scalar.transform(descriptor_list)
        return [descriptor_list, feature_array, label_array]

    def visual_vocabulary(self):
        if not os.path.exists(self.visual_words_path):
            descriptor_list = self.compute_features(self.train_dict)[0]
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
            kmeans.fit(descriptor_list)
            visual_words = kmeans.cluster_centers_
            print("visual_vocabulary is created")
            np.save(self.visual_words_path, visual_words)
        else:
            print("visual_vocabulary is Loading")
            visual_words = np.load(self.visual_words_path)
        return visual_words

    @staticmethod
    def find_index(v1, v2):
        distances = []
        for val in v2:
            dist = distance.euclidean(v1, val)
            distances.append(dist)
        return np.argmax(np.array(distances))

    def create_histogram(self, feature_array, visual_words):
        feature_hist = []
        for features in feature_array:
            histogram = np.zeros(len(visual_words))
            for feature in features:
                histogram[Caltech10Classifier.find_index(feature, visual_words)] += 1
            feature_hist.append(histogram)
        feature_hist = np.array(feature_hist)
        return feature_hist

    def normalize_features(self,feature_array):
        norm_feature_array =[]
        for features in feature_array:
            features = self.scalar.transform(features)
            norm_feature_array.append(np.array(features))
        return norm_feature_array

    def feature_histograms(self):
        visual_words = self.visual_vocabulary()

        if not os.path.exists(self.train_feature_path):
            train_feature_array, train_labels = self.compute_features(self.train_dict,False)[1:]
            train_feature_array = self.normalize_features(train_feature_array)
            train_features = self.create_histogram(train_feature_array, visual_words)

            print("train data is created")
            outfile = open(self.train_feature_path,'wb')
            train_data = {
                "features" : train_features,
                "labels" : train_labels
            }
            pickle.dump(train_data,outfile)
            outfile.close()
        else:
            print("train features are Loading")
            infile = open(self.train_feature_path,'rb')
            train_data = pickle.load(infile)
            infile.close()

        if not os.path.exists(self.test_feature_path):
            test_feature_array, test_labels = self.compute_features(self.test_dict,False)[1:]
            test_feature_array = self.normalize_features(test_feature_array)
            test_features = self.create_histogram(test_feature_array, visual_words)

            print("test features is created")
            outfile = open(self.test_feature_path,'wb')
            test_data = {
                "features" : test_features,
                "labels" : test_labels
            }
            pickle.dump(test_data,outfile)
            outfile.close()
        else:
            print("train features are Loading")
            infile = open(self.test_feature_path,'rb')
            test_data = pickle.load(infile)
            infile.close()
        return train_data, test_data

    def svm_classifier(self):
        train_data, test_data = self.feature_histograms()
        Xtrain, Ytrain = train_data['features'], train_data['labels']
        Xtest, Ytest = test_data['features'], test_data['labels']

        classifier = SVC(C=100, kernel='rbf' , gamma=0.001)
        classifier.fit(Xtrain, Ytrain)

        Ypred_train = classifier.predict(Xtrain)
        train_score = accuracy_score(Ytrain,Ypred_train)
        print("Train Accuracy :",train_score)

        Ypred_test = classifier.predict(Xtest)
        test_score = accuracy_score(Ytest,Ypred_test)
        print("Test Accuracy :",test_score)

if __name__ == "__main__":
    model = Caltech10Classifier()
    model.choose_classes()
    image_dict = model.load_data()
    model.split_data(image_dict)
    model.svm_classifier()
