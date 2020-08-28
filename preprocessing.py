import pandas as pd
import os
from sklearn.model_selection import train_test_split


class DataPreprocessing:

    def __init__(self, data_path, test_ratio):
        self.data_dir = data_path
        self.test_ratio = test_ratio
        self.ROWS = 139
        self.COLS = 139

    def prepare_image_path_df(self):
        paths = []
        y = []

        for crack_pth in os.listdir(self.data_dir + 'crack')[:1500]:
            paths.append(self.data_dir + 'crack/' + crack_pth)
            y.append('crack')
        for pothole_pth in os.listdir(self.data_dir + 'pothole')[:900]:
            paths.append(self.data_dir + 'pothole/' + pothole_pth)
            y.append('pothole')
        for good_pth in os.listdir(self.data_dir + 'good')[:1500]:
            paths.append(self.data_dir + 'good/' + good_pth)
            y.append('good')

        paths_train, paths_test, y_train, y_test = train_test_split(paths, y, test_size=self.test_ratio,
                                                                    random_state=42, shuffle=True)

        train_path_dict = [{"filename": paths_train[i], "class": str(y_train[i])} for i in range(len(paths_train))]
        test_path_dict = [{"filename": paths_test[i], "class": str(y_test[i])} for i in range(len(paths_test))]

        self.train_df = pd.DataFrame.from_dict(train_path_dict)
        self.test_df = pd.DataFrame.from_dict(test_path_dict)
        self.y_test = y_test

