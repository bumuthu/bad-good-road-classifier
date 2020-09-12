import pandas as pd
import os
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from dtype import img_as_float

class DataPreprocessing:

    def __init__(self, data_path, test_ratio):
        self.data_dir = data_path
        self.test_ratio = test_ratio
        self.ROWS = 139
        self.COLS = 139

    def random_noise(self, image, mode='gaussian', seed=None, clip=True, **kwargs):

        mode = mode.lower()

        # Detect if a signed image was input
        if image.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.

        image = img_as_float(image)
        if seed is not None:
            np.random.seed(seed=seed)

        allowedtypes = {
            'gaussian': 'gaussian_values',
            'localvar': 'localvar_values',
            'poisson': 'poisson_values',
            'salt': 'sp_values',
            'pepper': 'sp_values',
            's&p': 's&p_values',
            'speckle': 'gaussian_values'}

        kwdefaults = {
            'mean': 0.,
            'var': 0.01,
            'amount': 0.05,
            'salt_vs_pepper': 0.5,
            'local_vars': np.zeros_like(image) + 0.01}

        allowedkwargs = {
            'gaussian_values': ['mean', 'var'],
            'localvar_values': ['local_vars'],
            'sp_values': ['amount'],
            's&p_values': ['amount', 'salt_vs_pepper'],
            'poisson_values': []}

        for key in kwargs:
            if key not in allowedkwargs[allowedtypes[mode]]:
                raise ValueError('%s keyword not in allowed keywords %s' %
                                 (key, allowedkwargs[allowedtypes[mode]]))

        # Set kwarg defaults
        for kw in allowedkwargs[allowedtypes[mode]]:
            kwargs.setdefault(kw, kwdefaults[kw])

        if mode == 'gaussian':
            noise = np.random.normal(kwargs['mean'], kwargs['var'] ** 0.5,
                                     image.shape)
            out = image + noise

        elif mode == 'localvar':
            # Ensure local variance input is correct
            if (kwargs['local_vars'] <= 0).any():
                raise ValueError('All values of `local_vars` must be > 0.')

            # Safe shortcut usage broadcasts kwargs['local_vars'] as a ufunc
            out = image + np.random.normal(0, kwargs['local_vars'] ** 0.5)

        elif mode == 'poisson':
            # Determine unique values in image & calculate the next power of two
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))

            # Ensure image is exclusively positive
            if low_clip == -1.:
                old_max = image.max()
                image = (image + 1.) / (old_max + 1.)

            # Generating noise for each unique value in image.
            out = np.random.poisson(image * vals) / float(vals)

            # Return image to original range if input was signed
            if low_clip == -1.:
                out = out * (old_max + 1.) - 1.

        elif mode == 'salt':
            # Re-call function with mode='s&p' and p=1 (all salt noise)
            out = self.random_noise(image, mode='s&p', seed=seed,
                               amount=kwargs['amount'], salt_vs_pepper=1.)

        elif mode == 'pepper':
            # Re-call function with mode='s&p' and p=1 (all pepper noise)
            out = self.random_noise(image, mode='s&p', seed=seed,
                               amount=kwargs['amount'], salt_vs_pepper=0.)

        elif mode == 's&p':
            out = image.copy()
            p = kwargs['amount']
            q = kwargs['salt_vs_pepper']
            flipped = np.random.choice([True, False], size=image.shape,
                                       p=[p, 1 - p])
            salted = np.random.choice([True, False], size=image.shape,
                                      p=[q, 1 - q])
            peppered = ~salted
            out[flipped & salted] = 1
            out[flipped & peppered] = low_clip

        elif mode == 'speckle':
            noise = np.random.normal(kwargs['mean'], kwargs['var'] ** 0.5,
                                     image.shape)
            out = image + image * noise

        # Clip back to original range, if necessary
        if clip:
            out = np.clip(out, low_clip, 1.0)

        return out

    def salt_pepper(self, image, noise_level):
        noise = self.random_noise(image, mode='s&p', amount=noise_level)
        out = np.array(255 * noise, dtype=np.uint8)
        return out

    def localvar_noise(self, image, noise_level):
        out = image + noise_level * image.std() * np.random.random(image.shape)
        return out

    # Save sault and pepper noisy images in a datset directory
    def save_sp_noises(self, paths):
        noise_levels = [0.05, 0.1, 0.2, 0.3]
        for n in noise_levels:
            for p in paths:
                path = p["filename"]
                img = cv2.imread(path)
                img_new = self.salt_pepper(img, n)
                new_path = self.data_dir + 'sp' + str(n) + '/' + os.path.basename(path)
                cv2.imwrite(new_path, img_new)

    # Save random noisy images in a datset directory
    def save_rand_noises(self, paths):
        noise_levels = [0.05, 0.1, 0.2, 0.3]
        for n in noise_levels:
            for p in paths:
                path = p["filename"]
                img = cv2.imread(path)
                img_new = self.salt_pepper(img, n)
                new_path = self.data_dir + 'rand' + str(n) + '/' + os.path.basename(path)
                cv2.imwrite(new_path, img_new)

    # Read sault and pepper noisy images
    def add_sp_noises(self, paths, level):
        new_dict = []
        n = level

        for p in paths:
            path = p["filename"]
            new_path = self.data_dir + 'sp' + str(n) + '/' + os.path.basename(path)

            new_dict.append({"filename": new_path, "class": p["class"]})
        return new_dict

    # Read random noisy images
    def add_rand_noises(self, paths, level):

        new_dict = []
        n = level

        for p in paths:
            path = p["filename"]
            new_path = self.data_dir + 'rand' + str(n) + '/' + os.path.basename(path)

            new_dict.append({"filename": new_path, "class": p["class"]})
        return new_dict

    def prepare_image_path_df(self, func, level):
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

        if func == 'sp':
            test_path_dict = self.add_sp_noises(test_path_dict, level)
        elif func == 'rand':
            test_path_dict = self.add_rand_noises(test_path_dict, level)

        self.train_df = pd.DataFrame.from_dict(train_path_dict)
        self.test_df = pd.DataFrame.from_dict(test_path_dict)
        self.y_test = y_test

