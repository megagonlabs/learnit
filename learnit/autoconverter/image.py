import os

from keras.applications.densenet import DenseNet121
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PreTrainedModel:
    """Singleton pre-trained model

    All models will be loaded at first calling

    """

    model_dict = {}

    def __getitem__(self, key):
        if key not in self.model_dict:
            method_name = "_add_{}".format(key)
            getattr(self, method_name)()

        return self.model_dict[key]

    @classmethod
    def _add_resnet50(cls):
        cls.model_dict["resnet50"] = {
            "name": "resnet50",
            "model": ResNet50(weights='imagenet', include_top=False),
            "image_size": (224, 224)
        }

    @classmethod
    def _add_densenet121(cls):
        cls.model_dict["densenet121"] = {
            "name": "densenet121",
            "model": DenseNet121(weights='imagenet', include_top=False),
            "image_size": (224, 224)
        }

    @classmethod
    def _add_mobilenet(cls):
        cls.model_dict["mobilenet"] = {
            "name": "mobilenet",
            "model": MobileNet(
                weights='imagenet', include_top=False,
                input_shape=(224, 224, 3), pooling="avg"
            ),
            "image_size": (224, 224)
        }

    @staticmethod
    def get_output_dimension(model):
        len_output = len(model.output.shape)
        dim = model.output.shape[len_output - 1]

        return dim


class ImageVectorizer(BaseEstimator, TransformerMixin):
    """Extracts image vector with predefined Keras DL Model.

    Default model: image net
    Input: url list
    Output: The last layer vector

    """

    def __init__(self, way="mobilenet", batch_size=512):
        self.trained_model = PreTrainedModel()[way]
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Transform function

        Args:
            X (np.array): list of string objects

        Returns:
            np.array

        """
        X = np.array(X)
        loop_size = X.shape[0] // self.batch_size
        loop_size += ((X.shape[0] % self.batch_size) > 0)

        return np.array([self.img2vec(x) for x in np.split(X, loop_size)])

    def img2vec(self, path_list):
        """Transform function

        Args:
            path_list (np.array): path url list

        Returns:
            np.array

        """
        img_list = []
        for path in path_list:
            if not path.startswith("/"):
                path = os.getcwd() + "/" + path
            img = image.load_img(path,
                                 target_size=self.trained_model["image_size"])
            x = image.img_to_array(img)
            x = preprocess_input(x)
            img_list.append(x)
        img_arr = np.array(img_list)
        pred = self.trained_model["model"].predict(img_arr)
        for i in range(len(self.trained_model["model"].output.shape) - 1):
            pred = pred[0]

        return pred

    def get_feature_names(self):
        # To make sure self.lda is fitted (is there any better way?)
        clsname = self.__class__.__name__
        dim = PreTrainedModel.get_output_dimension(self.trained_model["model"])
        return [u"{}_{}_{}".format(clsname, self.trained_model["name"], x)
                for x in range(dim)]
