from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import EfficientNetB2 , preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from collections import OrderedDict
from functools import partial


class FeatureExtractor:
    def __init__(self):
        base_model = EfficientNetB2(weights = 'imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('predictions').output)

    def extract(self, img):
        """
        Extract a deep feature from an input image
        Args:
            img: from PIL.Image.open(path) or tensorflow.keras.preprocessing.image.load_img(path)

        Returns:
            feature (np.ndarray): deep feature with the shape=(4096, )
        """
        img = img.resize((260, 260))  # EfficientNet must take a 260x260 img as an input
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel
        feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
        return feature / np.linalg.norm(feature)  # Normalize

