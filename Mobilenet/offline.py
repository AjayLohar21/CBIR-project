from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np
import time

if __name__ == '__main__':
    fe = FeatureExtractor()
    start = time.time()
    for img_path in sorted(Path("./static/img1").glob("*.jpg")):
        print(img_path)  # e.g., ./static/img/xxx.jpg
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path("./static/feature1") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)

    for img_path in sorted(Path("./static/img1").glob("*.jpeg")):
        print(img_path)  # e.g., ./static/img/xxx.jpg
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path("./static/feature1") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)

    for img_path in sorted(Path("./static/img1").glob("*.png")):
        print(img_path)  # e.g., ./static/img/xxx.jpg
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path("./static/feature1") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)
    end = time.time()
    print((end-start)/60)