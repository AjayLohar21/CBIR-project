from PIL import Image
from Densenet import FeatureExtractor1
from efficientnetmod import FeatureExtractor2
from pathlib import Path
import numpy as np
import time

if __name__ == '__main__':
    fe1 = FeatureExtractor1()
    fe2=FeatureExtractor2()
    start = time.time()
    for img_path in sorted(Path("./static/img1").glob("*.jpg")):
        print(img_path)  # e.g., ./static/img/xxx.jpg
        feature1 = fe1.extract(img=Image.open(img_path))
        feature2 = fe2.extract(img=Image.open(img_path))
        feature=np.add(feature1,feature2)
        feature_path = Path("./static/feature1") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)

    for img_path in sorted(Path("./static/img1").glob("*.jpeg")):
        print(img_path)  # e.g., ./static/img/xxx.jpg
        feature1 = fe1.extract(img=Image.open(img_path))
        feature2 = fe2.extract(img=Image.open(img_path))
        feature=np.add(feature1,feature2)
        feature_path = Path("./static/feature1") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)

    for img_path in sorted(Path("./static/img1").glob("*.png")):
        print(img_path)  # e.g., ./static/img/xxx.jpg
        feature1 = fe1.extract(img=Image.open(img_path))
        feature2 = fe2.extract(img=Image.open(img_path))
        feature=np.add(feature1,feature2)
        feature_path = Path("./static/feature1") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)
    end = time.time()
    print((end-start)/60)