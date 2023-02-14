import numpy as np
from PIL import Image
from Densenet import FeatureExtractor1
from efficientnetmod import FeatureExtractor2
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
from io import FileIO


app = Flask(__name__)

# Read image features
fe1 = FeatureExtractor1()
fe2=FeatureExtractor2()
features = []
img_paths = []
for feature_path in Path("./static/feature1").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img1") / (feature_path.stem + ".jpeg"))
 
features = np.array(features)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        queryA = fe1.extract(img)
        queryB = fe2.extract(img)
        query=np.add(queryA,queryB)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[1:11]  # Top 10 results
        scores = [(dists[id], img_paths[id]) for id in ids]
        name_path=[]
        for id in scores:
            nam=id[1]
            nam=repr(nam)
            idx=nam.index("_")
            nam=nam[25:idx]
            
            
            
            name_path.append(nam)
        # name_path=[(dists[id])for id in ids]
        inx=uploaded_img_path.index("_")
        inp=uploaded_img_path[inx+1:]
        inx=inp.index("_")
        inp=inp[0:inx]
        c=name_path.count(inp) 
        recall = c/200
        precision=c/10
        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores,precision=precision,recall=recall)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run("0.0.0.0")
