import argparse
import atexit
import os
import pickle
from os.path import isfile
import numpy as np
import cv2
from flask import Flask, request, Response
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

import face_model
from configs import *

app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

parser = argparse.ArgumentParser(description='face model recognition')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../resources/models/model,0', help='path to load model.')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)
print('Face model initialized.')

if not isfile(FEATURES_FILE):
    with open(FEATURES_FILE, 'wb') as file_writer:
        pickle.dump({}, file_writer)

with open(FEATURES_FILE, 'rb') as file_reader:
    features_dict = pickle.load(file_reader)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_features(name, file_path):
    img = cv2.imread(file_path)
    img = model.get_input(img)
    feature = model.get_feature(img)
    if feature.any() and (name not in features_dict):
        features_dict[name] = feature
        with open(FEATURES_FILE, 'wb') as file_writer:
            pickle.dump(features_dict, file_writer)
        return True
    else:
        return False


def get_identity(feature):
    all_names = list(features_dict.keys())
    all_embed = np.stack(list(features_dict.values()), axis=0)
    feature = np.expand_dims(feature, 0)
    simmilarity = np.squeeze(cosine_similarity(feature, all_embed), axis=0)
    index, max_val = np.argmax(simmilarity), np.max(simmilarity)
    name = None
    if max_val >= THRESHOLD:
        name = all_names[index.item()]
    return name


@app.route('/register_face', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        name = request.form['name']
        if file.filename == '':
            return Response("{No such files.}", status=400, mimetype='application/json')
        if name == '':
            return Response("{Name is not provided.}", status=400, mimetype='application/json')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # save features.
            if save_features(name, file_path):
                return Response("{Registered successfully.}", status=201, mimetype='application/json')
            else:
                return Response("{File can not be saved. Register failed.}", status=400, mimetype='application/json')
        else:
            return Response("{File extension is not allowed.}", status=400, mimetype='application/json')

    return Response("{Method not allowed.}", status=400, mimetype='application/json')


@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            return Response("{No such files.}", status=400, mimetype='application/json')
        if file and allowed_file(file.filename):
            np_img = np.fromstring(file.read(), np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)
            img = model.get_input(img)
            feature = model.get_feature(img)
            if feature.any():
                name = get_identity(feature)
                if name:
                    return Response("{name:%s}" % name, status=400, mimetype='application/json')
                else:
                    return Response("{Can not get identity}", status=400, mimetype='application/json')
            else:
                return Response("{Can not recognize human face.}", status=400, mimetype='application/json')
        else:
            return Response("{File extension is not allowed.}", status=400, mimetype='application/json')

    return Response("{Method not allowed.}", status=400, mimetype='application/json')


@app.route('/unregister_face', methods=['POST'])
def recognize_face():
    if request.method == 'POST':
        name = request.form['name']
        if name in features_dict:
            del features_dict[name]
            return Response("{message: Delete identity successfully.}", status=200, mimetype='application/json')
        return Response("{message: Name does not appear in database.}", status=200, mimetype='application/json')

    return Response("{Method not allowed.}", status=400, mimetype='application/json')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6868, debug=True, threaded=False)
