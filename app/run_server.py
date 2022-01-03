#
from logging import handlers
import dill 
import numpy as np
import pandas as pd
from pathlib import Path
import os

import re
from nltk.stem import WordNetLemmatizer

from werkzeug.wrappers import request
dill._dill._reverse_typemap['ClassType'] = type

#
import flask 
import logging
from logging.handlers import RotatingFileHandler
from time import strftime

#
app = flask.Flask(__name__)
model = None

#Логирование
handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def load_model(model_path):
    global model
    with open(model_path, 'rb') as f:
        model = dill.load(f)
    print(model)

# Путь до модели 
modelpath = Path.cwd() /"app" /"models" / "SGDClassifier_pipeline.dill"
load_model(modelpath)


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i,'',input_txt)
    return input_txt


@app.route('/', methods=["GET"])
def general():
    return """ Welcome! Please use 'http://<address>/predict' to POST."""


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    dt = strftime("[%Y-%b-%d %H:%M:%S]")
    if flask.request.method == "POST":

        tweet = ""
        request_json = flask.request.get_json()
        if request_json["tweet"]:
            tweet = request_json["tweet"]
        logger.info(f'{dt} Data: tweeet={tweet}')

        df = pd.DataFrame({'Tweet': [tweet],})
        
        # Обработка входного твита 
        stemmer = WordNetLemmatizer()
        df['Tweet'] = np.vectorize(remove_pattern)(df['OriginalTweet'], '@[\w]*')
        df['Tweet'] = df['Tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
        df['Tweet'] = df['Tweet'].str.replace('[^a-zA-Z#]+',' ')
        df['Tweet'] = [stemmer.lemmatize(word) for word in df['Tweet']]
        df['Tweet'] = df['Tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))
        df['Tweet'] = df['Tweet'].apply(lambda x: x.split())
        df['Tweet'] = df['Tweet'].apply(lambda x: ''.join(w+" " for w in x))
        logger.info(f'{dt} Data: tweeet={df["Tweet"]}')
        
        try:
            preds = model.predict_proba(df)
        
        except AttributeError as e:
            logger.warning(f'{dt} Exseption: {str(e)}')
            data['predictions'] = str(e)
            data['success'] = False
            return flask.jsonify(data)
        
        data['predictions'] = preds[:, 1][0]
        data['success'] = True

    return flask.jsonify(data)


if __name__ == "__main__":
    print("* Loading the model and Flask starting server..."
        "please wait until server has fully started")
    port = int(os.environ.get('PORT', 8180))
    app.run(host='0.0.0.0', debug=True, port=port)

