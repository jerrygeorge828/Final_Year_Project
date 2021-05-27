#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


# In[3]:


app = Flask(__name__)

MODEL_PATH ='pepper_trial_4.h5'

model = load_model(MODEL_PATH)


# In[5]:


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(128, 128))
    x = image.img_to_array(img)
    x=x/255
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="Healthy"
    else:
        preds="Diseased"
    return preds


# In[6]:


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# In[7]:


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


# In[ ]:


if __name__ == '__main__':
    app.run(port=5001,debug=True)

