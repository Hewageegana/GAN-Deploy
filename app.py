import cv2
import io,os
import h5py
import base64
import argparse
import torch
import time
import pickle
import requests
import numpy as np
import pandas as pd
from torch import nn
from PIL import Image
from datetime import timedelta
import matplotlib.pyplot as plt
from torch.autograd import Variable
from diffusers.utils import load_image
from sklearn.neighbors import NearestNeighbors
from diffusers import DiffusionPipeline as GANpipeline
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

model_id = "zuu/gan-T2I-pipe"
pipe = GANpipeline.from_pretrained(model_id)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        prompt = request.form['prompt']
        image = pipe(prompt).images[0]  
        image.save(f"result/{prompt}.png")
        

    return jsonify({'image': f"result/{prompt}.png"})

if __name__ == '__main__':
    app.run(debug=false, host='0.0.0.0', port=5000)