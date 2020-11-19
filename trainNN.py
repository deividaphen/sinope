__author__     = 'Deivid Aparecido Henrique'
__email__      = 'deividaphen@gmail.com'

import pandas as pd
import numpy as np
import keras
import matminer
import pickle

from keras.models import load_model
from NN import trainNN, evaluateModel

#constants
epochs = 500
batch_size = 32

#creates NN predictor
X = pd.read_json('data/descriptors.json')
y = pd.read_json('data/targets.json')
yK = y['K_VRH']
yG = y['G_VRH']

#importing and training K model
model_raw = load_model('models/K_NN_raw.h5')
model = trainNN(model_raw, X, yK, epochs, batch_size)
model.save('models/K_NN.h5')
evaluateModel(model,X,yK)

#importing and training G model
model_raw = load_model('models/G_NN_raw.h5')
model = trainNN(model_raw, X, yG, epochs, batch_size)
model.save('models/G_NN.h5')
evaluateModel(model,X,yG)