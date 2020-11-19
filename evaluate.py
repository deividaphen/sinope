__author__     = 'Deivid Aparecido Henrique'
__email__      = 'deividaphen@gmail.com'

import pandas as pd

from keras.models import load_model
from matminer.utils.io import load_dataframe_from_json
from NN import evaluateModel

#imports data
X = pd.read_json('data/descriptors.json')
y = pd.read_json('data/targets.json')
yK = y['K_VRH']
yG = y['G_VRH']

model = load_model('models/K_NN_raw.h5')
print('\n---------------K_VRH raw results (K, X)---------------')
evaluateModel(model,X,yK)