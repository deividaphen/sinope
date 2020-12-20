__author__     = 'Deivid Aparecido Henrique'
__email__      = 'deividaphen@gmail.com'

import os
import pandas as pd
import numpy as np
import keras
import matminer
from matminer.utils.io import load_dataframe_from_json, store_dataframe_as_json
import pickle

from NN import searchNN, searchHyper, evaluateModel, trainNN

print ("\nAvailable cores: {}".format(os.sched_getaffinity(0)))
os.sched_setaffinity(0, {0, 2}) # running on one core
print ("\nRunning on core/cores {}".format(os.sched_getaffinity(0)))

#constants
epochs = 10 #3000
batch_size = 32

#imports data
X = load_dataframe_from_json('data/descriptors.json')
y = load_dataframe_from_json('data/targets.json')
yK = y['K_VRH']
yG = y['G_VRH']


#searching and training best model for K_VRH
print('\n---------------Searching for K_VRH model---------------\n')

#optimizing neuron numbers
model_raw, nh1K, nh2K, nh3K, gs_df = searchNN(X,yK)
gs_df.to_csv('data/K_gs.csv')
model_raw.save('models/K_NN_raw.h5')
print('\nRaw model predictions:')
evaluateModel(model_raw,X,yK)

#optimizing other hyperparameters:
model, leakK, dropK, leraK = searchHyper(nh1K, nh2K, nh3K, X, yK)
model.save('models/K_NN_raw2.h5')
print('\nOptimized model predictions:')
evaluateModel(model,X,yK)

#final training K model
modelK, kK= trainNN(X, yK, epochs, batch_size, nh1K, nh2K, nh3K, leakK, dropK, leraK)
modelK.save('models/K_NN.h5')


#searching and training best model for G_VRH
print('\n---------------Searching for G_VRH model---------------\n')

#optimizing neuron numbers
model_raw, nh1G, nh2G, nh3G, gs_df = searchNN(X,yG)
gs_df.to_csv('data/G_gs.csv')
model_raw.save('models/G_NN_raw.h5')
print('\nRaw model predictions:')
evaluateModel(model_raw,X,yG)

model, leakG, dropG, leraG = searchHyper(nh1G, nh2G, nh3G, X, yG)
model.save('models/G_NN_raw2.h5')
print('\nOptimized model predictions:')
evaluateModel(model,X,yG)

#final training G model
modelG, kG = trainNN(X, yG, epochs, batch_size, nh1G, nh2G, nh3G, leakG, dropG, leraG)
modelG.save('models/G_NN.h5')


#final results
print('\n---------------Final models---------------')
print('\nBulk Modulus')
print('Number of neurons on layer 1: ',nh1K)
print('Number of neurons on layer 2: ',nh2K)
print('Number of neurons on layer 3: ',nh3K)
print('Leak: ',leakK)
print('Dropout: ',dropK)
print('Learning rate : ',leraK)
print('Constant k of LR decay: ',kK)
evaluateModel(modelK,X,yK)

print('\nShear Modulus')
print('Number of neurons on layer 1: ',nh1G)
print('Number of neurons on layer 2: ',nh2G)
print('Number of neurons on layer 3: ',nh3G)
print('Leak: ',leakG)
print('Dropout: ',dropG)
print('Learning rate : ',leraG)
print('Constant k of LR decay: ',kG)
evaluateModel(modelG,X,yG)