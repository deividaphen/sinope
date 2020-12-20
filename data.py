__author__     = 'Deivid Aparecido Henrique'
__email__      = 'deividaphen@gmail.com'

import pandas as pd
import numpy as np
import keras
import matminer
import pickle

from sklearn.preprocessing import MinMaxScaler
from matminer.utils.io import load_dataframe_from_json, store_dataframe_as_json

#dataframe with all numerical descriptors
fdf = load_dataframe_from_json('metisdb.json')

#excluding non-ionic compounds
not_ionic = fdf['compound possible'] == 0
fdf = fdf[not_ionic]

#completing null or non-finite cells
fdf = fdf.fillna(0)
fdf = fdf.replace([np.inf, -np.inf], 0)

#possible properties to be predicted
targetsList = ['K_VRH','G_VRH','elastic_anisotropy','poisson_ratio']
y = fdf[targetsList]

#excluded = non-numerical descriptors
excluded = ['material_id', 'structure', 'elastic_anisotropy',
			'K_VRH', 'G_VRH', 'poisson_ratio', 'elasticity',
            'formula', 'composition', 'composition_oxid',
            'HOMO_character', 'HOMO_element',
            'LUMO_character', 'LUMO_element']

X = fdf.drop((targetsList+excluded), axis=1)

#normalizing values
scaler = MinMaxScaler(feature_range=(0, 1))
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

#correcting indexes
X, y = X.sort_index(), y.sort_index()

#exporting
print("The descriptor dataset has {} entries".format(fdf.shape))
print (X.head())

store_dataframe_as_json(X, 'data/descriptors.json', compression=None, orient='split')
store_dataframe_as_json(y, 'data/targets.json', compression=None, orient='split')