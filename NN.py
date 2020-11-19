import numpy as np
import pandas as pd
import pickle
import tensorflow as tf 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from keras.callbacks import CSVLogger
from keras.layers import Dense, Activation, LeakyReLU
from keras.models import Sequential, load_model
from keras.wrappers.scikit_learn import KerasRegressor

RSEED = 11277149           # Random seed to ensure reproducibility
FOLDS = 5            # N-fold cross validation, i.e. 1/N of samples reserved to testing 
SEARCH = 300         # number of models that will be searched randomly
epochs = 150
batch_size = 32
min_neurons = 20
max_neurons = 300
np.set_printoptions(precision=3)

def searchNN(X, y):
    # Input the full dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1/FOLDS), random_state=RSEED)
    
    def create_model(neurons_hidden_1=1, neurons_hidden_2=1, neurons_hidden_3=1):
        # A method to create a generic NN with 3 layers
        model = Sequential()

        # Input layer
        model.add(Dense(neurons_hidden_1, input_dim=X.shape[1]))
        model.add(layer=LeakyReLU(alpha=0.01))

        # Hidden layers
        model.add(Dense(neurons_hidden_2, activation='relu'))
        model.add(layer=LeakyReLU(alpha=0.01))
        model.add(Dense(neurons_hidden_3, activation='relu'))
        model.add(layer=LeakyReLU(alpha=0.01))

        # Output layer
        model.add(Dense(1, activation='relu'))
        model.add(layer=LeakyReLU(alpha=0.01))
        model.compile(loss='mean_squared_error', optimizer='adam')

        return model
    
    kreg = KerasRegressor(build_fn=create_model, verbose=0)
    
    nSeries = []
    for i in range(min_neurons, max_neurons):
        nSeries.append(i)

    param_grid = {
    'neurons_hidden_1': nSeries,
    'neurons_hidden_2': nSeries,
    'neurons_hidden_3': nSeries}
    
    rs_model = RandomizedSearchCV(kreg, param_grid, n_jobs = -1, cv=FOLDS,
                              scoring = 'neg_mean_squared_error',  n_iter = SEARCH, 
                              verbose = 1, random_state=RSEED)
                              
    
    rs_model.fit(X_train, y_train)
    print('A summary of the best NN model:{}'.format(rs_model.best_params_))
    
    results_df = pd.DataFrame(rs_model.cv_results_)

    # Defining the number of neurons based on the best estimator
    nh3 = list(rs_model.best_params_.values())[0]
    nh2 = list(rs_model.best_params_.values())[1]
    nh1 = list(rs_model.best_params_.values())[2]

    model = create_model(nh1, nh2, nh3)

    return (model, results_df)

def trainNN(model, X, y, epochs, batch_size):
    # Input the model to be evaluated,the full dataset, and the numbers of training steps
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1/FOLDS), random_state=RSEED)
    
    csv_log = CSVLogger('training.log', separator=',', append=False)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[csv_log])

    return model

def evaluateModel(model, X, y):
    # Input the model to be evaluated and the full dataset.    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1/FOLDS), random_state=RSEED)
        
    # Making predictions
    ytrain_pred = model.predict(X_train)
    ytest_pred = model.predict(X_test)
    
    mse_train = np.sqrt(mean_squared_error(y_train, ytrain_pred))
    print('Train RMSE = {}'.format(mse_train))
    
    mse_test = np.sqrt(mean_squared_error(y_test, ytest_pred))
    print('Test RMSE = {}'.format(mse_test))    

    return()

#
# --- end