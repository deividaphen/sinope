import numpy as np
import pandas as pd
import pickle
import math
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.layers import Dense, Activation, LeakyReLU, Dropout
from keras.models import Sequential, load_model
from keras.wrappers.scikit_learn import KerasRegressor

RSEED = 11277149     # random fixed seed
FOLDS = 5            # N-fold cross validation; 1/N of the samples used for testing the models
SEARCH = 2 #300      # number of models that will be searched randomly to optimize hyperparameters
epochs = 4 #100       # number of training steps during search
batch_size = 32      # number of samples used in each epoch
min_neurons = 20     # minimum number of neurons in each layer
max_neurons = 300    # maximum number of neurons in each layer
k_SEARCH = 2 #10     # number of k values tested
k_min,k_max = 0,0.1  # minimum and maximum k values

# searchNN - optimizes number of neurons in each layer
def searchNN(X, y):
    # dividing databases into train and test
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
                              verbose = 0, random_state=RSEED)
    
    rs_model.fit(X_train, y_train, epochs=50)

    print('Summary of the NN model with optimized neuron numbers:{}'.format(rs_model.best_params_))
    
    results_df = pd.DataFrame(rs_model.cv_results_)

    # Defining the number of neurons based on the best estimator
    nh3 = list(rs_model.best_params_.values())[0]
    nh2 = list(rs_model.best_params_.values())[1]
    nh1 = list(rs_model.best_params_.values())[2]

    model = create_model(nh1, nh2, nh3)

    return (model, nh1, nh2, nh3, results_df)

# searchNN - optimizes other hyperparemeters
def searchHyper(nh1, nh2, nh3, X, y):
    # dividing databases into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1/FOLDS), random_state=RSEED)

    def create_model(leak=0.01, drop=0.2, lera=0.001):
        
        model = Sequential()

        # input
        model.add(Dense(nh1, input_dim=X.shape[1]))
        model.add(layer=LeakyReLU(alpha=leak))
        model.add(Dropout(drop))

        # hidden layers
        model.add(Dense(nh2, activation='relu'))
        model.add(layer=LeakyReLU(alpha=leak))
        model.add(Dropout(drop))

        model.add(Dense(nh3, activation='relu'))
        model.add(layer=LeakyReLU(alpha=leak))
        model.add(Dropout(drop))

        # output
        model.add(Dense(1, activation='relu'))
        model.add(layer=LeakyReLU(alpha=leak))

        opt = tf.keras.optimizers.Adam(lera)
        model.compile(loss='mean_squared_error', optimizer=opt)

        return model

    kreg = KerasRegressor(build_fn=create_model, verbose=0)
    
    leak_vec = []
    drop_vec = []
    lera_vec = []
    len_vec = 20
    for i in range(len_vec):
    	leak_vec.append((i/len_vec)*(0.2-0.01)+0.01)
    	drop_vec.append((i/len_vec)*(0.5-0.05)+0.05)
    	lera_vec.append((i/len_vec)*(0.02-0.001)+0.001)

    param_grid = {
    'leak': leak_vec,
    'drop': drop_vec,
    'lera': lera_vec}
    
    rs_model = RandomizedSearchCV(kreg, param_grid, n_jobs=-1, cv=FOLDS,
                              scoring='neg_mean_squared_error', n_iter=SEARCH,
                              verbose=0, random_state=RSEED)

    rs_model.fit(X_test, y_test, epochs=epochs)

    print('A summary of the best NN model with optimized hyperparameters:{}'.format(rs_model.best_params_))

    leak = list(rs_model.best_params_.values())[0]
    drop = list(rs_model.best_params_.values())[1]
    lera = list(rs_model.best_params_.values())[2]

    model = create_model(leak, drop, lera)
   
    return (model, leak, drop, lera)

def trainNN(X, y, epochs, batch_size, nh1, nh2, nh3, leak, drop, lera):
    # dividing databases into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1/FOLDS), random_state=RSEED)
    
    #create new model
    def create_model(k):
        
        model = Sequential()

        # input
        model.add(Dense(nh1, input_dim=X.shape[1]))
        model.add(layer=LeakyReLU(alpha=leak))
        model.add(Dropout(drop))

        # hidden layers
        model.add(Dense(nh2, activation='relu'))
        model.add(layer=LeakyReLU(alpha=leak))
        model.add(Dropout(drop))

        model.add(Dense(nh3, activation='relu'))
        model.add(layer=LeakyReLU(alpha=leak))
        model.add(Dropout(drop))

        # output
        model.add(Dense(1, activation='relu'))
        model.add(layer=LeakyReLU(alpha=leak))

        def decay(epoch):
            leraUpdated = lera*math.exp(-k*(epoch))
            return leraUpdated

        opt = tf.keras.optimizers.Adam(lr=lera,decay=decay)

        model.compile(loss='mean_squared_error', optimizer=opt)

        return model

    kreg = KerasRegressor(build_fn=create_model, verbose=0))

    # possible k values
    k_vec = []
    for i in range(k_SEARCH):
        k_vec.append((i/k_SEARCH)*(k_max-k_min)+k_min)

    param_grid={'k':k_vec}

    # fitting randomized search
    rs_model = RandomizedSearchCV(kreg, param_grid, n_jobs=-1, cv=FOLDS,
                              scoring='neg_mean_squared_error', n_iter=SEARCH,
                              verbose=0, random_state=RSEED)


    csv_log = CSVLogger('training.log', separator=',', append=False)

    rs_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[csv_log,lrate])

    print('A summary of the best NN model with optimized Learning Rate Decay:{}'.format(rs_model.best_params_))

    k = list(rs_model.best_params_.values())[0]

    #creating final model
    model = create_model(k)

    return (model, k)

def evaluateModel(model, X, y):
    # dividing databases into train and test    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1/FOLDS), random_state=RSEED)
        
    # making predictions
    ytrain_pred = model.predict(X_train)
    ytest_pred = model.predict(X_test)
    
    # train and test results (square root of the mean squared error)
    mse_train = np.sqrt(mean_squared_error(y_train, ytrain_pred))
    print('Train RMSE = {}'.format(mse_train))
    
    mse_test = np.sqrt(mean_squared_error(y_test, ytest_pred))
    print('Test RMSE = {}'.format(mse_test))

    return(mse_train,mse_test)