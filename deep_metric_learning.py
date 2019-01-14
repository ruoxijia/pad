import copy
import numpy as np
import numpy.random as rng
import pandas as pd
import keras
from keras.layers import Input,merge
from keras.models import Sequential, Model
from sklearn.preprocessing import StandardScaler
from keras.layers.core import *
from keras import optimizers
import pandas as pd
import pdb
import matplotlib.pyplot as plt
# np.random.seed(0)

def euc_dist(x):
    'Merge function: euclidean_distance(u,v)'
    s = x[0] - x[1]
    output = K.sum(K.square(s),axis=1,keepdims=True)
    return output

def euc_dist_shape(input_shape):
    'Merge output shape'
    shape = list(input_shape)
    outshape = (shape[0][0],1)
    return tuple(outshape)

class Deep_Metric:
    def __init__(self,mode):
        self.mode = mode
    def train(self, x1_train,x2_train,y_train,x1_test,x2_test,y_test):
        self.scaler = StandardScaler()
        self.batch_size = 32
        self.epochs = 100 #100 for arrival etc
        input_shape = x1_train[0].shape
        kernels = 32
        # print(input_shape)
        # define networks
        left_input = Input(shape=input_shape,batch_shape=(None,input_shape[0]))
        right_input = Input(shape=input_shape,batch_shape=(None,input_shape[0]))
        if self.mode == 'linear':
            model = Sequential()
            model.add(Dense(input_shape[0], activation=self.mode, use_bias=True, input_shape=input_shape))
        else:
            model = Sequential()
            model.add(Dense(kernels, activation=self.mode,use_bias=True, input_shape = input_shape))
            model.add(Dropout(0.2))
            model.add(Dense(int(kernels/2), activation=self.mode,use_bias=True))
            model.add(Dropout(0.2))
            model.add(Dense(kernels, activation=self.mode,use_bias=True))
            model.add(Dropout(0.2))
        encoded_l = model(left_input)
        encoded_r = model(right_input)
        both = merge([encoded_l,encoded_r],mode=euc_dist,output_shape=euc_dist_shape)
        self.distance_model = Model(input=[left_input,right_input],outputs=both)

        # optimizer
        learning_rate = 1e-3
        adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.distance_model.compile(loss=self.contrastive_loss, optimizer=adam)

        data = np.append(x1_train, x2_train, axis=0)
        self.scaler.fit(data)
        x1_train = self.scaler.transform(x1_train)
        x2_train = self.scaler.transform(x2_train)
        if x1_test.shape[0] != 0:
            x1_test = self.scaler.transform(x1_test)
            x2_test = self.scaler.transform(x2_test)
            print('===training===')
            self.distance_model.fit([x1_train, x2_train], y_train,
                                    batch_size=self.batch_size,
                                    epochs=self.epochs,
                                    validation_data=([x1_test, x2_test], y_test), verbose=0)
            self.d_test = self.distance_model.predict([x1_test, x2_test])
            # similar_ind = np.where(y_test == 1)[0]
            # dissimilar_ind = np.where(y_test == 0)[0]
            # print('similar distances %s' % np.mean(self.d_test[similar_ind]))
            # print('similar distances var %s' % np.std(self.d_test[similar_ind]))
            # print('dissimilar distances %s' % np.mean(self.d_test[dissimilar_ind]))
            # print('dissimilar distances var %s' % np.std(self.d_test[dissimilar_ind]))
        else:
            print('===training===')
            self.distance_model.fit([x1_train, x2_train], y_train,
                                batch_size=self.batch_size,
                                epochs=self.epochs,
                                verbose=0)





    def transform(self, data_pairs):
        x, y = data_pairs
        # x = x.reshape((1,len(x)))
        # y = y.reshape((1,len(y)))
        x = self.scaler.transform(np.array([x]))
        y = self.scaler.transform(np.array([y]))
        distance = self.distance_model.predict([x,y])
        return distance

    def contrastive_loss(self, y_true, d):
        # 1 means simlar, and 0 means dissimilar
        margin = 1
        return K.mean(y_true*0.5*d + (1-y_true)*0.5*K.maximum(margin-d,0))



