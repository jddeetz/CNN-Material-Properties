#Run the Optimizer

import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding3D, BatchNormalization, Flatten, Conv3D
from keras.layers import AveragePooling3D, MaxPooling3D, Dropout, GlobalMaxPooling3D, GlobalAveragePooling3D
from keras.models import Model
from keras import regularizers


def CNNModel(input_shape):
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeros
    X = ZeroPadding3D(padding=(2, 2, 2))(X_input) 

    # CONV -> BN -> RELU Block applied to X
    X = Conv3D(32, (5, 5, 5), strides=(1, 1, 1), padding='valid', kernel_regularizer=regularizers.l2(0.01), name='conv0')(X)
    X = BatchNormalization(axis=4, name='bn0')(X)
    X = Activation('relu')(X)
    # MAXPOOL
    X = MaxPooling3D((2, 2, 2), name='max_pool0')(X)

    # CONV -> BN -> RELU Block applied to X
    X = Conv3D(64, (5, 5, 5), strides=(1, 1, 1), padding='same', kernel_regularizer=regularizers.l2(0.01), name='conv1')(X)
    X = BatchNormalization(axis=4, name='bn1')(X)
    X = Activation('relu')(X)
    # MAXPOOL
    X = MaxPooling3D((2, 2, 2), name='max_pool1')(X)

    # CONV -> BN -> RELU Block applied to X
    X = Conv3D(128, (5, 5, 5), strides=(1, 1, 1), padding='same', kernel_regularizer=regularizers.l2(0.01), name='conv2')(X)
    X = BatchNormalization(axis=4, name='bn2')(X)
    X = Activation('relu')(X)
    # MAXPOOL
    X = MaxPooling3D((2, 2, 2), name='max_pool2')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(100, activation='relu', name='fc0', kernel_regularizer=regularizers.l2(0.01))(X)
    X = Dense(1, activation='relu', name='fc1', kernel_regularizer=regularizers.l2(0.01))(X)
    
    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='CNNModel')

    return model


