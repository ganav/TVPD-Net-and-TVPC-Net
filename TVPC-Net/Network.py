from tensorflow.keras.layers import Lambda,GlobalAveragePooling2D,Dense,concatenate,Input,add,Reshape,LeakyReLU, PReLU,UpSampling2D,Conv2D, Conv2DTranspose,MaxPooling2D, Activation,Flatten,Lambda
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import tensorflow.keras as K
import sys
from tensorflow.keras.optimizers import Adam,SGD
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers import InputSpec, Layer


def res_block(model, kernal_size, filters, strides):
    gen = model
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model) 
    model = add([gen, model])
    return model


def generator(shape,shape2,n_class):

    inp1_ = Input(shape = shape)# add concat
    inp2_ = Input(shape = shape2)# add concat

    inp1 = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "valid")(inp1_)
    inp1 = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "valid")(inp1)
    inp1 = MaxPooling2D((2,2), strides=(2,2))(inp1)
    for index in range(3):
        inp1 = res_block(inp1, 3, 128, 1)

    inp2 = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "valid")(inp2_)
    inp2 = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "valid")(inp2)
    inp2 = MaxPooling2D((2,2), strides=(2,2))(inp2)
    for index in range(3):
        inp2 = res_block(inp2, 3, 128, 1)

    model = concatenate([inp1,inp2],axis=-1)

    model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "valid")(model)
    model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "valid")(model)
    model = MaxPooling2D((2,2), strides=(2,2))(model)
    for index in range(3):
        model = res_block(model, 3, 128, 1)

    model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "valid")(model)
    model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "valid")(model)
    model = MaxPooling2D((2,2), strides=(2,2))(model)
    for index in range(3):
        model = res_block(model, 3, 128, 1)

    model = Flatten()(model)
    model = Dense(n_class,name='last_dense')(model)
    model = Activation('softmax')(model)
           
    model2 = Model(inputs = [inp1_,inp2_], outputs = model)
    optimizer = Adam(learning_rate=0.00001, decay=0.000001)
    model2.compile(loss=['categorical_crossentropy'],
        optimizer=optimizer,metrics=['accuracy'])

    return model2