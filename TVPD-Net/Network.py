
# Modules
from tensorflow.keras.layers import Lambda,GlobalAveragePooling2D,Dense,concatenate,Input,add,Reshape,LeakyReLU,BatchNormalization
from tensorflow.keras.layers import PReLU,UpSampling2D,Conv2D, Conv2DTranspose,MaxPooling2D, Activation,Flatten,Lambda,Add
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import sys
from tensorflow.keras.optimizers import Adam,SGD
import numpy as np
from tensorflow.python.keras.layers import InputSpec, Layer

from layer_utils import ReflectionPadding2D, res_block

# the paper defined hyper-parameter:chr
channel_rate = 64
# Note the image_shape must be multiple of patch_shape
patch_shape = (channel_rate, channel_rate, 3)
ngf = 64
ndf = 64
input_nc = 3
output_nc = 3
n_blocks_gen = 10

def get_gan_network(discriminator, shape, generator):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x,gan_output])
    return gan



def generator(shape,chan):
    """Build generator architecture."""
    # Current version : ResNet block
    inputs = Input(shape=shape)

    x = ReflectionPadding2D((3, 3))(inputs)
    x = Conv2D(filters=128, kernel_size=(7, 7), padding='valid')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)

    n_downsampling = 2
    for i in range(n_downsampling):
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=2, padding='same')(x)
        #x = BatchNormalization()(x)
        x = Activation('relu')(x)

    for i in range(n_blocks_gen):
        x = res_block(x, 128, use_dropout=True)

    for i in range(n_downsampling):
        x = UpSampling2D()(x)
        x = Conv2D(filters=int(128), kernel_size=(3, 3), padding='same')(x)
        #x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = ReflectionPadding2D((3, 3))(x)
    x = Conv2D(filters=chan, kernel_size=(7, 7), padding='valid')(x)
    x = Activation('tanh')(x)

    model = Model(inputs=inputs, outputs=x, name='Generator')
    return model
  

def discriminator(shape):
    """Build discriminator architecture."""
    n_layers = 2
    inputs = Input(shape=shape)

    x = Conv2D(filters=ndf, kernel_size=3, strides=2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)

    nf_mult, nf_mult_prev = 1, 1
    for n in range(n_layers):
        nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
        x = Conv2D(filters=ndf*nf_mult, kernel_size=3, strides=2, padding='same')(x)
        #x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

    nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
    x = Conv2D(filters=ndf*nf_mult, kernel_size = 3, strides=2, padding='same')(x)
    x = Conv2D(filters=ndf, kernel_size = 3, strides=2, padding='same')(x)
    #x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x, name='Discriminator')
    return model
