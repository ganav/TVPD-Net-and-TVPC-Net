
import Network, math
import  Utils,random
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input,Lambda
from tensorflow.keras.layers import TimeDistributed
from tqdm import tqdm
import numpy as np
import argparse,csv, time,cv2
from tensorflow.keras.optimizers import Adam,SGD
#import metrics, test
#from keras_apps_ import efficientNet, resnet_common, densenet, inception_resnet_v2,inception_v4

np.random.seed(10)


def train(epochs, batch_size, output_dir, model_save_dir, ext,n_class,chan):
    image_shape = (300,300,1)
    image_shape2 = (300,300,3)

    generator = Network.generator(image_shape,image_shape2,n_class)

    print(generator.summary())
 
    x_train,label1 = Utils.load_training_data('G:\\projects\\paper 18\\source\\data\\visible dataset divided\\db1\\class_blur\\','', ext,n_class)
   
    batch_count = len(x_train)


    list_a = list(range(0,batch_count))
    print(list_a[:10])
    random.shuffle(list_a)
    print(list_a[:10])

    for e in tqdm(range(1, epochs+1)):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        loss1 = 0
        acc1 = 0

        for num in list_a:
            loss1 = generator.train_on_batch(x_train[num],label1[num])[0]
            
        if e % 10 == 0:
            generator.save(model_save_dir + 'gen_model%d.h5' % e)

batch_size=8
epochs=100
model_save_dir='./model/'
output_dir='./output/'
ext='.png'
n_class = 28
chan = 3
train(epochs, batch_size, output_dir, model_save_dir,ext,n_class,chan)
