
import Network, Utils,sys
from tensorflow.keras.layers import Lambda,GlobalAveragePooling2D,Dense,concatenate,Input,add,Reshape,LeakyReLU, PReLU,UpSampling2D,Conv2D, Conv2DTranspose,MaxPooling2D, Activation,Flatten,Lambda
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers import InputSpec, Layer
from tqdm import tqdm
import argparse,csv,time
from tensorflow.keras.applications.vgg16 import VGG16
#from losses import wasserstein_loss, perceptual_loss
import matplotlib.pyplot as plt

np.random.seed(10)
chan=1
image_shape = (300,300,chan)

def train(epochs, output_dir, model_save_dir,  ext):
    batch_size = 1

    val_X = Utils.load_training_data('G:\\projects\\paper 18\\source\\data\\visible dataset divided\\db2\\deblur_val\\', ext)
    train_Y = Utils.load_training_data('G:\\projects\\paper 18\\source\\data\\visible dataset divided\\db1\\deblur_orig\\', ext)
    train_X = Utils.load_training_data('G:\\projects\\paper 18\\source\\data\\visible dataset divided\\db1\\deblur_blur\\', ext)  

    batch_count = len(train_Y)
    #blur
    #train_X,train_Y = Utils.motion_blur(train_Set)
    
    generator = Network.generator(image_shape,chan)
    print('generator :')
    print(generator.summary())
    discriminator = Network.discriminator(image_shape)
    print('discriminator :')
    print(discriminator.summary())
    sys.exit()
 
    #optimizer = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)
    #generator.compile(loss='mse', optimizer=optimizer)

    d_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    d_on_g_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    discriminator.compile(optimizer=d_opt, loss="binary_crossentropy")
    loss = ["mse", "binary_crossentropy"]
    loss_weights = [100, 1]

    gan = Network.get_gan_network(discriminator, image_shape, generator)
    gan.compile(optimizer=d_on_g_opt, loss=loss, loss_weights=loss_weights)
    #gan.compile(loss=[vgg_loss, "binary_crossentropy"],loss_weights=[1., 1e-3],optimizer=optimizer)
    
    print('gan :')
    print(gan.summary())
    
    real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
    fake_data_Y = np.random.random_sample(batch_size)*0.2
    gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
    data_csv = []#save logs
    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        generator_loss = 0
        discriminator_loss = 0

        list_a = list(range(0,batch_count))
        #print(list_a[:10])
        np.random.shuffle(list_a)

        for rand_nums in tqdm(list_a):
            HQ_img = train_Y[rand_nums]
            blurry_img = train_X[rand_nums]

            generated_img = generator.predict(blurry_img)   
            
            discriminator.trainable = True
            
            d_loss_real = discriminator.train_on_batch(HQ_img, real_data_Y)
            d_loss_fake = discriminator.train_on_batch(generated_img, fake_data_Y)
            discriminator_loss = discriminator_loss + 0.5 * np.add(d_loss_fake, d_loss_real)
            
            discriminator.trainable = False
            generator_loss = generator_loss + gan.train_on_batch(blurry_img, [HQ_img,gan_Y])[0]
                             
        Utils.app_('losses',[generator_loss/batch_count,discriminator_loss/batch_count])

        if e % 10 == 0:
            generator.save(model_save_dir + 'model%d.h5' % e)

            #validate
            #val_X,val_Y = Utils.motion_blur(train_val)
            for i in range(len(val_X)):
                #val = Utils.normalize(val_X[i])
                res = generator.predict(val_X[i])
                res = Utils.denormalize(res)
                inp_ = Utils.denormalize(val_X[i])
                #out_ = Utils.denormalize(val_Y[i])

                plt.figure(figsize=(30, 10))
                plt.subplot(1, 2, 1)
                plt.imshow(inp_, interpolation='nearest')
                plt.title('input',fontsize = 33)
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(res, interpolation='nearest')
                plt.title('result',fontsize = 33)
                plt.axis('off')
                #plt.subplot(1, 3, 3)
                #plt.imshow(out_, interpolation='nearest')
                #plt.title('target',fontsize = 33)
                #plt.axis('off')
                plt.tight_layout()
                plt.savefig('./output/' + str(e) + '_' + str(i) +'.png')
                plt.close('all')

        #if e % 50 == 0:#since dataset was randomly blurred Blur again at each Xth epoch
            #train_X,train_Y = Utils.motion_blur(train_Set)

if __name__== "__main__":
                 
    epochs=200
    model_save_dir='./model/'
    output_dir='./output/'
    ext='.png'
    train(epochs, output_dir, model_save_dir, ext)