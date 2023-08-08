from __future__ import print_function, division
import scipy

import Network, Utils,sys
from tensorflow.keras.layers import Lambda,GlobalAveragePooling2D,Dense,concatenate,Input,add,Reshape,LeakyReLU, PReLU,UpSampling2D,Conv2D, Conv2DTranspose,MaxPooling2D, Activation,Flatten,Lambda
from tensorflow.keras.models import Model,load_model
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers import InputSpec, Layer
from tqdm import tqdm
import argparse,csv,time
import blurring,random
import matplotlib.pyplot as plt
import sys,glob,os, cv2,datetime
from layer_utils import ReflectionPadding2D, res_block

def get_img_parts(img_path):
	"""Given a full path to a video, return its parts."""
	parts = img_path.split('\\')
	#print(parts)
	leng=len(parts)
	filename = parts[leng-1]
	return  filename


pxl = 30
canvas = 100
params = 0.005# this param helps to define probability of big shake. Recommended expl = 0.005.
trajectory = blurring.Trajectory(canvas=canvas, max_len=pxl, expl=params).fit()#np.random.choice(params)
psf = blurring.PSF(canvas=canvas, trajectory=trajectory).fit()

#:param canvas: size of domain where our trajectory os defined.
#:param max_len: maximum length of our trajectory (size of movement).
#:param expl: this param helps to define probability of big shake. Recommended expl = 0.005. 

dir_B='G:\\projects\\paper 19\\source\\data\\images for paper openDB 4\\/*.png'
dir_C='G:\\projects\\paper 19\\source\\data\\images for paper openDB 4\\'

img_files = glob.glob(dir_B)

#model, model modified, model deblurGAN orig
model = load_model('./model db2/model190.h5', custom_objects={'ReflectionPadding2D': ReflectionPadding2D})
#converter = tf.lite.TFLiteConverter.from_keras_model(model)
#tflite_model = converter.convert()
#open(os.path.join(save_dir, 'full_generator_{}_{}.tflite'.format(epoch_number, current_loss)),
     #"wb").write(tflite_model)

#model=load_model('./model modified/model100.h5')
print(model.summary())

print(len(img_files))
for img_path in img_files:

	name = get_img_parts(img_path)
	blur_img = cv2.imread(img_path,1)# cv2.IMREAD_COLOR  cv2.IMREAD_GRAYSCALE , cv2.IMREAD_UNCHANGED, 1, 0 or -1
	print(blur_img.shape)
	blur_img = cv2.resize(blur_img, (300, 300), interpolation = cv2.INTER_AREA)
	print(blur_img.shape)
	#blur_img = cv2.imread(dir_B+name,-1)

	#imgs_A = blurring.BlurImage(or_img, PSFs=psf, part=np.random.choice([1, 2, 3])).\
				#blur_image(save=True)
	blur_img = Utils.normalize(blur_img)
	out = model.predict(blur_img)
	out = Utils.denormalize(out)
	'''if blur_img.shape[2]==3:#if grayscale image
		out = model.predict(imgs_A)
		out = Utils.denormalize(out)
	else:
		imgs_A = imgs_A[np.newaxis,...]#add 1 more dimension
		imgs_A = np.moveaxis(imgs_A, 0, -1)#reorder the shape of dimension
		start = time.time()
		out = model.predict(imgs_A)
		end = time.time()
		print(end - start)
		out = Utils.denormalize(out)
		out=np.squeeze(out, axis=-1)#remove previously added dimension'''
				
	os.makedirs(dir_C, exist_ok=True)
	print(dir_C + 'kk' + name)
		
	cv2.imwrite(dir_C + '__' + name, out)
print('end')
	#break






