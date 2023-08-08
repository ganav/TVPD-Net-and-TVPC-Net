import tensorflow as tf
#from skimage import data, io, filters
import numpy as np
from numpy import array
from numpy.random import randint
from scipy import ndimage
import os,cv2,csv,sys, glob,math
import matplotlib.pyplot as plt
import tensorflow.keras as Ker
from tensorflow.keras.preprocessing.image import img_to_array
# Display
from IPython.display import Image, display
import matplotlib.cm as cm
import blurring

plt.switch_backend('agg')

pxl = 30
canvas = 100
params = 0.005# this param helps to define probability of big shake. Recommended expl = 0.005.

trajectory = blurring.Trajectory(canvas=canvas, max_len=pxl, expl=params).fit()#np.random.choice(params)
psf = blurring.PSF(canvas=canvas, trajectory=trajectory).fit()

def app_(name,data):
    with open(name+'.csv', 'a+', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(data)#remove the last candle which is not closed yet

        
def get_img_parts(img_path):
    """Given a full path to a video, return its parts."""
    parts = img_path.split('\\')

    leng=len(parts)
    filename = parts[leng-1]
    path=parts[0]+'\\'
 
    for i in range(1,leng-3):
        path=path+parts[i]+'\\'
    return path+'b\\'+parts[4]+'\\',  filename

def normalize(input_data):
    input_data = input_data[np.newaxis,...]
    return array((input_data / 255.).astype(np.float32))#(input_data.astype(np.float32) - 127.5)/127.5

def denormalize(input_data):
    input_data = (input_data * 255.).astype(np.uint8)

    input_data = np.squeeze(input_data, axis=0)#remove previously added dimension
    #b,g,r = cv2.split(input_data)
    #input_data=cv2.merge((r,g,b))
    return input_data

def load_training_data(input_dir, ext):

    files = []
    images = glob.glob(input_dir + '*' + ext)

    for img in images:
        imgs_A = cv2.imread(img,1)# cv2.IMREAD_COLOR  cv2.IMREAD_GRAYSCALE , cv2.IMREAD_UNCHANGED, 1, 0 or -1
        #imgs_A = cv2.resize(imgs_A, (300, 300), interpolation = cv2.INTER_AREA)
        #print(imgs_A.shape)
            
        #comment below 4 lines in case of color input
        #imgs_A = imgs_A[np.newaxis,...]#add 1 more dimension
        #imgs_A = imgs_A[np.newaxis,...]#add 1 more dimension
        #imgs_A = np.moveaxis(imgs_A, 0, -1)#reorder the shape of dimension

        files.append(normalize(imgs_A))#normalize

    print(len(files))

    return files#array(files)   

def motion_blur(train_Set):
    train_X,train_Y = [],[]
    for or_img in train_Set:
        bl_img = blurring.BlurImage(or_img, PSFs=psf, part=np.random.choice([1, 2, 3])).\
            blur_image(save=True)

        train_X.append(normalize(bl_img))
        train_Y.append(normalize(or_img))
        
    return train_X,train_Y
    
