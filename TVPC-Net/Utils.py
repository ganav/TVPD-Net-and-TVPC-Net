
import tensorflow as tf
#from skimage import data, io, filters
import numpy as np
from numpy import array
from numpy.random import randint
#from scipy.misc import imresize
from scipy import ndimage
import os,cv2,csv,sys, glob,math
import matplotlib.pyplot as plt
import tensorflow.keras as Ker
from tensorflow.keras.preprocessing.image import img_to_array
# Display
from IPython.display import Image, display
import matplotlib.cm as cm

plt.switch_backend('agg')


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
    return (input_data / 255.).astype(np.float32)#(input_data.astype(np.float32) - 127.5)/127.5

    
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)
    
def load_data_from_dirs(path,dirs, ext, n_class):
    files1,files2 = [],[]

    path2 = path +'\\' + dirs + '\\'
    class_folders = glob.glob(path2+ '/*')

    for fold in range(len(class_folders)):
        class_files = glob.glob(class_folders[fold]+ '/*'+ext)
        #print(path2+ '/*'+ext)
        #print(class_files)

        for img in class_files:

            imgs_A = cv2.imread(img,-1)# cv2.IMREAD_COLOR  cv2.IMREAD_GRAYSCALE , cv2.IMREAD_UNCHANGED, 1, 0 or -1
            #imgs_A = cv2.resize(imgs_A, (200, 200), interpolation = cv2.INTER_AREA)
            imgs_A = imgs_A[np.newaxis,...]#add 1 more dimension
            #imgs_A = np.moveaxis(imgs_A, 0, -1)#reorder the shape of dimension
            
            indx = tf.one_hot(fold, n_class)
            indx = indx[np.newaxis,...]
            indx = np.array(indx)

            files1.append(imgs_A)
            files2.append(indx)

    files1 = normalize(array(files1))
    print(len(files1))
    print(len(files2))

    return np.array(files1),np.array(files2)     
    
def load_training_data(path,directory, ext,n_class):
    x_train,labels = load_data_from_dirs(path,directory, ext, n_class)
    return x_train,labels
