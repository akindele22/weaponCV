import pandas as pd 
import os
import numpy as np 
import cv2
from keras.preprocessing import image 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from skimage.segmentation import mark_boundaries 

def get_image_value(path, dim): 
    '''This function will read an image and convert to a specified version and resize depending on which algorithm is being used. '''
    img = image.load_img(path, target_size = dim)
    img = image.img_to_array(img)
    return img/255

def get_img_array(img_paths, dim): 
    '''This fucntion takes a list of image paths and returns the np array corresponding to each image.  It also takes the dim and whether edge is specified in order to pass it to another function to apply these parameters.  This function uses get_image_value to perform these operations'''
    final_array = []
    from tqdm import tqdm
    for path in tqdm(img_paths):
        img = get_image_value(path, dim)
        final_array.append(img)
    final_array = np.array(final_array)  
    return final_array

def get_tts():
    '''This function will create a train test split'''  

    DIM =  (150,150) 
    np.random.seed(10)        
    pistol_paths = [f'../Separated/FinalImages/Pistol/{i}' for i in os.listdir('../Separated/FinalImages/Pistol')] 
    pistol_labels = [1 for i in range(len(pistol_paths))]
    rifle_paths = [f'../Separated/FinalImages/Rifle/{i}' for i in os.listdir('../Separated/FinalImages/Rifle')] 
    rifle_labels = [2 for i in range(len(rifle_paths))]    
    neg_paths = [f'../Separated/FinalImages/NoWeapon/{i}' for i in os.listdir('../Separated/FinalImages/NoWeapon')]
    np.random.shuffle(neg_paths)
    neg_paths = neg_paths[:len(pistol_paths)- 500]
    neg_labels = [0 for i in range(len(neg_paths))]

    np.random.shuffle(pistol_paths)
    pistol_paths = pistol_paths[:len(rifle_paths)+150]
    neg_paths = neg_paths[:len(rifle_paths)+150]

    pistol_labels = [1 for i in range(len(pistol_paths))]
    knife_labels = [2 for i in range(len(rifle_paths))]
    
    paths = pistol_paths + rifle_paths + neg_paths
    labels = pistol_labels + rifle_labels + neg_labels
    x_train, x_test, y_train, y_test = train_test_split(paths, labels, stratify = labels, train_size = .90, random_state = 10)

    new_x_train = get_img_array(x_train, DIM)
    new_x_test = get_img_array(x_test, DIM)
    
    print('Train Value Counts')
    print(pd.Series(y_train).value_counts())
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Test Value Counts')
    print(pd.Series(y_test).value_counts())
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('X Train Shape')
    print(new_x_train.shape)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('X Test Shape')
    print(new_x_test.shape)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_test = to_categorical(y_test)
    y_train = to_categorical(y_train)
    tts = (new_x_train, new_x_test, y_train, y_test)
    return tts

x_train, x_test, y_train, y_test = get_tts()

#uncomment the code below to see what the images look like
#cv2.imshow('test', x_train[25])
# cv2.waitKey(0)
# cv2.destroyAllWindows()