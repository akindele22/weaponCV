import matplotlib.pyplot as plt
import pandas as pd

from pylab import *

from skimage.feature import hog
from skimage import  exposure

import os
cwd = os.getcwd()
# giving file extension
ext = ('jpeg')

desktop = os.path.join(cwd, "PNEUMONIA")
files = os.listdir(desktop)

def delete_file(full_path):
     #import os
     if os.path.exists(full_path):
          os.remove(full_path)
     else:
          print("The file does not exist")





def read_image(full_path):
        full_path = os.path.join(desktop, f)
        print(full_path)
        var1 = full_path
        var2 = ".csv"
        var3 = var1 + var2
        image = imread(full_path)
        
def hog_fe(image):
     try:
          hog_image  = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True)
          DF = pd.DataFrame(hog_image)
          DF.to_csv(var3)



                    #print("x")
     except:
          print("delete the image")
          delete_file(full_path)
          print("try again")
     

     
     



     
for f in files:
     
     	if f.endswith(ext):

               full_path = os.path.join(desktop, f)
               print(full_path)
               var1 = full_path
               var2 = ".csv"
               var3 = var1 + var2
               image = imread(full_path)
               
               
               #print(full_path)
               #var1 = full_path
               #var2 = ".csv"
               #var3 = var1 + var2
               #image = imread(full_path)
               #hog_image  = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True)
     
               #fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    #cells_per_block=(1, 1), visualize=True, channel_axis=-1)
               DF = pd.DataFrame(hog_image)
               DF.to_csv(var3)

	
# Python code to illustrate
# working of try()
def divide(x, y):
    try:
        # Floor Division : Gives only Fractional Part as Answer
        result = x // y
        print("Yeah ! Your answer is :", result)
    except ZeroDivisionError:
        print("Sorry ! You are dividing by zero ")
 
# Look at parameters and note the working of Program
divide(3, 0)