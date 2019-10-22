import cv2 as cv
import numpy as np
import glob
import time
import sys
import winsound
sys.path.append('./..')
import functions.functions as fn

algorithm = cv.AKAZE_create()

keypoints = []
image = []
compute_time_arry = []
all_images = fn.loadimages("../real_images/*_orig.*")

for image_to_compare, title in all_images:

    start_time= time.time()
    kp_2, desc_2 = algorithm.detectAndCompute(image_to_compare, None)
    total_time = time.time()- start_time
    print("title: "+ title)
    print("nombre de keypoints: %d " %len(kp_2))
    print("time : %s" %(time.time()- start_time))
    image.append(title)
    keypoints.append(len(kp_2))
    compute_time_arry.append(total_time)

zipped = zip(image,keypoints,compute_time_arry)
fn.save_stats_to_file('getting_kp_akaze.csv',zipped)

winsound.MessageBeep()





