import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import sys
import glob
import pandas as pd
sys.path.append('../..')
# import Work.functions.functions
import functions.functions as fn
import BFMatcher.urls as url

algo_start_time = time.time()
transformed = 0

for test_image_url in glob.iglob(url.get_test_image_url()):
    transformed = transformed + 1
    test_image = cv2.imread(test_image_url, 0)
    test_image_name = test_image_url.rsplit('/', 1)[1]
    test_image_name = test_image_name.rsplit('.', 1)[0]
    
    akaze = cv2.AKAZE_create()
    algo_name = 'akaze_'
    # find the keypoints and descriptors with akaze
    kp1, des1 = akaze.detectAndCompute(test_image, None)
    # create BFMatcher object
    # bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    bf = cv2.BFMatcher()
    title = algo_name + 'match_for_ '
    fig = plt.figure()

    time_started = time.time()
    # images = fn.loadimages('../../../../real_images/*')
    desc_comp_time = []
    image_names = []
    matching_time = []
    percentage_sim = []
    match_distance = []
    original_image =[]
    minimum_distance = sys.maxsize
    print ("sorting matches...")
    j = 0

    for image_f in glob.iglob(url.get_original_urls()):
            # print path_to_images
        img = cv2.imread(image_f, 0)
            # image = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
        img_name = image_f.rsplit('/', 1)[1]
        img_name = img_name.rsplit('.', 1)[0]
        print(img_name)
        # Match descriptors.
        
        start_time = time.time()
        kp2, des2 = akaze.detectAndCompute(img, None)
        desc_comp_time.append(time.time() - start_time)
        print (str(time.time() - start_time) + "seconds")
        print("Title: " + img_name + "  is number  " + str(j+1) + "  :::: and for transformed image ("+test_image_name+") number ::: " + str(transformed))
        extr_match = time.time()
        matches = bf.match(des1, des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        good_match=[]
        total_dis = 0
        dist_array = []
       
        for i in range(0,len(matches)):
            p1 = matches[i].distance
            good_match.append(p1)

        if(sum(good_match) <= minimum_distance):
            minimum_distance = sum(good_match)
            original_image = good_match
            matche_img_name = img_name
            orig_X = np.arange(len(original_image))
        
        matching_time.append(time.time() - extr_match)
        image_names.append(img_name)
        x = np.arange(len(good_match))
        # match_distance.append(good_match)

        plt.plot(x, good_match)
        # plt.plot(x, original_image, label='')

        plt.xticks(rotation=-40)
        j = j + 1
        
    zipper = zip(image_names,matching_time,desc_comp_time)
    fn.save_descriptors_to_file('../../database/match/'+algo_name+'_match_'+test_image_name+'.csv',zipper)
    print("finished after:  " + str(time.time()-time_started) + "   :seconds")
    # plt.title("compared to " + img_url)
    plt.plot(orig_X, original_image, 'b',label=matche_img_name)
    plt.xlabel('number of descriptors')
    plt.ylabel('distance')
    plt.legend()
 
    fig.savefig(title+test_image_name)
print ("--------------------------------------END-------------------------------------------")
print("The total execution time for akaze is :  %s seconds" % (time.time() - algo_start_time)) 
