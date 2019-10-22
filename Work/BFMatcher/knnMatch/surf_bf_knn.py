import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import sys
sys.path.append('../../')
from functions import functions as fn
import time
import BFMatcher.urls as url

algo_start_time = time.time()
transformed = 0
image_urls = url.get_test_image_url()
original_images_urls = url.get_original_urls()

for test_image_url in glob.iglob(image_urls):
    transformed = transformed + 1
    test_image = cv2.imread(test_image_url, 0)
    test_image_name = test_image_url.rsplit('/', 1)[1]
    test_image_name = test_image_name.rsplit('.', 1)[0]

    sift = cv2.xfeatures2d.SURF_create(1000)
    kp_1, desc_1 = sift.detectAndCompute(test_image, None)
    print ('surf started')

    bf = cv2.BFMatcher()
    title = 'surf_bf_knn_for_'
    fig = plt.figure()
    # Load all the images
    for p in np.arange(0.40, 0.45, 0.05):
        p = 0.62
        p = round(p,2)
        percent = []
        image_names = []
        compute_time_arry = []
        time_for_desc = []
        j = 0
        for image_f in glob.iglob(original_images_urls):
            image_to_compare = cv2.imread(image_f, 0)
            img_name = image_f.rsplit('/', 1)[1]
            # Match descriptors.
            begin_time = time.time()
            # 2) Check for similarities between the 2 images
            kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
            time_for_desc.append(time.time() - begin_time)
            print ("-----description----")
            j = j + 1
            start_time = time.time()
            matches = bf.knnMatch(desc_1, desc_2, k=2)
            good_points = []

            for m, n in matches:
                if m.distance < p*n.distance:
                    good_points.append(m)
            number_keypoints = 0
            if len(kp_1) <= len(kp_2):
                number_keypoints = len(kp_1)
            else:
                number_keypoints = len(kp_1)
            
            total_time = time.time() - start_time
            number_keypoints = max(len(desc_1),len(desc_2))
            percentage_similarity = float(len(good_points)) / number_keypoints * 100

            print("Title: " + img_name + "  is number  " + str(j) + "  :::: and p = " + str(p) + " for transformed image ("+test_image_name+") number ::: " + str(transformed))
            print("--- %s seconds ---" % (time.time() - start_time))
            print("Similarity: " + str(int(percentage_similarity)) + " % \n")

            image_names.append(img_name)
            percent.append(int(percentage_similarity))
            compute_time_arry.append(total_time)
            # print type(percent)
            plot_zip = sorted(zip(image_names, percent ,compute_time_arry,time_for_desc),key=lambda pair: pair[1], reverse=True)
            image_names, percent, compute_time_arry, time_for_desc = [list(tup) for tup in zip(*plot_zip)]

            save_zip = zip(image_names,percent, compute_time_arry,time_for_desc)
    
        fn.save_percentage_to_file('../../database/knnMatch/surf_bf_knn.csv', save_zip)
        X = image_names[:3]
        Y = percent[:3]
        plt.plot(X, Y, label=p)
        plt.legend()
        print (image_names[:3], percent[:3],p)
        print ("---------------------------------------------------------------------------------")
    # print("The total execution time for surf is :  %s seconds" % (time.time() - algo_start_time)) 
    plt.xlabel('images')
    # plt.xticks(rotation=30)
    plt.ylabel('percent similarity')   

    # plt.show()
    fig.savefig(title+test_image_name)
print ("---------------------------------------END------------------------------------------")
print("The total execution time for surf bf knn is :  %s seconds" % (time.time() - algo_start_time)) 


