import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import time
import sys
sys.path.append('..')
import functions.functions as fn
import urls as url

algo_start_time = time.time()
image_urls = url.get_test_image_url()
original_images_urls = url.get_original_urls()
transformed = 0

for test_image_url in glob.iglob(image_urls):
    transformed = transformed + 1
    test_image = cv2.imread(test_image_url, 0)
    test_image_name = test_image_url.rsplit('/', 1)[1]
    test_image_name = test_image_name.rsplit('.', 1)[0]

    sift = cv2.ORB_create()
    kp_1, desc_1 = sift.detectAndCompute(test_image, None)
    # kp_2, desc_2 = sift.detectAndCompute(img, None)

    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=12,  # 12
                        key_size=20,     # 20
                        multi_probe_level=2)  # 2

    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    title = 'orb_flann_in'
    fig = plt.figure()

    for p in np.arange(0.40, 0.45, 0.05):
        p=0.63
        p = round(p,2)
        percent = []
        image_names = []
        compute_time_arry = []
        time_for_desc = []
        j = 0
        # exit()
        # for image_to_compare, title in all_images_to_compare:
        for image_url in glob.iglob(original_images_urls):
            image_to_compare = cv2.imread(image_url, 0)
            img_name = image_url.rsplit('/', 1)[1]
            j= j + 1

            # 2) Check for similarities between the 2 images
            begin_time = time.time()
            kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
            time_for_desc.append(time.time() - begin_time)
            print ("-----description----")
            
            start_time = time.time()
            matches = flann.knnMatch(desc_1, desc_2, k=2)
            good_points = []
        
            for m_n in matches:
                    if len(m_n) != 2:
                        continue
                    (m,n) = m_n
                    if m.distance < p*n.distance:
                        good_points.append(m)

            total_time = time.time() - start_time
            number_keypoints = max(len(desc_1),len(desc_2))
            percentage_similarity = float(len(good_points)) / number_keypoints * 100
            
            print("Title: " + img_name + "  is number  " + str(j) + "  :::: and p = " + str(p) + " for transformed image ("+test_image_name+") number ::: " + str(transformed))
            print("Similarity: " + str(percentage_similarity) + "% \n")
            image_names.append(img_name)
            percent.append(int(percentage_similarity))
            compute_time_arry.append(total_time)
            # print type(percent)
            plot_zip = sorted(zip(image_names, percent ,compute_time_arry,time_for_desc),key=lambda pair: pair[1], reverse=True)
            # percent, image = (zip(*plot_zip))
            # image, percent, compute_time_arry, time_for_desc  = (zip(*plot_zip))
            image_names, percent, compute_time_arry, time_for_desc = [list(tup) for tup in zip(*plot_zip)]

            save_zip = zip(image_names,percent, compute_time_arry,time_for_desc)

            # print(save_zip)
            # list(percent)
            # print plot_zip
    
        fn.save_percentage_to_file('../database/flann/orb_flann.csv', save_zip)
        X = image_names[:3]
        Y = percent[:3]
        plt.plot(X, Y, label=p)
        plt.legend()
        print (image_names[:3], percent[:3],p)
        print ("---------------------------------------------------------------------------------")
    # print("The total execution time is :  %s seconds" % (time.time() - algo_start_time)) 
    plt.xlabel('images')
    # plt.xticks(rotation=30)
    plt.ylabel('percent similarity')   
    # plt.cm.gist_ncar(np.random.random())
    # plt.show()
    fig.savefig(title+test_image_name)
print ("------------------------------------END---------------------------------------------")
print("The total execution time is :  %s seconds" % (time.time() - algo_start_time)) 
