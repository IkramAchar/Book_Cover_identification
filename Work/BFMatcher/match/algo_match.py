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


transformed = 0
print(url.get_test_image_url())

for test_image_url in glob.iglob(url.get_test_image_url()):
    transformed = transformed + 1
    test_image = cv2.imread(test_image_url, 0)
    test_image_name = test_image_url.rsplit('/', 1)[1]
    test_image_name = test_image_name.rsplit('.', 1)[0]

    print(test_image_name)
    # exit()
    # Initiate ORB detector
    # orb = cv2.xfeatures2d.SURF_create(1000)
    orb = cv2.ORB_create()
    algo_name = 'surf_'
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(test_image, None)
    # create BFMatcher object
    # bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    bf = cv2.BFMatcher()
    title = algo_name + '_BFMatcher() Match for '
    fig = plt.figure()
    num_plots = 15
    # plt.cm.gist_ncar
    # colormap = plt.cm.gist_ncar
    # .set_prop_cycle
    # plt.gca().set_color_cycle([colormap(i)
    #                            for i in np.linspace(0, 0.78, num_plots)])

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
        kp2, des2 = orb.detectAndCompute(img, None)
        desc_comp_time.append(time.time() - start_time)
        print (str(time.time() - start_time) + "seconds")
        print("Title: " + img_name + "  is number  " + str(j+1) + "  :::: and for transformed image ("+test_image_name+") number ::: " + str(transformed))
        extr_match = time.time()
        matches = bf.match(des1, des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # print (type(matches[0]))
        # exit()
        # Draw first 10 matches.
        good_match=[]
        total_dis = 0
        dist_array = []
        # print "//////////-------///////////////----////////////////-----///////////"
        for i in range(0,len(matches)):
            p1 = matches[i].distance
            good_match.append(p1)

        if(sum(good_match) <= minimum_distance):
            minimum_distance = sum(good_match)
            original_image = good_match
            matche_img_name = img_name
            orig_X = np.arange(len(original_image))
        
        matching_time.append(time.time() - extr_match)
        x = np.arange(len(good_match))
        match_distance.append(good_match)
        # plt.close('all')
        # df = pd.DataFrame({"x":x, "y":original_image})
        # plt.plot(df.x, df.y, label=img_name)
        plt.plot(x, good_match)
        # plt.plot(x, original_image, label='')

        plt.xticks(rotation=-40)
        # df2 = df[df.y >= 100]

        # plt.plot(df.x,df.y, label="original")
        # plt.plot(df2.x,df2.y, label="filtered to y <= 15")
        # plt.autoscale()
        # plt.tight_layout()
        # ax = plt.gca() # grab the current axis
        # ax.set_xticks([1,50]) # choose which x locations to have ticks
        # ax.set_xticklabels([1,"key point",50]) #
        # plt.legend()
        j = j + 1

    #     percentage_sim.append(similarity)
        image_names.append(img_name)

    zipper = zip(image_names,matching_time,desc_comp_time)

    fn.save_descriptors_to_file('../../database/match/'+algo_name+'_match_'+test_image_name+'.csv',zipper)
    print("finished after:  " + str(time.time()-time_started) + "   :seconds")
    # plt.title("compared to " + img_url)
    plt.plot(orig_X, original_image, 'b',label=matche_img_name)
    plt.xlabel('number of descriptors')
    plt.ylabel('distance')
    plt.legend()
    # plt.cm.gist_ncar(np.random.random())
    # plt.show()

    fig.savefig(title+test_image_name)
    if(transformed == 3):
        exit()
    # figure.suptitle('test title', fontsize=20)
    # plt.xlabel('xlabel', fontsize=18)
    # plt.ylabel('ylabel', fontsize=16)
    # figure.savefig('test.jpg')
