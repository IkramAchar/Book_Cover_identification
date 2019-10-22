# Copyright (c) 2016, Oleg Puzanov
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import os
import numpy as np
import cv2
# import imgcluster
from matplotlib import pyplot as plt
import save_cluster as clusters
import time
import pickle
import csv
import glob
DIR_NAME = '../original_images/'


c = clusters.read_clusters('orb_cluster_match.pkl')
center_index = clusters.read_clusters('orb_centers_match.pkl')
image_descriptors = open('../database/orb_descriptors_dic.pkl', 'rb')
# get zipped object
print('reading descriptors from a file')
desc_dict = pickle.load(image_descriptors)

num_clusters = len(set(c))
images = os.listdir(DIR_NAME)
print(type(images))
print(c)

def get_percent_similarity(desc_1,desc_2):

    matches = bf.match(desc_1, desc_2)
    matches = sorted(matches, key=lambda x: x.distance)
    percentage_similarity = 0

    for i in range(0,len(matches)):
        p1 = matches[i].distance
        if p1 <= 250:
            percentage_similarity += 1
    percentage_similarity = float(percentage_similarity / len(matches) * 100)

    return percentage_similarity


def get_image_cluster(num_clusters,center_index,desc_1):
    max_per = 0

    for n in range(num_clusters):
            print("\n --- Images from cluster #%d ---" % n)
            name = "{}_{}".format("cluster_num", n)
            name = []
            cluster_center = c[center_index[n]]
            print("Image at center %s" % images[center_index[n]])
            for i in np.argwhere(c == n):
                j = i[0]
                name.append(j)
        
            image_desc = desc_dict[images[center_index[n]]]
            print("for image::::::::",images[center_index[n]])
            print("--------------d$$$$$$$$$$$$$$$$$$$---------------")
            percentage_similarity = get_percent_similarity(desc_1,image_desc)
            print(percentage_similarity)

            # if percentage_similarity == 100:
            #     print("Image is the one %s" % images[center_index[n]])
            #     print("le temps maximum est ", (time.time() - start_time))
            #     with open('save_test.csv', 'a') as csvfile:
            #         fieldnames = ["im_typ", "sim_image"]
            #         writer = csv.DictWriter(csvfile, delimiter='\t', fieldnames=fieldnames)
            #         if csvfile.tell() == 0:
            #             writer.writeheader()
            #         writer.writerow({"im_typ": test_image,"sim_image": real_image})
            #     exit()
            
            if max_per < percentage_similarity:
                max_per =  percentage_similarity
                array_max = name
                cluster_num = n
    return array_max, cluster_num

start_time = time.time()
transformed = 0

for test_image_url in glob.iglob(clusters.get_test_image_url()):
    transformed = transformed + 1
    orb = cv2.ORB_create()
    img1 = cv2.imread(test_image_url,0)
    test_image = clusters.get_image_name(test_image_url)
    sift = cv2.ORB_create()
    bf = cv2.BFMatcher()
    kp_1, desc_1 = sift.detectAndCompute(img1, None)
    array_max, cluster_num = get_image_cluster(num_clusters,center_index,desc_1)
    max_per = 0 

    # print(name)
    print("-------------")
    print(array_max)
    print("Image matching for cluster %d",cluster_num)
    for p in range(len(array_max)):

        print("Image %s" % images[array_max[p]])
        image_desc = desc_dict[images[array_max[p]]]
        percentage_similarity = get_percent_similarity(desc_1,image_desc)
        if max_per < percentage_similarity:
            max_per = max(max_per, percentage_similarity)
            real_image = images[array_max[p]]
            
    print("we think the match is ::::" + real_image)
    print("le temps maximum est ", (time.time() - start_time))

    with open('orb_accuracy.csv', 'a') as csvfile:
            fieldnames = ["im_typ", "sim_image"]
            writer = csv.DictWriter(csvfile, delimiter='\t', fieldnames=fieldnames)
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerow({"im_typ": test_image,"sim_image": real_image})
