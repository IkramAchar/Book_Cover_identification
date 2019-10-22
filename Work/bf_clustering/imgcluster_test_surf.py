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
import imgcluster
from matplotlib import pyplot as plt
import save_cluster as clusters

DIR_NAME = '../original_images/'


# Demo for clustering a set of 20 images using 'imgcluster' module. ../images/train/ ../images/train/
# To be executed in the standalone mode by default. IP[y] Notebook requires some minor adjustments.

""" True (reference) labels for the provided images - defined manually according to the semantic
    meaning of images. For example: bear, raccoon and fox should belong to the same cluster.
    Please feel free to change the true labels according to your perception of these images  :-)
"""
TRUE_LABELS = [0, 1, 2, 1, 0, 1, 3, 3, 3, 3, 3, 1, 0, 2, 2, 1, 2, 0, 2, 2]

# if __name__ == "__main__":
#     matrix = imgcluster.build_similarity_matrix(DIR_NAME, algorithm='SIFT')
#     # clusters.save_matrix(matrix,'ap_matrix.pkl')
#     c = imgcluster.do_cluster(DIR_NAME, algorithm='SIFT', print_metrics=True, labels_true=None)
#     print(c)
#     clusters.save_clusters(c,'ap_cluster_obj.pkl')

#     num_clusters = len(set(c))
#     images = os.listdir(DIR_NAME)
#     print(type(images))
#     print(images)
   
#     for n in range(num_clusters):
#         print("\n --- Images from cluster #%d ---" % n)

#         for i in np.argwhere(c == n):
#             if i != -1:
                
#                 # print("Image %s" % images[i])
#                 img = cv2.imread('%s/%s' % (DIR_NAME, images[i]))
#                 plt.axis('off')
#                 plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# #                 plt.show()


def get_percent_similarity(img):
    kp_2, desc_2 = sift.detectAndCompute(img, None)

    matches = bf.knnMatch(desc_1, desc_2, k=2)
    good_points = []
    
    for m, p in matches:
        if m.distance < 0.7*p.distance:
            good_points.append(m)
    number_keypoints = 0
    if len(kp_1) <= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_1)
    # total_time = time.time() - start_time
    number_keypoints = max(len(desc_1),len(desc_2))
    percentage_similarity = float(len(good_points)) / number_keypoints * 100

    return percentage_similarity

img1 = cv2.imread('../original_images/400_orig.jpeg')
# sift = cv2.xfeatures2d.SIFT_create()
sift = cv2.ORB_create()
bf = cv2.BFMatcher()
kp_1, desc_1 = sift.detectAndCompute(img1, None)
c = clusters.read_clusters('akaze_cluster_obj.pkl')
center_index = clusters.read_clusters('akaze_centers.pkl')
# i, j = zip(*c)
# clusters, center_index = [list(tup) for tup in zip(*c)]

print(center_index)
# exit(c)

num_clusters = len(set(c))
# num_clusters = len(set(center_index))
images = os.listdir(DIR_NAME)
print(type(images))
print(c)
# print(num_clusters)

max_per = 0

# for k in range(num_clusters):
#        class_members = c == k
#        name=[]
#        for i in np.argwhere(c == k):
#            j = i[0]
#            name.append(j)
#        cluster_center = c[center_index[k]]
#        print(cluster_center)
#        print("Image %s" % images[cluster_center])
#        print("something funny")
#        print(name)

# exit()
for n in range(num_clusters):
        print("\n --- Images from cluster #%d ---" % n)
        # print(np.argwhere(c==n))
        # print('biasi')

        name = "{}_{}".format("cluster_num", n)
        name = []
        cluster_center = c[center_index[n]]
        print("Image at center %s" % images[center_index[n]])
        for i in np.argwhere(c == n):
            # print(j)
            j = i[0]
            name.append(j)
            # print(" %s" % images[j])
            
            # print(j)
       
        
        img = cv2.imread('%s/%s' % (DIR_NAME, images[center_index[n]]))
        percentage_similarity = get_percent_similarity(img)
        print(percentage_similarity)
        if percentage_similarity == 100:
            print("Image is the one %s" % images[center_index[n]])
            exit()
        
        if max_per <= percentage_similarity:
            max_per = max(max_per, percentage_similarity)
            array_max = name
            cluster_num = n
        

        # if n == 1:
        #     break
            # if i != -1:
                # print("Image %s" % images[j])
                # img = cv2.imread('%s/%s' % (DIR_NAME, images[j]))
                # plt.axis('off')
                # # j = j +1
                # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # plt.show()
max_per = 0 
print(name)
print("-------------")
print(array_max)
print("Image matching for cluster %d",cluster_num)
for p in range(len(array_max)):
    # print(array_max[p])
    print("Image %s" % images[array_max[p]])
    img = cv2.imread('%s/%s' % (DIR_NAME, images[array_max[p]]))
    percentage_similarity = get_percent_similarity(img)
    if max_per < percentage_similarity:
        max_per = max(max_per, percentage_similarity)
        real_image = images[array_max[p]]
    # print(n)


print("we think the match for ::::" + real_image)
