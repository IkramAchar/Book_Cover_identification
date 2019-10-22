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
import sift_flann as imgcluster
from matplotlib import pyplot as plt
import save_cluster as clusters

DIR_NAME = '../original_images/'

# Demo for clustering a set of 20 images using 'imgcluster' module.
# To be executed in the standalone mode by default. IP[y] Notebook requires some minor adjustments.

""" True (reference) labels for the provided images - defined manually according to the semantic
    meaning of images. For example: bear, raccoon and fox should belong to the same cluster.
    Please feel free to change the true labels according to your perception of these images  :-)
"""
TRUE_LABELS = [0, 1, 2, 1, 0, 1, 3, 3, 3, 3, 3, 1, 0, 2, 2, 1, 2, 0, 2, 2]

if __name__ == "__main__":
    # matrix = imgcluster.build_similarity_matrix(DIR_NAME, algorithm='SIFT')
    # clusters.save_matrix(matrix,'ap_matrix.pkl')
    cluster_labels , center_indexs = imgcluster.do_cluster(DIR_NAME, algorithm='SIFT', print_metrics=True, labels_true=None)
    
    clusters.save_clusters(cluster_labels,'sift_cluster_flann.pkl')
    clusters.save_clusters(center_indexs,'sift_centers_flann.pkl')

    num_clusters = len(set(cluster_labels))
    images = os.listdir(DIR_NAME)
    print(type(images))
    print(images)
    # colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k in range(num_clusters):
        class_members = cluster_labels == k
        name=[]
        for i in np.argwhere(cluster_labels == k):
            j = i[0]
            name.append(j)
        cluster_center = cluster_labels[center_indexs[k]]
        print(cluster_center)
        print("Image %s" % images[center_indexs[k]])
        print("something funny")
        print(name)
        # cluster_center = X[cluster_centers_indices[k]]


    for n in range(num_clusters):
        print("\n --- Images from cluster #%d ---" % n)
        

        for i in np.argwhere(c == n):
            # if i != -1:
            j = i[0]
#           name.append(j)
#           print("Image %s" % images[j])
                
            print("Image %s" % images[j])
                # img = cv2.imread('%s/%s' % (DIR_NAME, images[i]))
                # plt.axis('off')
# #                 plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# # #                 plt.show()


# img1 = cv2.imread('../Work/original_images/adam_orig.jpeg')
# sift = cv2.ORB_create()
# bf = cv2.BFMatcher()
# kp_1, desc_1 = sift.detectAndCompute(img1, None)
# c = clusters.read_clusters('orb_cluster_obj.pkl')

# num_clusters = len(set(c))
# images = os.listdir(DIR_NAME)
# print(type(images))
# print(c)
# # print(images)


# for n in range(num_clusters):
#         print("\n --- Images from cluster #%d ---" % n)
#         # print(np.argwhere(c==n))
#         # print('biasi')

#         name = "{}_{}".format("cluster_num", n)
#         name = []
        
#         for i in np.argwhere(c == n):
#             # print(j)
#             j = i[0]
#             name.append(j)
#             print("Image %s" % images[j])
            
#             # print(j)
#         print(name[0])
#         # print("Image %s" % images[n])
#         # img = cv2.imread('%s/%s' % (DIR_NAME, images[name[0]]))
#         # kp_2, desc_2 = sift.detectAndCompute(img, None)

#         # matches = bf.knnMatch(desc_1, desc_2, k=2)
#         # good_points = []
        
#         # for m, n in matches:
#         #     if m.distance < 0.7*n.distance:
#         #         good_points.append(m)
#         # number_keypoints = 0
#         # if len(kp_1) <= len(kp_2):
#         #     number_keypoints = len(kp_1)
#         # else:
#         #     number_keypoints = len(kp_1)
#         # # total_time = time.time() - start_time
#         # number_keypoints = max(len(desc_1),len(desc_2))
#         # percentage_similarity = float(len(good_points)) / number_keypoints * 100
#         # print(percentage_similarity)
#             # if i != -1:
#                 # print("Image %s" % images[j])
#                 # img = cv2.imread('%s/%s' % (DIR_NAME, images[j]))
#                 # plt.axis('off')
#                 # # j = j +1
#                 # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#                 # plt.show()
        
# print(name)

