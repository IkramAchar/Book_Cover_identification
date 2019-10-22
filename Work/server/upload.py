from flask import Flask, render_template, request,jsonify
from werkzeug import secure_filename
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import save_cluster as clusters
import time
import pickle
DIR_NAME = '../Work/original_images/'


app = Flask(__name__)
c = clusters.read_clusters('orb_cluster_match.pkl')
center_index = clusters.read_clusters('orb_centers_match.pkl')
image_descriptors = open('../database/orb_descriptors_dic.pkl', 'rb')
desc_dict = pickle.load(image_descriptors)
sift = cv2.ORB_create()
bf = cv2.BFMatcher()

num_clusters = len(set(c))
images = os.listdir(DIR_NAME)

@app.route('/upload', methods=['POST'])
def upload_file():
    start_time = time.time()
    print (request.files)
    # checking if the file is present or not.
    if 'file' not in request.files:
        return "No file found"
    f = request.files['file']
    # f.save(secure_filename(f.filename))
    file = request.files['file']
    # file.filename = "wiga.jpg"
    file.save(file.filename)
    # get zipped object
    
    print(type(images))
    print(c)

    image = cv2.imread(file.filename)
    kp , desc_1 = sift.detectAndCompute(image, None)
    # return "file successfully saved"

    array_max, cluster_num = get_image_cluster(num_clusters,center_index,desc_1)
    max_per = 0 
    print("THE TYPE IS", type(array_max))
    if isinstance(array_max, list):

        print("Image matching for cluster %d",cluster_num)
        for p in range(len(array_max)):

            print("Image %s" % images[array_max[p]])
            image_desc = desc_dict[images[array_max[p]]]
            # image_desc = cv2.imread('%s/%s' % (DIR_NAME, images[array_max[p]]))
            percentage_similarity = get_percent_similarity(desc_1,image_desc)
            if max_per < percentage_similarity:
                max_per = max(max_per, percentage_similarity)
                real_image = images[array_max[p]]
                
    else:
        print("we are in")
        real_image = array_max
        max_per = 100
    
    print("we think the match is ::::" + real_image)
    print("le temps maximum est ", (time.time() - start_time))


    return jsonify({'message': 'its okay','filename': file.filename,'image name':real_image,'similarity percentage': max_per, 'cluster':cluster_num})

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
            name = "{}_{}".format("cluster_num", n)
            name = []
            cluster_center = c[center_index[n]]
            print("Image at center %s" % images[center_index[n]])
            for i in np.argwhere(c == n):
                j = i[0]
                name.append(j)
        
            image_desc = desc_dict[images[center_index[n]]]
            percentage_similarity = get_percent_similarity(desc_1,image_desc)
            print(percentage_similarity)
            

            if percentage_similarity == 100:
                print("Image is the one %s" % images[center_index[n]])
                return images[center_index[n]], n
                exit()
            
            if max_per <= percentage_similarity:
                max_per = max(max_per, percentage_similarity)
                array_max = name
                cluster_num = n
    return array_max, cluster_num


if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=5005)
    app.run(debug = True)
   
