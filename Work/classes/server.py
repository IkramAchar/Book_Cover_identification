import os
import cv2
import glob
import sys
import csv
import time
import os
import pickle as pickle
import datetime
import functions as fn


class Matching_Algorithm:

    def __init__(self, algorithm_to_use, image_url, bf_or_flann_matcher, get_train_images_url=""):
        algorithm = algorithm_to_use.upper()
        matcher_obj = bf_or_flann_matcher.upper()
        self.image = cv2.imread(image_url, cv2.COLOR_BGR2GRAY)
        print(self.image)
        # self.matcher_obj = cv2.BFMatcher()
        self.matcher_name = bf_or_flann_matcher
        self.train_images_url = get_train_images_url
        self.get_matcher(matcher_obj)
        self.create_algorithm_obj(algorithm)

    def create_algorithm_obj(self, algorithm):
        try:
            if algorithm == "SIFT":
                self.algorithm = cv2.SIFT_create()
            elif algorithm == "SURF":
                self.algorithm = cv2.xfeatures2d.SURF_create()
            elif algorithm == "ORB":
                self.algorithm = cv2.ORB_create()
            elif algorithm == "AKAZE":
                self.algorithm = cv2.AKAZE_create()
            elif algorithm == "KAZE":
                self.algorithm = cv2.KAZE_create()
            else:
                raise Exception("object not created")

            print (algorithm + " object successfully created")
        except Exception as error:
            print (repr(
                error) + "\nError on algorith_name, ENTER a valid algorithm name e.g ORB, SIFT etc")
            exit()

    def get_matcher(self, matcher_obj):
        try:
            if matcher_obj == "BF":
                self.matcher_obj = cv2.BFMatcher()
            elif matcher_obj == "FLANN":
                index_params = dict(algorithm=0, trees=5)
                search_params = dict()
                self.matcher_obj = cv2.FlannBasedMatcher(
                    index_params, search_params)
            else:
                raise Exception("Wrong matcher!")
            print ("You have chosen " + matcher_obj + "matcher")
        except Exception as error:
            print (repr(error) + "\n enter bf or flann")
            exit()

    def get_keypoint_and_desc(self):
       self.kp_1, self.desc_1 = self.algorithm.detectAndCompute(
           self.image, None)
       return self.kp_1, self.desc_1

    def loadimages(self, path_to_images=""):
        if path_to_images == "":
            if self.train_images_url != "":
               path_to_images = self.train_images_url
            else:
                path_to_images = raw_input(
                    "Please provide the url to your train images:")
        all_images_to_compare = []
        titles = []
        i = 1
        print("loading images \r")

        for f in glob.iglob(path_to_images):
            imag = cv2.imread(f)
            image = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
            titles.append(f.rsplit('/', 1)[1])
            print (i),
            sys.stdout.flush()
            all_images_to_compare.append(image)
            i += 1

        return zip(all_images_to_compare, titles)

    def compare_images(self, provided_distance=""):
        if provided_distance == "":
            d = 0.65
        else:
            d = provided_distance
        percent = []
        image = []
        compute_time_arry = []
        get_desc_time = []
        all_images_to_compare = self.loadimages()
        kp_1, self.desc_1 = self.get_keypoint_and_desc()
        print('comparing image descriptors for distance ' + str(d))

        for image_to_compare, title in all_images_to_compare:
            desc_extract_time = time.time()
#            print(image_to_compare)
            kp_2, desc_2 = self.algorithm.detectAndCompute(
                image_to_compare, None)
            get_desc_time.append(time.time() - desc_extract_time)

            start_time = time.time()
            matches = self.matcher_obj.knnMatch(self.desc_1, desc_2, k=2)

            good_points = []
            for m, n in matches:
                if m.distance < d*n.distance:
                    good_points.append(m)
            number_keypoints = 0
            if len(desc_2) <= len(self.desc_1):
                number_keypoints = len(desc_2)
            else:
                number_keypoints = len(self.desc_1)

            print("Title: " + title)
            percentage_similarity = float(
                len(good_points)) / number_keypoints * 100
            total_time = time.time() - start_time
            print("--- %s seconds ---" % (total_time))
            print("Similarity: " + str((percentage_similarity)) + " %\n")
            percent.append(str(percentage_similarity))
            image.append(title)
            compute_time_arry.append(total_time)

        self.zipped = zip(image, compute_time_arry, get_desc_time, percent)
        return self.zipped

    def save_descriptor_to_DB(self, file_name):
        print('computing descriptors in percentage(%) ...')
        array_des = []
        j = 0
        for image_to_compare, titles in self.loadimages():
                # get key points and descriptors
                kp_2, desc_2 = self.algorithm.detectAndCompute(
                    image_to_compare, None)
                # creating an array of decriptors
                array_des.append(desc_2)
                remaing = j/float(len(titles)) * 100
                print (int(remaing), '|'),
                j = j + 1
                sys.stdout.flush()
                print (titles + 'have fan')
        #creating a zipped object of image descriptors with their titles
        zipped_data = zip(array_des, titles)
        # creating a .pkl database
        # print zipped_data
        print('% \n saving to database...')
        output = open('../database/book_image_database_desc_' +
                      file_name + '.pkl', 'wb')
        # writing to a database
        pickle.dump(zipped_data, output)
        # output.close()

    def read_desc_from_DB(self, file_name):

        # open a database file
        pkl_file = open(
            '../database/book_image_database_desc_' + file_name + '.pkl', 'rb')
        # get zipped object
        print('reading descriptors from a file')
        zipped_obj = pickle.load(pkl_file)

        print(zipped_obj)
        pkl_file.close()

        percent = []

        image = []
        for image_descriptor, image_name in zipped_obj:
            start_time = time.time()
            kp_1, desc_1 = self.get_keypoint_and_desc()

            matches = self.matcher_obj.knnMatch(
                self.desc_1, image_descriptor, k=2)

            good_points = []
            for m, n in matches:
                if m.distance < 0.6*n.distance:
                    good_points.append(m)
            number_keypoints = 0
            if len(image_descriptor) <= len(desc_1):
                number_keypoints = len(image_descriptor)
            else:
                number_keypoints = len(desc_1)

            print("Title: " + image_name)
            percentage_similarity = float(
                len(good_points)) / number_keypoints * 100
            print("--- %s seconds ---" % (time.time() - start_time))
            print("Similarity: " + str((percentage_similarity)) + " %\n")
            percent.append(str(int(percentage_similarity)))
            image.append(image_name)

            # pprint.pprint(data1)
        zipped = sorted(zip(percent, image),
                        key=lambda pair: pair[0], reverse=True)
        # zipped = sorted(zipped, key = lambda x: x[0])

        print('writing results to a file...')
        with open('../database/Results_Comparison/' + file_name+'desc_comp_results.csv', 'a') as csvfile:
            fieldnames = ['similarity(%)', 'image name', 'time and date']

            # spamwriter = csv.writer(csvfile, delimiter=' ',
            #                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer = csv.DictWriter(
                csvfile, delimiter='\t', fieldnames=fieldnames)
            writer.writeheader()
            for percentage, name_image in zipped:
                writer.writerow(
                    {'similarity(%)': percentage + " %", 'image name': name_image, 'time and date': datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")})
                # writer.writerow({'first_name': 'Lovely', 'last_name': 'Spam'})
                # writer.writerow({'first_name': 'Wonderful', 'last_name': 'Spam'})
        print('finished writing to a file')
        print('Done!!!')

    def save_stats_to_file(self, file_name):
        zipped_file = self.compare_images()
        im_typ = 'image type'
        percent_sim = 'percentage similarity(%)'
        compute_time = 'computational time'
        desc_time = 'Time to fetch descriptors'
        file_name = '../database/' + file_name + \
            "_stats_"+self.matcher_name + "_matcher.csv"

        with open(file_name, 'a') as csvfile:
            fieldnames = [im_typ, compute_time, desc_time, percent_sim]

            writer = csv.DictWriter(
                csvfile, delimiter='\t', fieldnames=fieldnames)
            #to check if the header is already returned
            if csvfile.tell() == 0:
                writer.writeheader()
            for img_type, time_taken, descrp_time, per_sim in zipped_file:
                writer.writerow(
                    {im_typ: img_type, compute_time: time_taken, desc_time: descrp_time, percent_sim: per_sim})
        print('finished saving  to file')
        print('Done!!!')


get_time = time.time()
m = Matching_Algorithm('orb', "../real_images/bird_orig.jpeg",
                       "bf", "../real_images/*")
#m.get_keypoint_and_desc()
m.compare_images()
# @TODO ikram look at the syntax the way to initialise it
# m.save_descriptor_to_DB('biasi')
# m.read_desc_from_DB('biasi')
# Matching_Algorithm("algorithm_to_use", "image_url", "bf_or_flann_matcher", "get_train_images_url")
# m.save_stats_to_file("biasi")
# p = Matching_Algorithm.
# matcher = "bf"

# algo = ['sift', 'surf', 'akaze','orb']
# algo = ['sift', 'surf']
# algo = ['sift']


# for algo_name in algo:

#     sift = Matching_Algorithm(algo_name, "../images/train/arabic.jpg", matcher, "../images/testBooks/arabic/*")
#     # sift = Matching_Algorithm(algo_name, "../images/train/butterfly.jpg", matcher, "../images/testBooks/butterfly/*")
#     # sift = Matching_Algorithm(algo_name , "../images/train/condame.jpg", matcher, "../images/testBooks/condame/*")
#     # sift = Matching_Algorithm(algo_name, "../images/train/life.JPEG", matcher, "../images/testBooks/lives/*")
#     # sift = Matching_Algorithm(algo_name, "../images/train/likeblack.jpg", matcher, "../images/testBooks/likeblack/*")
#     # sift = Matching_Algorithm(algo_name, "../images/train/madam.jpg", matcher, "../images/testBooks/madam/*")
#     # sift = Matching_Algorithm(algo_name, "../images/train/malcomx.jpg", matcher, "../images/testBooks/malcomx/*")
#     # sift = Matching_Algorithm(algo_name, "../images/train/memory.jpg", matcher, "../images/testBooks/memory/*")
#     # sift = Matching_Algorithm(algo_name, "../images/train/prayer.jpg", matcher, "../images/testBooks/prayer/*")
#     # sift = Matching_Algorithm(algo_name, "../images/train/rose.jpg", matcher, "../images/testBooks/rose/*")
#     # sift = Matching_Algorithm(algo_name, "../images/train/sherlock.jpg", matcher, "../images/testBooks/sherlock/*")
#     # sift = Matching_Algorithm(algo_name, "../images/train/the_100.jpg", matcher, "../images/testBooks/100/*")
#     print algo_name
#     for i in range(0, 10):
#         print("--- %s iteration ---" % (i + 1))
#         sift.save_stats_to_file(algo_name)

os.system('afplay /System/Library/Sounds/Sosumi.aiff')
get_time = time.time() - get_time
# print "time taken is " + str(get_time)
print("time taken is:  %s seconds!" % (get_time))
# sift.loadimages("../images/test/original_book.jpg")
