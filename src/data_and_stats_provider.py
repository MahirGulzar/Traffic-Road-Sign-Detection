from os import listdir

import cv2
import numpy as np

from src.classifier import Classifier


'''
    @Author: Mahir Gulzar

    Provider class is responsible for handling misc operations like:
    
    1- Reading and cleaning dataset
    2- Finding, assembling and preparing features for training of our Classifier 
    3- Tweaking HOG parameters
    4- Training the model 
    5- Evaluating the trained model
    
    
    Acts as a data provider for classifier and returns predicted labels to
    program entry point.
'''
class Provider():
    def __init__(self):
        self.model= Classifier()
        self.class_number=13
        self.size=32


    '''
        Load dataset of eight different types of signs and
        mark their directory name as their labels
    '''
    def load_dataset(self):
        dataset = []
        labels = []
        for sign_type in range(self.class_number):
            sign_list = listdir("./dataset/{}".format(sign_type))
            for sign_file in sign_list:
                if '.png' in sign_file:
                    path = "./dataset/{}/{}".format(sign_type, sign_file)
                    print(path)
                    img = cv2.imread(path, 0)
                    img = cv2.resize(img, (self.size, self.size))
                    img = np.reshape(img, [self.size, self.size])
                    dataset.append(img)
                    labels.append(sign_type)
        return np.array(dataset), np.array(labels)

    def model_trainer(self):
        print('Loading data from data.png ... ')
        # Load data.
        data, labels = self.load_dataset()
        print(data.shape)

        print("\n--------------------------------------------------------")

        print('Shuffle data ... ')
        # Shuffle data
        rand = np.random.RandomState(10)
        shuffle = rand.permutation(len(data))
        data, labels = data[shuffle], labels[shuffle]

        print("\n--------------------------------------------------------")

        print('Deskew images ... ')
        data_balanced = list(map(self.balance_image, data))

        print("\n--------------------------------------------------------")

        print('Retrieving Hog ...')
        # HoG feature descriptor
        hog = self.hog()

        print("\n--------------------------------------------------------")

        print('Calculating HoG descriptor for every image ... ')
        hog_descriptors = []
        for img in data_balanced:
            hog_descriptors.append(hog.compute(img))
        hog_descriptors = np.squeeze(hog_descriptors)

        print("\n--------------------------------------------------------")

        print('Spliting data into training (90%) and test set (10%)... ')
        train_n = int(0.9 * len(hog_descriptors))
        # data_train, data_test = np.split(data_deskewed, [train_n])
        hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
        labels_train, labels_test = np.split(labels, [train_n])

        print("\n--------------------------------------------------------")

        print('Training Classifier ...')
        model = self.model
        model.train(hog_descriptors_train, labels_train)

        print("\n--------------------------------------------------------")

        model.save('trained_model.dat')

        print("\n--------------------------------------------------------")

        return model


    '''
          Deskew the images based on image centroids and wrap affine 
          to transform the image.
    '''

    def balance_image(self, img):

        m = cv2.moments(img)
        # print(m)
        if abs(m['mu02']) < 1e-2:
            return img.copy()
        skew = m['mu11'] / m['mu02']
        M = np.float32([[1, skew, -0.5 * self.size * skew], [0, 1, 0]])

        # transform skewed image
        img = cv2.warpAffine(img, M, (self.size, self.size), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        return img

    '''
        Return Histogram of Oriented Gradients for SVM to train upon.
        Reference : https://gurus.pyimagesearch.com/lesson-sample-histogram-of-oriented-gradients-and-car-logo-recognition/
    '''

    def hog(self, ):

        winSize = (20, 20)
        blockSize = (10, 10)
        blockStride = (5, 5)
        cellSize = (10, 10)
        nbins = 9
        derivAperture = 1
        winSigma = -1.
        histogramNormType = 0
        L2HysThreshold = 0.2
        gammaCorrection = 1
        nlevels = 64
        signedGradient = True

        return cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                 histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)

    def predict_label(self,model, data):

        gray_scale = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        img = [cv2.resize(gray_scale, (self.size, self.size))]

        img_balanced = list(map(self.balance_image, img))
        hog = self.hog()
        hog_descriptors = np.array([hog.compute(img_balanced[0])])
        hog_descriptors = np.reshape(hog_descriptors, [-1, hog_descriptors.shape[1]])
        return int(model.predict(hog_descriptors)[0])

