import cv2
import numpy as np
from os import listdir



'''
    Provider class responsible for handling misc operations like:
    
    1- Reading and cleaning datasets
    2- Evaluating the trained model
    
    Acts as a data provider for classifier. 
'''


class Provider():
    def __init__(self):
        self.class_number=13
        self.size=32

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

