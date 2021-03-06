import cv2

'''
    @Author: Mahir Gulzar
'''

'''
    Starting point for abstraction
    Reference: https://stackoverflow.com/questions/8687885/python-opencv-svm-implementation
'''
class Model(object):

    def load(self, fn):
        self.classifier_model.load(fn)
    def save(self, fn):
        self.classifier_model.save(fn)


'''
    Using OpenCV Support Vector Machine as our base classifier.
'''
class Classifier(Model):

    def __init__(self):
        C = 10
        gamma = 0.5
        self.classifier_model=cv2.ml.SVM_create()
        self.classifier_model.setKernel(cv2.ml.SVM_RBF)
        self.classifier_model.setGamma(gamma)
        self.classifier_model.setType(cv2.ml.SVM_C_SVC)
        self.classifier_model.setC(C)

    def train(self, train_samples, feedback):

        self.classifier_model.train(train_samples,cv2.ml.ROW_SAMPLE,feedback)

    def load_trained(self, filepath):
        C = 10
        gamma = 0.5
        self.classifier_model = cv2.ml.SVM_load(filepath)
        self.classifier_model.setKernel(cv2.ml.SVM_RBF)
        self.classifier_model.setGamma(gamma)
        self.classifier_model.setType(cv2.ml.SVM_C_SVC)
        self.classifier_model.setC(C)


    def predict(self, test_samples):
        return self.classifier_model.predict(test_samples)[1].ravel()