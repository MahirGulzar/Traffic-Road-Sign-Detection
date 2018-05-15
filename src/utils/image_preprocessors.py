import imutils
import cv2
import numpy as np
import math

from src.utils.helper_functions import SIGNS



'''
    A utility class responsible for following:
    
    Model input image preparation:
        1- Adjust image contrast
        2- Laplacian of Gaussian
        3- Binarization
        4- Contour detection and filtering by threshold
        5- Sign extraction
        
    Results:
        6- Annotating extracted sign by calling retrieving prediction results
    
'''

class Image_Preprocessor:

    def __init__(self):
        pass

    '''
        Takes an image adjust image contrasts
        :returns equalized hist image
    '''
    def adjust_contrast(self,image):
        # Convert image into grayscale
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        # Split image channels
        channels = cv2.split(grayscale)
        # Equalize hist on channel zero
        channels[0] = cv2.equalizeHist(channels[0])
        # Merge back the equalized channels to image
        equalized_hist = cv2.merge(channels)
        equalized_hist = cv2.cvtColor(equalized_hist, cv2.COLOR_YCrCb2BGR)
        return equalized_hist

    '''
        Applying laplacian of gaussian to 
        detect edges without noise.
        
        :returns LoG image
    '''
    def LoG(self,image):

        # Gaussian Filter
        # smooths out the noise in laplacian
        LoG_image = cv2.GaussianBlur(image, (3, 3), 0)

        gray = cv2.cvtColor(LoG_image, cv2.COLOR_BGR2GRAY)
        # Laplacian to detect edges
        LoG_image = cv2.Laplacian(gray, cv2.CV_8U, 3, 3, 2)  # parameter
        LoG_image = cv2.convertScaleAbs(LoG_image)

        return LoG_image

    '''
        Prepropesses Image i.e 
        1- Adjust contrasts
        2- Laplacian of Gaussian
        3- Thresholding
    '''
    def adjust_image(self,image):
        image = self.adjust_contrast(image)
        image = self.LoG(image)
        # Applying threshold for binarization
        image = cv2.threshold(image, 32, 255, cv2.THRESH_BINARY)[1]

        return image


    '''
        Keep components above given thresh value
    '''
    def filter_connected_components(self,image, threshold):

        # cv2.imshow('connected componnents',image)
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

        print(nlabels)
        print(labels)
        print(stats)
        print(centroids)
        sizes = stats[1:, -1];
        nlabels = nlabels - 1

        filtered_img = np.zeros((labels.shape), dtype=np.uint8)

        for i in range(0, nlabels):
            if sizes[i] >= threshold:
                filtered_img[labels == i + 1] = 255
        return filtered_img


    '''
        Retrieve contours in the image
    '''
    def findContour(self,image):
        contour = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = contour[0] if imutils.is_cv2() else contour[1]

        return contour


    '''
        Compute moments of the contour finds centroid
        and check if contour signatures fits the given threshold
        if < threshold then sign else not. 
    '''
    def if_is_sign(self,given_contour, centroid, threshold):

        computed = []
        for c in given_contour:
            c = c[0]
            distance = math.sqrt((c[0] - centroid[0]) ** 2 + (c[1] - centroid[1]) ** 2)
            computed.append(distance)

        max_value = max(computed)
        signature = [float(dist) / max_value for dist in computed]


        temp = sum((1 - s) for s in signature)
        temp = temp / len(signature)

        if temp < threshold:
            return True, max_value + 2
        else:
            return False, max_value + 2


    '''
        Crop the sign on the given coordinates of the image
    '''
    def extract_sign(self,image, coordinate):

        width = image.shape[1]
        height = image.shape[0]
        top = max([int(coordinate[0][1]), 0])
        bottom = min([int(coordinate[1][1]), height - 1])
        left = max([int(coordinate[0][0]), 0])
        right = min([int(coordinate[1][0]), width - 1])
        return image[top:bottom, left:right]


    '''
        Find contour with withing distance threshold and returns 
        the largest sign
    '''
    def get_max_sign(self,image, contours, threshold, distance_theshold):

        max_distance = 0
        coordinate = None
        sign = None

        for c in contours:

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            is_sign, distance = self.if_is_sign(c, [cX, cY], 1 - threshold)

            # if the given contour was a sign check distance then previous max_distance


            if is_sign and distance > max_distance and distance > distance_theshold:
                max_distance = distance
                coordinate = np.reshape(c, [-1, 2])
                left, top = np.amin(coordinate, axis=0)
                right, bottom = np.amax(coordinate, axis=0)

                # get coordinate of the current max_distance contour
                coordinate = [(left - 2, top - 2), (right + 3, bottom + 1)]
                sign = self.extract_sign(image, coordinate)

        return sign, coordinate


    '''
        Generate a color mask with bitwise_or
    '''
    def exclude_colors(self,img):

        # frame = cv2.GaussianBlur(img, (3, 3), 0)
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # blue masks
        lower_blue = np.array([100, 128, 0])
        upper_blue = np.array([215, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # white mask
        lower_white = np.array([0, 0, 128], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # black mask
        lower_black = np.array([0, 0, 0], dtype=np.uint8)
        upper_black = np.array([170, 150, 50], dtype=np.uint8)
        mask_black = cv2.inRange(hsv, lower_black, upper_black)

        mask0 = cv2.bitwise_or(mask_blue, mask_white)
        mask = cv2.bitwise_or(mask0, mask_black)

        return mask


    def analyze(self,image, min_size_components, circle_thresh, model, count, current_sign_type,provider):

        # global provider

        original_image = image.copy()

        binary_image = self.adjust_image(image)

        binary_image = self.filter_connected_components(binary_image, min_size_components)

        # cv2.imshow('before bitwise',binary_image)

        # Applying bitwise and tone down further noise
        binary_image = cv2.bitwise_and(binary_image, binary_image, mask=self.exclude_colors(image))

        # cv2.imshow('Binary Image', binary_image)
        contours = self.findContour(binary_image)

        print(contours)

        sign, coordinate = self.get_max_sign(original_image, contours, circle_thresh, 15)

        text = ""
        sign_type = -1


        # Get predicted result by passing the cropped sign to our trained model here

        if sign is not None:
            sign_type = provider.predict_label(model, sign)
            sign_type = sign_type if sign_type <= 8 else 8
            # Annotate text with the predicted sign type label
            text = SIGNS[sign_type]
            cv2.imwrite('/output_dir/' + str(count) + '_' + text + '.png', sign)

        # create annotated text on frame with rect and sign type

        if sign_type > 0 and sign_type != current_sign_type:
            cv2.rectangle(original_image, coordinate[0], coordinate[1], (0, 255, 0), 1)
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(original_image, text, (coordinate[0][0], coordinate[0][1] - 15), font, 1, (0, 0, 255), 2,
                        cv2.LINE_4)

        print("Detected Sign:"+str(sign_type))
        return coordinate, original_image, sign_type, text

