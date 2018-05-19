import argparse
import cv2
import numpy as np
import math
# from skvideo.io import VideoWriter

from src.utils.helper_functions import clear_output
from src.data_and_stats_provider import Provider
from src.utils.image_preprocessors import Image_Preprocessor

'''
    Traffic Sign Detector: Main entry point of the application
'''

provider = Provider()


def main(args):

    global provider

    # Clearing previous output
    clear_output()

    # Training the model
    trained_model=provider.model_trainer(args.load);

    # open input video
    video = cv2.VideoCapture(args.file)

    # separate video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    output_writer = cv2.VideoWriter('outputVideo.avi', fourcc, 29.0, (640, 480))

    term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    roiBox = None
    roiHist = None

    #----------------------------------------

    # temp variables for each frame
    count = 0
    curr_sign = None
    curr_text = ""
    curr_size = 0
    sign_count = 0
    coordinates = []

    # location array appends location of each detected sign
    location = []

    #----------------------------------------

    # image preprocessor object to be used to analyze each input frame
    im_processor=Image_Preprocessor()

    while True:
        success, frame = video.read()
        if not success:
            break


        frame = cv2.resize(frame, (640, 480))
        coordinate, image, sign_type, text = im_processor.analyze(frame, 300,
                                                          0.65, trained_model, count,
                                                                  curr_sign,provider)

        if coordinate is not None:
            cv2.rectangle(image, coordinate[0], coordinate[1], (255, 255, 255), 1)


        if sign_type > 0 and (not curr_sign or sign_type != curr_sign):
            curr_sign = sign_type
            curr_text = text

            # Extracting tl,br of coordinates
            top = int(coordinate[0][1] * 1.05)
            left = int(coordinate[0][0] * 1.05)
            bottom = int(coordinate[1][1] * 0.95)
            right = int(coordinate[1][0] * 0.95)

            # pos of the sign
            location = [count, sign_type if sign_type <= 8 else 8, coordinate[0][0], coordinate[0][1], coordinate[1][0],
                        coordinate[1][1]]

            # rect around the coordinate
            cv2.rectangle(image, coordinate[0], coordinate[1], (0, 255, 0), 1)
            font = cv2.FONT_HERSHEY_PLAIN
            # put sign type text around drawn rect
            cv2.putText(image, text, (coordinate[0][0], coordinate[0][1] - 15), font, 1, (0, 0, 255), 2, cv2.LINE_4)

            tl = [left, top]
            br = [right, bottom]

            curr_size = math.sqrt(math.pow((tl[0] - br[0]), 2) + math.pow((tl[1] - br[1]), 2))

            roi = frame[tl[1]:br[1], tl[0]:br[0]]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
            roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)

            roiBox = (tl[0], tl[1], br[0], br[1])

        elif curr_sign:
            # if current sign persists

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)

            (r, roiBox) = cv2.CamShift(backProj, roiBox, term)
            pts = np.int0(cv2.boxPoints(r))
            s = pts.sum(axis=1)
            tl = pts[np.argmin(s)]
            br = pts[np.argmax(s)]
            size = math.sqrt(pow((tl[0] - br[0]), 2) + pow((tl[1] - br[1]), 2))

            if curr_size < 1 or size < 1 or size / curr_size > 30 or math.fabs(
                            (tl[0] - br[0]) / (tl[1] - br[1])) > 2 or math.fabs(
                            (tl[0] - br[0]) / (tl[1] - br[1])) < 0.5:
                curr_sign = None
            else:
                curr_size = size

            if sign_type > 0:

                # Extracting tl,br of coordinates
                top = int(coordinate[0][1])
                left = int(coordinate[0][0])
                bottom = int(coordinate[1][1])
                right = int(coordinate[1][0])

                # pos of the sign
                location = [count, sign_type if sign_type <= 8 else 8, left, top, right, bottom]
                # rect around the coordinate
                cv2.rectangle(image, coordinate[0], coordinate[1], (0, 255, 0), 1)
                font = cv2.FONT_HERSHEY_PLAIN
                # put sign type text around drawn rect
                cv2.putText(image, text, (coordinate[0][0], coordinate[0][1] - 15), font, 1, (0, 0, 255), 2, cv2.LINE_4)

            elif curr_sign:
                # pos of the sign
                location = [count, sign_type if sign_type <= 8 else 8, tl[0], tl[1], br[0], br[1]]
                # rect around the coordinate
                cv2.rectangle(image, (tl[0], tl[1]), (br[0], br[1]), (0, 255, 0), 1)

                # put sign type text around drawn rect
                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(image, curr_text, (tl[0], tl[1] - 15), font, 1, (0, 0, 255), 2, cv2.LINE_4)

        if curr_sign:
            sign_count += 1
            coordinates.append(location)

        cv2.imshow('Detection', image)
        count = count + 1

        output_writer.write(image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            output_writer.release()
            break

    output_writer.release()
    return




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ITS Project: Spring-2018")

    parser.add_argument(
        '--file',
        default='latest video.mp4',
        help="Input video file"
    )
    parser.add_argument(
        '--load',
        default=None,
        help='Pass a file path to Load already trained model',
        nargs='?'
    )

    args = parser.parse_args()
    main(args)
