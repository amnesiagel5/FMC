from django.shortcuts import render
from django.http import HttpResponse

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
from imutils.video import FPS
from datetime import datetime
# from sendsms import sendSMS
import time
import os
# from plyer import notification
import playsound
import winsound
import numpy as np
import argparse
import sys
import cv2

from math import pow, sqrt

# Create your views here.
def index(request):
	return render(request,'FaceMaskChecker/index.html')

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-e", "--use-email", type=bool, default=0, help="boolean indicating if Emails need to be send or not")
# args = vars(ap.parse_args())

# setup e-mail config if yes.
# if args["use_email"]:
    # from sendEmail import sendEmail

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

# initialize our list of faces, their corresponding locations,
# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

# loop over the detections
	for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the detection
		confidence = detections[0, 0, i, 2]

# filter out weak detections by ensuring the confidence is
# greater than the minimum confidence
		if confidence > 0.4:
    # compute the (x, y)-coordinates of the bounding box for
    # the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

# ensure the bounding boxes fall within the dimensions of
# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

# extract the face ROI, convert it from BGR to RGB channel
# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

# add the face and bounding boxes to their respective
# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

# only make a predictions if at least one face was detected
	if len(faces) > 0:
    # for faster inference we'll make batch predictions on *all*
    # faces at the same time rather than one-by-one predictions
    # in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding locations
	return (locs, preds)


def video(request):
    prototxtPath = "FaceMaskChecker/face_detector/deploy.prototxt"
    weightsPath = "FaceMaskChecker/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    maskNet = load_model("FaceMaskChecker/maskmodel.h5")

    # initialize the video stream
    print("[INFO] starting video stream...")
    # Parse the arguments from command line
    arg = argparse.ArgumentParser(description='Social distance detection')
# removed social distancing arguments
    next_frame_towait = 5  # for sms
    fps = FPS().start()

    labels = [line.strip() for line in open(r"C:\Users\user\PycharmProjects\FMC\class_labels.txt")]

    # Generate random bounding box bounding_box_color for each label
    bounding_box_color = np.random.uniform(0, 255, size=(len(labels), 3))


# Load model
    print("\nLoading model...\n")
    network = cv2.dnn.readNetFromCaffe("FaceMaskChecker/ssd/SSD_MobileNet_prototxt.txt", "FaceMaskChecker/ssd/SSD_MobileNet.caffemodel")

    cap = cv2.VideoCapture(0)

    frame_no = 0

    while 1 and (cap.isOpened()):

        ret, frame = cap.read()

        frame = imutils.resize(frame, width=800)
        (h, w) = frame.shape[:2]
        # Place and Cam number of camera - subject to change when integrated to Admin's dashboard
        placeCam = "Place: Main Gate CAM_01"
        frame = cv2.putText(frame, placeCam, (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_8)

        # Resize the frame to suite the model requirements. Resize the frame to 300X300 pixels
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        network.setInput(blob)
        detections = network.forward()

        pos_dict = dict()
        coordinates = dict()
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask, improper) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
        # label = "Mask" if mask > withoutMask else "No Mask"
        # color = (0, 255, 0) if label == "Mask" else (0, 0, 255)  # 3 brg

        # LIVE COUNT
            nums = len(locs)
            # initialize count of 3 datasets, improper and no face masks are violations
            maskCount = 0
            improperCount = 0
            withoutmaskCount = 0

            for i in range(nums):

                # include the probability in the label
                if (mask > withoutMask and mask > improper):
                    label = "With Face Mask"
                    color = (0, 255, 0)
                    maskCount += 1
                elif (withoutMask > mask and withoutMask > improper):
                    label = "Improper Face Mask"
                    color = (255, 0, 0)
                    improperCount += 1
                    # Alarm when wearing 'Improper Face Mask'
                    winsound.PlaySound(
                        r'C:/Users/user/PycharmProjects/FMC/FaceMaskChecker/static/assets/alarm/beep.wav',
                        winsound.SND_ASYNC)
                    # path = os.path.abspath("mask_incorrect_US_female.wav")
                    # path = os.path.abspath("FaceMaskChecker/static/assets/alarm/mask_incorrect_US_female.wav")
                    # playsound(path)
                elif (improper > mask and improper > withoutMask):
                    label = "No Face Mask"
                    color = (0, 0, 255)
                    withoutmaskCount += 1
                    winsound.PlaySound(
                        r'C:/Users/user/PycharmProjects/FMC/FaceMaskChecker/static/assets/alarm/beep.wav',
                        winsound.SND_ASYNC)
                else:
                    winsound.PlaySound(None, winsound.SND_PURGE)


                label = "{}: {:.2f}% 30* Temperature".format(label, max(mask, withoutMask, improper) * 100)
                # display the label and bounding box rectangle on the output
                cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            # Focal length
                F = 615

            text1 = "WithFaceMask:{} ImproperFaceMask:{} NoFaceMask:{}".format(maskCount, improperCount, withoutmaskCount)
            frame = cv2.putText(frame, text1, (30, 550), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_8)
            text2 = "Area&RiskStatus:"
            frame = cv2.putText(frame, text2, (30, 580), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_8)
            ratio = withoutmaskCount + improperCount / (maskCount + improperCount + withoutmaskCount + 0.000001)

            if ratio >= 0.1 and withoutmaskCount >= 3:
                text = "Danger!, High"
                frame = cv2.putText(frame, text, (260, 580), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_8)
                # if fps._numFrames >= next_frame_towait:  # to send danger sms again,only after skipping few seconds
                    # msg = "** Face Mask System Alert ** \n\n"
                    # msg += "Camera ID: 001 \n\n"
                    # msg += "Status: Danger! \n\n"
                    # msg += "NoFaceMask Count: " + str(withoutmaskCount) + " \n"
                    # msg += "ImproperFaceMask Count: " + str(improperCount) + " \n"
                    # msg += "WithFaceMask Count: " + str(maskCount) + " \n"
                    # c = time.localtime()  # get struct_time
                    # Get date and time and save it inside a variable
                    # msg += "Date-Time of alert: \n" + time.strftime("%B %d, %Y - %I:%M:%S %p", c)
                    # sendSMS(msg,[7041677471])
                    # print('Sms sent')
                    # sendEmail(msg)
                    # next_frame_towait = fps._numFrames + (5 * 25)

            elif ratio != 0 and np.isnan(ratio) != True:
                text = "Warning, Medium"
                frame = cv2.putText(frame, text, (260, 580), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_8)

            else:
                text = "Safe, Low"
                frame = cv2.putText(frame, text, (260, 580), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_8)

        for i in range(detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > 0.2:

                class_id = int(detections[0, 0, i, 1])

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')

                # Filtering only persons detected in the frame. Class Id of 'person' is 15
                if class_id == 15.00:

                    # Draw bounding box for the object
                    cv2.rectangle(frame, (startX, startY), (endX, endY), bounding_box_color[class_id], 2)

                    label = "{}: {:.2f}%".format(labels[class_id], confidence * 100)
                    print("{}".format(label))


                    coordinates[i] = (startX, startY, endX, endY)

                    # Mid point of bounding box
                    x_mid = round((startX+endX)/2,4)
                    y_mid = round((startY+endY)/2,4)

                    height = round(endY-startY,4)

                    # Distance from camera based on triangle similarity
                    distance = (165 * F)/height
                    print("Distance(cm):{dist}\n".format(dist=distance))

                    # Mid-point of bounding boxes (in cm) based on triangle similarity technique
                    x_mid_cm = (x_mid * distance) / F
                    y_mid_cm = (y_mid * distance) / F
                    pos_dict[i] = (x_mid_cm, y_mid_cm, distance)

        # Distance between every object detected in a frame
        close_objects = set()
        for i in pos_dict.keys():
            for j in pos_dict.keys():
                if i < j:
                    dist = sqrt(pow(pos_dict[i][0]-pos_dict[j][0], 2) + pow(pos_dict[i][1]-pos_dict[j][1], 2) + pow(pos_dict[i][2]-pos_dict[j][2], 2))

                    # Check if distance less than 2 metres or 200 centimetres
                    if dist < 200:
                        close_objects.add(i)
                        close_objects.add(j)

        for i in pos_dict.keys():
            if i in close_objects:
                COLOR = (0, 0, 255)
            else:
                COLOR = (0, 255, )
            (startX, startY, endX, endY) = coordinates[i]

            cv2.rectangle(frame, (startX, startY), (endX, endY), COLOR, 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            # Convert cms to feet
            cv2.putText(frame, 'Depth: {i} ft'.format(i=round(pos_dict[i][2]/30.48, 4)), (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)


        cv2.namedWindow('FaceMaskChecker&Detection', cv2.WINDOW_NORMAL)
# LIVE Count temporary
        text3 = "WithFaceMask:  ImproperFaceMask:  NoFaceMask:  "
        frame = cv2.putText(frame, text3, (30, 550), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2, cv2.LINE_8)
        text4 = "Area&RiskStatus:"
        frame = cv2.putText(frame, text4, (30, 580), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2, cv2.LINE_8)

# process to display date and time in video feed
        if ret:
            # describe the type of font you want to display
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL

            c = time.localtime()  # get struct_time
            d = time.strftime("%B %d, %Y - %I:%M:%S %p", c)
            # Get date and time and save it inside a variable
            dt = d

            # put the dt variable over the video frame
            frame = cv2.putText(frame, dt, (10, 52), font, 1, (255, 255, 255), 1, cv2.LINE_8)


# Show frame
        cv2.imshow('FaceMaskChecker&Detection', frame)

        key = cv2.waitKey(1) & 0xFF

        # Press `q` to exit
        if key == ord("q"):
            break

    # Clean
    cap.release()
    cv2.destroyAllWindows()
    return render(request, 'FaceMaskChecker/video.html')
