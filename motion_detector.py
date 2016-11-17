#http://www.pyimagesearch.com/2016/01/04/unifying-picamera-and-cv2-videocapture-into-a-single-class-with-opencv/

import numpy as np
import argparse
import datetime
import cv2
import imutils
from imutils.video import VideoStream
from pyimagesearch.motion_detection import SingleMotionDetector
import json
from scipy.spatial import distance as dist
import time
import requests
from threading import Timer

ap = argparse.ArgumentParser()

ap.add_argument("-p", "--picamera", type=int, default=-1,
    help="whether or not the Raspberry Pi camera should be used")

args = vars(ap.parse_args())

with open('config.json') as data_file:
    data = json.load(data_file)

mailgunSecretApiKey = data["mailgunSecretApiKey"]
print("mailgunSecretApiKey:", mailgunSecretApiKey)
mailgunToAddress = data["mailgunToAddress"]
print("mailgunToAddress:", mailgunToAddress)
mailgunDomainName = data["mailgunDomainName"]
print("mailgunDomainName:", mailgunDomainName)
minFrames = data["minFrames"]
print("minFrames:", minFrames)
timeToWaitBetweenNotification = data["timeToWaitBetweenNotification"]
print("timeToWaitBetweenNotification:", timeToWaitBetweenNotification)

# initialize the video stream and allow the cammera sensor to warmup
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

#TODO: check what conf returns if not defined
if mailgunSecretApiKey is None:
    print("mail gun secret key is not defined")

time.sleep(2.0)

md = SingleMotionDetector(accumWeight=0.1)
total = 0
consec = None
frameShape = None
waitBetweenNotification = False


# keep looping
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # grab the current timestamp and draw it on the frame
    timestamp = datetime.datetime.now()
    cv2.putText(frame, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # if we do not already have the dimensions of the frame, initialize it
    if frameShape is None:
        frameShape = (int(gray.shape[0] / 2), int(gray.shape[1] / 2))

    # if the total number of frames has reached a sufficient number to construct a
    # reasonable background model, then continue to process the frame
    if total > minFrames:  
        # detect motion in the image
        motion = md.detect(gray)

        # if the `motion` object not None, then motion has occurred in the image
        if motion is not None:
            # unpack the motion tuple, compute the center (x, y)-coordinates of the
            # bounding box, and draw the bounding box of the motion on the output frame
            (thresh, (minX, minY, maxX, maxY)) = motion
            cX = int((minX + maxX) / 2)
            cY = int((minY + maxY) / 2)
            cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 0, 255), 2)

            # if the number of consecutive frames is None, initialize it using a list
            # of the number of total frames, the frame itself, along with distance of
            # the centroid to the center of the image
            if consec is None:
                consec = [1, frame, dist.euclidean((cY, cX), frameShape)]

            # otherwise, we need to update the bookkeeping variable
            else:
                # compute the Euclidean distance between the motion centroid and the
                # center of the frame, then increment the total number of *consecutive
                # frames* that contain motion
                d = dist.euclidean((cY, cX), frameShape)
                consec[0] += 1

                # if the distance is smaller than the current distance, then update the
                # bookkeeping variable
                if d < consec[2]:
                    consec[1:] = (frame, d)
            print(consec[0])
            cv2.imshow("Frame", frame)
            # if a sufficient number of frames have contained motion, log the motion
            if consec[0] == minFrames and waitBetweenNotification is False:
                
                def setWaitBetweenNotification():
                    global waitBetweenNotification
                    waitBetweenNotification = False

                waitBetweenNotification = True
                t = Timer(timeToWaitBetweenNotification, setWaitBetweenNotification)
                t.start()

                # MOTION DETECTED!  DO SOMETHING COOL!
                cv2.imwrite("frame.jpg", consec[1])
                print("[INFO] logging motion to file: {0}".format(timestamp))
                r = requests.post(
                    "https://api.mailgun.net/v3/{0}/messages".format(mailgunDomainName),
                    auth=("api", mailgunSecretApiKey),
                    files=[("inline", open("frame.jpg", 'rb'))],
                    data={"from": "Motion Camera <mailgun@{0}>". format(mailgunDomainName),
                          "to": [mailgunToAddress],
                          "subject": "Motion detected",
                          "text": "Motion was detected at: {0}".format(timestamp),
                          "html": '<html>yo: <img src="cid:frame.jpg"></html>'})
                print(r.status_code)
                print(r.text)

                consec = None


        # otherwise, there is no motion in the frame so reset the consecutive bookkeeing
        # variable
        else:
            consec = None

    # update the background model and increment the total number of frames read thus far
    md.update(gray)
    total += 1

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
