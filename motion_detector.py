import numpy as np
import argparse
import datetime
import cv2
import imutils
from pyimagesearch.motion_detection import SingleMotionDetector
import json
from scipy.spatial import distance as dist
import time
import requests
from threading import Timer

ap = argparse.ArgumentParser()

ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

with open('config.json') as data_file:
    data = json.load(data_file)

mailgunSecretApiKey = data["mailgunSecretApiKey"]
mailgunToAddress = data["mailgunToAddress"]
mailgunDomainName = data["mailgunDomainName"]
minFrames = data["minFrames"]
timeToWaitBetweenNotification = data["timeToWaitBetweenNotification"]
print(data)
print(mailgunSecretApiKey, mailgunToAddress, mailgunDomainName, minFrames)

#TODO: check what conf returns if not defined
if mailgunSecretApiKey is None:
    print("mail gun secret key is not defined")

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
    print("Streaming from camera")

# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])
    print("reading", args["video"])

time.sleep(2.0)

md = SingleMotionDetector(accumWeight=0.1)
total = 0
consec = None
frameShape = None
waitBetweenNotification = False


# keep looping
while True:
    # grab the current frame
    (success, frame) = camera.read()

    if success is False:
        print("End of stream")
        break

    #stuff
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
            # if a sufficient number of frames have contained motion, log the motion
            if consec[0] == minFrames and waitBetweenNotification is False:
                
                def setWaitBetweenNotification():
                    waitBetweenNotification = False

                waitBetweenNotification = True
                t = Timer(timeToWaitBetweenNotification, setWaitBetweenNotification)
                t.start()

                # MOTION DETECTED!  DO SOMETHING COOL!
                print("[INFO] logging motion to file: {0}".format(timestamp))
                r = requests.post(
                    "https://api.mailgun.net/v3/{0}/messages".format(mailgunDomainName),
                    auth=("api", mailgunSecretApiKey),
                    data={"from": "Excited User <mailgun@{0}>". format(mailgunDomainName),
                          "to": [mailgunToAddress],
                          "subject": "Motion detected",
                          "text": "Motion was detected at: {0}".format(timestamp)})
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

    cv2.imshow("Frame", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
