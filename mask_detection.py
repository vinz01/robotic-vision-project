from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import os
import cv2

# initialise
face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
landmark_detect = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Read from camera
vs = VideoStream(src=0).start()
time.sleep(1.0)

while True:

	# Read from camera
	frame = vs.read()

	# Resize to speed up processing
	frame = imutils.resize(frame, width=600)

	# Chuyen ve gray
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces in photos
	faces = face_detect.detectMultiScale(gray, scaleFactor=1.1,		minNeighbors=5, minSize=(100, 100),		flags=cv2.CASCADE_SCALE_IMAGE)

	# Browse through the items
	for (x, y, w, h) in faces:

		# Create a rectangle around the face
		rect = dlib.rectangle(int(x), int(y), int(x + w),
			int(y + h))

		# Identify landmarks
		landmark = landmark_detect(gray, rect)
		landmark = face_utils.shape_to_np(landmark)

		# Capture mouth area
		(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
		mouth = landmark[mStart:mEnd]

		# Take the rectangle around the mouth
		boundRect = cv2.boundingRect(mouth)
		cv2.rectangle(frame,
					  (int(boundRect[0]), int(boundRect[1])),
					  (int(boundRect[0] + boundRect[2]),  int(boundRect[1] + boundRect[3])), (0,0,255), 2)

		# Calculate average saturation
		hsv = cv2.cvtColor(frame[int(boundRect[1]):int(boundRect[1] + boundRect[3]),int(boundRect[0]):int(boundRect[0] + boundRect[2])], cv2.COLOR_RGB2HSV)
		sum_saturation = np.sum(hsv[:, :, 1])
		area = int(boundRect[2])*int(boundRect[3])
		avg_saturation = sum_saturation / area

		# Check with threshold
		if avg_saturation>100:
			cv2.putText(frame, "ALERT!!! Not Wearing Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
						2)
		if avg_saturation < 100:
			cv2.putText(frame, "POSSIBLY WEARING MASK", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
						2)

	
	cv2.imshow("Camera", frame)

	
	key = cv2.waitKey(1) & 0xFF
	if key == 27:
		break


cv2.destroyAllWindows()
vs.stop()

'''
Detect faces then use landmark to detect mouth area. 
Then calculate the average saturation and compare with a threshold we 
set to check whether or not to wear a mask. This method has 
the advantage of not needing data, the speed is higher than that 
of method 1 but sometimes due to changing lighting conditions, 
it may not be able to detect correctly.

The task of this file will be to read images from the webcam, detect faces, 
capture mouth areas and calculate color saturation to determine whether a mask is worn or not.

First we convert the oral region image to HSV and get the Saturation channel separately to calculate.
Next we calculate the average of the saturation of the mouth.
We compare that average to the threshold we set, which we choose is 100. 
If the value <100 is a mask (much white) and vice versa.

 
'''