import cv2 
import numpy as np
import dlib
from collections import OrderedDict
import imutils
from imutils import face_utils
import keras
from keras.models import load_model
import cv2
import numpy as np
import dlib
from math import hypot

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
 

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

RGB_SKOLTECH_GREEN = (35, 180, 164)
WINDOWS_SHAPE = (400, 400) # (len_x, len_y)
GESTURE_SIZE = (64, 64)
GESTURE_TYPES = ['stop', 'victory', 'thumbs_up', 'empty']

GEST2IND = dict(zip(GESTURE_TYPES, np.arange(len(GESTURE_TYPES))))
IND2GEST = dict(zip(np.arange(len(GESTURE_TYPES)),GESTURE_TYPES))

# gesture_model = load_model('model-013-0.94.h5')
gesture_model = load_model('model-013-0.94.h5')

# preprocessing of our object
imgMustache = cv2.imread('mustache.png',-1)
orig_mask = imgMustache[:,:,3]
orig_mask_inv = cv2.bitwise_not(orig_mask)
imgMustache = imgMustache[:,:,0:3]
origMustacheHeight, origMustacheWidth = imgMustache.shape[:2]

# pig nose
nose_image = cv2.imread("pig_nose.png")
nose_mask = np.zeros((1000, 1000), np.uint8)
pig_mode = False

video_cap = cv2.VideoCapture(0)


def preprocess_gesture_area(img):
	# resize image
	img = cv2.resize(img, GESTURE_SIZE)
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     gray = cv2.GaussianBlur(gray, (5, 5), 3)
#     gray = th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#     gray = gray / 255
	return img / 255

def get_gesture_area_img(frame):
	return frame[0:WINDOWS_SHAPE[1], frame.shape[1] - WINDOWS_SHAPE[0]: ]

def get_haar_cascade(img_):
	img = img_.copy()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	return img


cntInd = 0
curInd = 3
while True:
	ret, frame = video_cap.read()
	frame = np.fliplr(frame)

	# haar_drawed = get_haar_cascade(frame)

	# cv2.imshow('video frame', haar_drawed)

	CHEEK_IDXS = OrderedDict([("left_cheek", (1,2,3,4,5,48,49,31)),
						("right_cheek", (11,12,13,14,15,35,53,54))
						 ])


	frame = imutils.resize(frame, width=1000)
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# haar cascade 
	# faces = face_cascade.detectMultiScale(gray,
	# 	scaleFactor=1.1,
	# 	minNeighbors=5,
	# 	minSize=(30, 30),
	# 	flags=cv2.CASCADE_SCALE_IMAGE
	# )
 

	frame = cv2.rectangle(frame,
						  (frame.shape[1] - WINDOWS_SHAPE[0], 0),
						  (frame.shape[1], WINDOWS_SHAPE[1]),
						  RGB_SKOLTECH_GREEN,
						  thickness=2,
						  lineType=8,
						  shift=0)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# img = imutils.resize(frame, width=600)

	overlay = frame.copy()
	detections = detector(gray, 0)
	for d in detections:
		shape = predictor(gray, d)
		shape = face_utils.shape_to_np(shape)
		for (x, y) in shape:
			cv2.circle(frame, (x, y), 1, RGB_SKOLTECH_GREEN, 2)		
		# for 
	#     # for (_, name) in enumerate(CHEEK_IDXS.keys()):
	#     #     pts = np.zeros((len(CHEEK_IDXS[name]), 2), np.int32) 
	#     #     for i,j in enumerate(CHEEK_IDXS[name]): 
	#     #         pts[i] = [shape.part(j).x, shape.part(j).y]

	#     #     pts = pts.reshape((-1,1,2))
	#     #     cv2.polylines(overlay,[pts],True,(0,255,0),thickness = 2)


	gesture_area = get_gesture_area_img(frame)
	
	cv2.imshow('Gesture area', gesture_area)

	gesture_area_processed = preprocess_gesture_area(gesture_area) 

	prediction = gesture_model.predict(gesture_area_processed[np.newaxis,:,:,:])

	ans_ind = np.argmax(prediction)

	if ans_ind == 0:
		if curInd == 0:
			cntInd += 1
		else:
			cntInd = 1
			curInd = 0
	elif ans_ind == 1:
		if curInd == 1:
			cntInd += 1
		else:
			cntInd = 1
			curInd = 1
	elif ans_ind == 2:
		if curInd == 2:
			cntInd += 1
		else:
			cntInd = 1
			curInd = 2

	if cntInd == 10:
		if ans_ind == 0: # stop
			break
		elif ans_ind == 1: # victory
			pig_mode = True
		elif ans_ind == 2: #thumbs up
			break
	if pig_mode:
		faces = detector(frame)
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		for face in faces:
				landmarks = predictor(gray_frame, face)

				top_nose = (landmarks.part(29).x, landmarks.part(29).y)
				center_nose = (landmarks.part(30).x, landmarks.part(30).y)
				left_nose = (landmarks.part(31).x, landmarks.part(31).y)
				right_nose = (landmarks.part(35).x, landmarks.part(35).y)
				nose_width = int(hypot(left_nose[0] - right_nose[0],
								   left_nose[1] - right_nose[1]) * 1.7)
				nose_height = int(nose_width * 0.77)


				top_left = (int(center_nose[0] - nose_width / 2),
									  int(center_nose[1] - nose_height / 2))
				bottom_right = (int(center_nose[0] + nose_width / 2),
							   int(center_nose[1] + nose_height / 2))

				nose_pig = cv2.resize(nose_image, (nose_width, nose_height))

				nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)

				_, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)

				nose_area = frame[top_left[1]: top_left[1] + nose_height,
							top_left[0]: top_left[0] + nose_width]
				
				nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
				final_nose = cv2.add(nose_area_no_nose, nose_pig)
				frame[top_left[1]: top_left[1] + nose_height,
							top_left[0]: top_left[0] + nose_width] = final_nose
	
	print(IND2GEST[ans_ind])
	
	font                   = cv2.FONT_HERSHEY_SIMPLEX
	bottomLeftCornerOfText = (600,200)
	fontScale              = 2
	fontColor              = RGB_SKOLTECH_GREEN
	lineType               = 2

	cv2.putText(frame, IND2GEST[ans_ind], 
	    bottomLeftCornerOfText, 
	    font, 
	    fontScale,
	    fontColor,
	    lineType)

	cv2.imshow('Camera', frame)
	if cv2.waitKey(1) == 27: # ESCAPE pressed
		break

	# if cv2.waitKey(1): break
video_cap.release()
cv2.destroyAllWindows()