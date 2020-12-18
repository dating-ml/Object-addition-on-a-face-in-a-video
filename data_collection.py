import numpy as np
import cv2
import os
import imutils

FRAME_SIZE = 1500
RGB_SKOLTECH_GREEN = (35, 180, 164)
OBSERVE_SIDE_LENGTH = 600
OBSERVE_TOP_LEFT_POINT = (500, 140)
SAVE_SIDE_LENGTH = 500

DATA_COLLECTION_PATH = 'gesture_data'
NAME_PERSON = 'yaroslav'

GESTURE_TYPES = ['stop', 'ok', 'victory', 'index_finger', 'thumbs_up', 'empty']
POINTER_TYPES = [0, 0, 0, 0, 0, 0]

def prepare_environment():
	if not os.path.exists(DATA_COLLECTION_PATH):
		os.mkdir(DATA_COLLECTION_PATH)

	person_path_dir = os.path.join(DATA_COLLECTION_PATH, NAME_PERSON)
	if not os.path.exists(person_path_dir):
		os.mkdir(person_path_dir)
		for hand_type in GESTURE_TYPES:
			gesture_path_dir = os.path.join(person_path_dir, hand_type)
			os.mkdir(gesture_path_dir)
	else:
		for i, hand_type in enumerate(GESTURE_TYPES):
			gesture_path_dir = os.path.join(person_path_dir, hand_type)
			collected_images = os.listdir(gesture_path_dir)
			POINTER_TYPES[i] = len(collected_images)

	

	print(POINTER_TYPES)
def get_obsrved_area(frame, point, sideSize):
  	# point[0] - x coord, point[1] - y coord
  	return frame[point[1] : point[1] + sideSize, point[0]: point[0] + sideSize]


prepare_environment()
video_cap = cv2.VideoCapture(0)
print('PERSON:', NAME_PERSON)
while True:
	ret, frame = video_cap.read()
	frame = np.fliplr(frame)
	frame = imutils.resize(frame, width=FRAME_SIZE)
	
	# cv2.imshow('Camera', frame)
	observed_area = get_obsrved_area(frame, OBSERVE_TOP_LEFT_POINT, OBSERVE_SIDE_LENGTH)
	cv2.imshow('Observed area', observed_area)

	observed_area_rsz = cv2.resize(observed_area, (SAVE_SIDE_LENGTH, SAVE_SIDE_LENGTH))
	
	keyboard_pressed = cv2.waitKey(1)
	if keyboard_pressed == ord('s'):
		saved_path = os.path.join(DATA_COLLECTION_PATH,
								  NAME_PERSON,
								  GESTURE_TYPES[0],
								  str(POINTER_TYPES[0]) + '.png')
		POINTER_TYPES[0] += 1
		cv2.imwrite(saved_path, observed_area_rsz)
		print('STOP gesture saved ({})'.format(POINTER_TYPES[0]))
	elif keyboard_pressed == ord('o'):
		saved_path = os.path.join(DATA_COLLECTION_PATH,
								  NAME_PERSON,
								  GESTURE_TYPES[1],
								  str(POINTER_TYPES[1]) + '.png')
		POINTER_TYPES[1] += 1
		cv2.imwrite(saved_path, observed_area_rsz)
		print('OK gesture saved ({})'.format(POINTER_TYPES[1]))
	elif keyboard_pressed == ord('v'):
		saved_path = os.path.join(DATA_COLLECTION_PATH,
								  NAME_PERSON,
								  GESTURE_TYPES[2],
								  str(POINTER_TYPES[2]) + '.png')
		POINTER_TYPES[2] += 1
		cv2.imwrite(saved_path, observed_area_rsz)
		print('VICTORY gesture saved ({})'.format(POINTER_TYPES[2]))
	elif keyboard_pressed == ord('f'):
		saved_path = os.path.join(DATA_COLLECTION_PATH,
								  NAME_PERSON,
								  GESTURE_TYPES[3],
								  str(POINTER_TYPES[3]) + '.png')
		POINTER_TYPES[3] += 1
		cv2.imwrite(saved_path, observed_area_rsz)
		print('FINGER(index) gesture saved ({})'.format(POINTER_TYPES[3]))
	elif keyboard_pressed == ord('t'):
		saved_path = os.path.join(DATA_COLLECTION_PATH,
								  NAME_PERSON,
								  GESTURE_TYPES[4],
								  str(POINTER_TYPES[4]) + '.png')
		POINTER_TYPES[4] += 1
		cv2.imwrite(saved_path, observed_area_rsz)
		print('THUMBS-UP gesture saved ({})'.format(POINTER_TYPES[4]))
	elif keyboard_pressed == ord('e'):
		saved_path = os.path.join(DATA_COLLECTION_PATH,
								  NAME_PERSON,
								  GESTURE_TYPES[5],
								  str(POINTER_TYPES[5]) + '.png')
		POINTER_TYPES[5] += 1
		cv2.imwrite(saved_path, observed_area_rsz)
		print('EMPTY gesture saved ({})'.format(POINTER_TYPES[5]))
	elif keyboard_pressed == 27: # ESCAPE pressed
		print('BYE!')
		break

