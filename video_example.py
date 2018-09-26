import pyyolo
import numpy as np
import sys
import cv2

darknet_path = './darknet'
datacfg = 'cfg/coco.data'
cfgfile = 'cfg/yolov3.cfg'
weightfile = '../yolov3.weights'
filename = darknet_path + '/data/dog.jpg'
thresh = 0.45
hier_thresh = 0.5

# OpenCV 
# cam = cv2.VideoCapture(-1)
# ret_val, img = cam.read()
# print(ret_val)
# if ret_val:
#     ret_val = cv2.imwrite(filename,img)
#     print(ret_val)

pyyolo.init(darknet_path, datacfg, cfgfile, weightfile)

def predict_sample_image():
	# From file
	print('----- test original C using a file')
	outputs = pyyolo.test(filename, thresh, hier_thresh, 0)
	for output in outputs:
		print(output)

	# Camera 
	print('----- test python API using a file')
	i = 1
	while i < 2:
		# ret_val, img = cam.read()
		orig_img = cv2.imread(filename)
		c, h, w, data = prepare_img(orig_img)
		outputs = pyyolo.detect(w, h, c, data, thresh, hier_thresh)	
		draw_bounding_boxes(orig_img, outputs)
		cv2.imwrite("predicted.jpg",  orig_img)
		i = i + 1

def prepare_img(img):
	img = img.transpose(2,0,1)
	c, h, w = img.shape[0], img.shape[1], img.shape[2]
	# print w, h, c 
	data = img.ravel()/255.0
	data = np.ascontiguousarray(data, dtype=np.float32)
	return c, h, w, data


def draw_bounding_boxes(img, outputs):
	for output in outputs:
		print(output)
		# left is X-coordinate of left side of rect, etc.-
		# Therefore, the top left corner is (left, top) and bottom right is (right, bottom)
		left, right, bottom, top = output['left'], output['right'], output['bottom'], output['top']
		prob, label = output['prob'], output['class']
		text = "{0} ({1:.3f})".format(label, prob)
		cv2.rectangle(img, (left, top), (right, bottom), color=(0, 255, 0))
		cv2.putText(img, text, (left, top), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)


def predict_video(filename):
	cap = cv2.VideoCapture(filename)

	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'MP4V')
	out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (1920,1080))

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret: # out of frames
			break
		c, h, w, data = prepare_img(frame)
		outputs = pyyolo.detect(w, h, c, data, thresh, hier_thresh)	
		draw_bounding_boxes(frame, outputs)
		out.write(frame)

	cap.release()
	out.release()

predict_video('MOV_0211.mp4')

# free model
#pyyolo.cleanup()