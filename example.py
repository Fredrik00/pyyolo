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
	img = orig_img.transpose(2,0,1)
	c, h, w = img.shape[0], img.shape[1], img.shape[2]
	# print w, h, c 
	data = img.ravel()/255.0
	data = np.ascontiguousarray(data, dtype=np.float32)
	outputs = pyyolo.detect(w, h, c, data, thresh, hier_thresh)	
	for output in outputs:
		print(output)
		# left is X-coordinate of left side of rect, etc.-
		# Therefore, the top left corner is (left, top) and bottom right is (right, bottom)
		left, right, bottom, top = output['left'], output['right'], output['bottom'], output['top']
		prob, label = output['prob'], output['class']
		text = "{0} ({1:.3f})".format(label, prob)
		cv2.rectangle(orig_img, (left, top), (right, bottom), color=(0, 255, 0))
		cv2.putText(orig_img, text, (left, top), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
	cv2.imwrite("predicted.jpg",  orig_img)
	i = i + 1

# free model
pyyolo.cleanup()
