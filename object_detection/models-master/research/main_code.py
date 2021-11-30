# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 21:50:05 2021

@author: Riad
"""

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import os
import pathlib

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display


from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


'''print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
logical_devices = tf.config.list_logical_devices('GPU')
print("Logical GPU: ",len(logical_devices))
'''
def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  model = tf.saved_model.load(str(model_dir))
  model = model.signatures['serving_default']

  return model

PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

PATH_TO_TEST_IMAGES_DIR = pathlib.Path('object_detection/test_images')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
TEST_IMAGE_PATHS

model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  input_tensor = tf.convert_to_tensor(image)
  input_tensor = input_tensor[tf.newaxis,...]
  output_dict = model(input_tensor)
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  if 'detection_masks' in output_dict:
      detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
      detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
      output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict

def show_inference(model, image,class_id):
  image_np = np.array(image)
  output_dict = run_inference_for_single_image(model, image_np)
  boxes = []
  classes = []
  scores = []
  for i,x in enumerate(output_dict['detection_classes']):
    if x==class_id and output_dict['detection_scores'][i] > 0.5:
      classes.append(x)
      boxes.append(output_dict['detection_boxes'][i])
      scores.append(output_dict['detection_scores'][i])
  boxes = np.array(boxes)
  classes = np.array(classes)
  scores = np.array(scores)
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      boxes,
      classes,
      scores,
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=2)
  
  #display(Image.fromarray(image_np))
  return image_np

'''img=[]
  
for image_path in TEST_IMAGE_PATHS:
  img.append(show_inference(detection_model, image_path, 1))


import cv2
import numpy as np
count=0
for im in img:
    cv2.imshow('org'+str(count),im)
    count=count+1

#cv2.imshow('org',img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
# For more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
import cv2
import numpy as np

# Playing video from file:
# cap = cv2.VideoCapture('vtest.avi')
# Capturing video from webcam:
cap = cv2.VideoCapture("http://192.168.0.101:4747/mjpegfeed")

currentFrame = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Handles the mirroring of the current frame
    #frame = cv2.flip(frame,1)
    #frame= cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    result=show_inference(detection_model, frame, 1)
    # Our operations on the frame come here

    # Saves image of the current frame in jpg file
    # name = 'frame' + str(currentFrame) + '.jpg'
    # cv2.imwrite(name, frame)

    # Display the resulting frame
    cv2.imshow('frame',result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()