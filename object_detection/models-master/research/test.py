# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 01:13:29 2021

@author: Riad
"""

from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
logical_devices = tf.config.list_logical_devices('GPU')
print("Logical GPU: ",len(logical_devices))