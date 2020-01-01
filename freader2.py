# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import collections
import numpy as np

from six.moves import xrange  
import tensorflow as tf

class XICSDataSet:    
    def __init__(self, height=20, width=195, batch_size=1000, noutput=15):
        self.depth = 1
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.noutput = noutput

    def trainingset_files_reader(self, im_file_name, lb_file_name, nfiles=1):

        im_filename_queue = tf.train.string_input_producer(im_file_name, shuffle=False)
        lb_filename_queue = tf.train.string_input_producer(lb_file_name, shuffle=False)

        imreader = tf.TextLineReader()
        lbreader = tf.TextLineReader()
        imkey, imvalue = imreader.read(im_filename_queue)
        lbkey, lbvalue = lbreader.read(lb_filename_queue)
        im_record_defaults = [[.0]]*self.height*self.width
        lb_record_defaults = [[.0]]*self.noutput
        im_data_tuple = tf.decode_csv(imvalue, record_defaults=im_record_defaults, field_delim = ' ')
        lb_data_tuple = tf.decode_csv(lbvalue, record_defaults=lb_record_defaults, field_delim = ' ')
        features = tf.pack(im_data_tuple)
        label = tf.pack(lb_data_tuple)

        depth_major = tf.reshape(features, [self.height, self.width, self.depth])

        min_after_dequeue = 10
        capacity = min_after_dequeue + 3 * self.batch_size
        example_batch, label_batch = tf.train.shuffle_batch([depth_major, label], batch_size=self.batch_size, capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)

        return example_batch, label_batch