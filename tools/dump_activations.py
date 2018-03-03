#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Script to dump filter and feature data from a trained net as images (modified version of script in private repository)
Reference: https://github.com/BVLC/caffe/blob/master/examples/00-classification.ipynb

__author__ = "Martin Lautenbacher"
__version__ = "0.5"

Usage:
    - image_path
    - model_prototxt
    - model_weigts
    - save_folder = images/
    - layer_list_file = feature_layers.txt
    - labels_file = imagenet_labels.txt
    - mode = 0 (0=cpu, 1=gpu)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os, sys, inspect
import json
import argparse

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from misc import mkdir_p

sys.path.insert(0, os.path.join(parentdir, 'caffe', 'python'))
import caffe

def main():

    parser = argparse.ArgumentParser(description='Command line arguments')

    parser.add_argument('prototxt', type = str, help = 'path to network prototxt')
    parser.add_argument('weights', type = str, help = 'path to .caffemodel weights')
    parser.add_argument('--outdir', type = str, default = os.path.join(currentdir, 'result'), help = 'output directory')
    parser.add_argument('--image_path', type = str, default = None, help = 'path to image. Will be used instead if image_list')
    parser.add_argument('--image_list', type = str, default = os.path.join(parentdir, 'INPUT', 'image_list.txt'), help = 'path to image list')
    parser.add_argument('--all_images', type = bool, default = False, help = 'should all images be used?')
    parser.add_argument('--unit_list',    type = str, default = None, help = 'path to list of units to consider')
    parser.add_argument('--gpu', type = bool, default = False, help = 'use gpu or not')


    args = parser.parse_args()

    if args.gpu:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    if args.unit_list != None:
        assert os.path.exists(args.unit_list)
        with open(args.unit_list, 'rt') as f:
            unit_list = json.load(f)
    else:
        unit_list = None

    net = caffe.Net(args.prototxt,
                    args.weights,
                    caffe.TEST)

    if args.image_path != None:
        assert os.path.exists(args.image_path), args.image_path + ' doesn\'t exist'
        image_path = args.image_path
    else:
        assert os.path.exists(args.image_list), args.image_list + ' doesn\'t exist'
        with open(args.image_list) as f:
            image_list = f.read().split()
            image_dir = os.path.dirname(args.image_list)
            image_path = os.path.join(image_dir, image_list[0])
            assert os.path.exists(image_path), image_path + ' doesnt\'t exist'


    mean_value = np.array([103.939, 116.779, 123.68]) #BGR!
    transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', mean_value)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))

    net.blobs['data'].reshape(1,        # batch size
                            3,         # 3-channel (BGR) images
                            224, 224)  # image size is 224x224
    

    def process_image(img_path, all_images = False, img_idx = None):
        image = caffe.io.load_image(img_path)
        net.blobs['data'].data[...] = transformer.preprocess('data', image)
        net.forward()

        if unit_list != None:
            for l, units in unit_list.iteritems():
                features = net.blobs[l].data.copy()
                if all_images:
                    if img_idx == 0:
                        filters = net.params[l][0].data.copy()
                        save_vis_data(filters, os.path.join(args.outdir, l, 'filters'), set(units))
                    save_vis_data(features, os.path.join(args.outdir, l, 'features', str(img_idx)), set(units))
                else:
                    save_vis_data(filters, os.path.join(args.outdir, l, 'filters'), set(units))
                    save_vis_data(features, os.path.join(args.outdir, l, 'features'), set(units))
        else:
            for l, _ in net.blobs.iteritems():
                if 'conv' in l:
                    features = net.blobs[l].data.copy()
                    if all_images:
                        if img_idx == 0:
                            filters = net.params[l][0].data.copy()
                            save_vis_data(filters, os.path.join(args.outdir, l, 'filters'))
                        save_vis_data(features, os.path.join(args.outdir, l, 'features', str(img_idx)))
                    else:
                        save_vis_data(filters, os.path.join(args.outdir, l, 'filters'))
                        save_vis_data(features, os.path.join(args.outdir, l, 'features'))
    
    if args.all_images:
        for i, p in enumerate(image_list):
            print 'processing image:', p
            process_image(os.path.join(image_dir, p), True, i)
    else:
        process_image(image_path)


def save_vis_data(data, folder, unit_list = None):
    #normalize
    #data = (data - data.min()) / (data.max() - data.min())
    mkdir_p(folder)
    if unit_list != None:
        x1 = data.shape[0]
        x2 = data.shape[1]
        for unit in unit_list:
            if x1 == 1:
                img = data[0][unit]
            else:
                idx_x1 = unit / x1
                idx_x2 = unit % x1
                img = data[idx_x1, idx_x2] #TODO correct?
            if img.shape[0] <= 40 or img.shape[1] <= 40:
                    img = cv2.resize(img, (50, 50),
                            interpolation=cv2.INTER_NEAREST)
            plt.imsave(os.path.join(folder, str(unit) + '.png'), img)

    else:
        count = 0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                img = data[i, j]
                if img.shape[0] <= 40 or img.shape[1] <= 40:
                    img = cv2.resize(img, (50, 50),
                            interpolation=cv2.INTER_NEAREST)
                plt.imsave(folder + '/' + str(count) + '.png', img)
                count += 1

if __name__ == '__main__':
    main()
