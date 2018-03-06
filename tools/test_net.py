#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Script to test a networks top-1 and top-5 accuracy
Reference: https://github.com/BVLC/caffe/blob/master/examples/00-classification.ipynb


__author__ = "Martin Lautenbacher"
__version__ = "0.5"


"""

import os, sys, inspect
import argparse
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

sys.path.insert(0, os.path.join(parentdir, 'caffe', 'python'))
import caffe
from misc import mkdir_p

def main():
    parser = argparse.ArgumentParser(description='Command line arguments')

    parser.add_argument('prototxt', type = str, help = 'path to network prototxt')
    parser.add_argument('weights', type = str, help = 'path to .caffemodel weights')
    parser.add_argument('labels', type = str, help = 'path to image list')    
    parser.add_argument('--outdir', type = str, default = os.path.join(currentdir, 'result'), help = 'output directory')
    parser.add_argument('--image_list', type = str, default = os.path.join(parentdir, 'INPUT', 'image_list.txt'), help = 'path to image list')
    parser.add_argument('--gpu', type = bool, default = False, help = 'use gpu or not')

    args = parser.parse_args()

    mkdir_p(args.outdir)

    if args.gpu:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    assert os.path.exists(args.labels), args.labels + " doesn\'t exist"
    with open (args.labels) as f:
        labels = f.read().split()

    
    net = caffe.Net(args.prototxt,
                args.weights,
                caffe.TEST)
    
    assert os.path.exists(args.image_list), args.image_list + ' doesn\'t exist'

    # with open(args.image_list) as f:
    #     image_list = f.read().split()
    #     image_dir = os.path.dirname(args.image_list)
    #     image_path = os.path.join(image_dir, image_list[0])
    #     assert os.path.exists(image_path), image_path + ' doesnt\'t exist'
    with open(args.image_list, 'rt') as f:
        image_list = []
        image_dir = os.path.dirname(args.image_list)
        for line in f:
            l = line.strip().split(' ')
            path = l[0]
            label_id = l[1]
            image_list.append((path, int(label_id)))

    mean_value = np.array([103.939, 116.779, 123.68]) #BGR!
    transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', mean_value)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))


    # net.blobs['data'].reshape(1,        # batch size
    #                         3,         # 3-channel (BGR) images
    #                         224, 224)  # image size is 224x224

    top_1_correct = 0
    top_5_correct = 0

    predictions_file = open(os.path.join(args.outdir, 'predictions.txt'), 'wt')

    for i, img_data in enumerate(image_list):
        image = caffe.io.load_image(os.path.join(image_dir, img_data[0]))
        net.blobs['data'].data[...] = transformer.preprocess('data', image)
        output = net.forward()
        output_prob = output['prob'][0]
        top_inds = output_prob.argsort()[::-1][:5]
        if output_prob.argmax() == img_data[1]:
            top_1_correct += 1
        if img_data[1] in top_inds:
            top_5_correct += 1
        # top_5_probs = zip(output_prob[top_inds], top_inds)
        debug_text = "%04d %s  %20s, %02d\tcorrect label: %20s, %02d\t -> %6s, top-5: %s -> %6s\n" % (i, img_data[0], labels[output_prob.argmax()], output_prob.argmax(), labels[img_data[1]], img_data[1], str(output_prob.argmax() == img_data[1]), str(top_inds), str(img_data[1] in top_inds))
        predictions_file.write(debug_text)

    predictions_file.close()

    top_1_avg = top_1_correct / float(len(img_data))
    top_5_avg = top_5_correct / float(len(img_data))

    with open(os.path.join(args.outdir, 'avg_correct.txt'), 'wt') as f:
        f.write("Top-1:\t%f\n" % top_1_avg)
        f.write("Top-1 correct: %4d of %4d\n" % (top_1_correct, len(image_list)))
        f.write("Top-5:\t%f\n" % top_5_avg)
        f.write("Top-5 correct: %4d of %4d\n" % (top_5_correct, len(image_list)))


if __name__ == '__main__':
    main()
