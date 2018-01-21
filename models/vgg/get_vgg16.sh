#!/bin/bash
wget -nc -O vgg16.caffemodel http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
wget -nc -O vgg16.prototxt https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/0067c9b32f60362c74f4c445a080beed06b07eb3/VGG_ILSVRC_16_layers_deploy.prototxt
sed -i 's/input_dim: 10/input_dim: 1/g' vgg16.prototxt
sed -i 's/input: "data"/input: "data"\nforce_backward: true/g' vgg16.prototxt
