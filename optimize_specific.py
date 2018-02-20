#! /usr/bin/env python2
"""
Created on Mon Feb 19 16:20:44 2018

@author: martin
"""

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
import argparse
import json
import settings

def main():
    parser = argparse.ArgumentParser(description='executes optimize_image.py for specific layers/units')
    parser.add_argument('unit_list',    type = str, default = None, help = 'path to list of units to consider')
    parser.add_argument('--caffe_root', type = str, default = settings.caffevis_caffe_root, help = 'Path to caffe root directory.')
    parser.add_argument('--nmt_pkl',      type = str, default = os.path.join(settings.caffevis_outputs_dir, 'find_max_acts_output.pickled'), help = 'Which pickled NetMaxTracker to load.')
    parser.add_argument('--net_prototxt', type = str, default = settings.caffevis_deploy_prototxt, help = 'network prototxt to load')
    parser.add_argument('--net_weights',  type = str, default = settings.caffevis_network_weights, help = 'network weights to load')
    parser.add_argument('--filelist',     type = str, default = settings.static_files_input_file, help = 'List of image files to consider, one per line. Must be the same filelist used to produce the NetMaxTracker!')
    parser.add_argument('--outdir',       type = str, default = settings.caffevis_outputs_dir, help = 'Which output directory to use. Files are output into outdir/layer/unit_%%04d/{maxes,deconv,backprop}_%%03d.png')
  
    parser.add_argument('-il', '--ignore_layers', nargs='*', default = [], help = 'ignore these layers')

    args = parser.parse_args()

    if args.unit_list != None:
        assert os.path.exists(args.unit_list)
        with open(args.unit_list, 'rt') as f:
            unit_list = json.load(f)
    
    for layer in unit_list:
            if not os.path.exists(os.path.join(args.outdir, layer)) and not layer in args.ignore_layers:
                units = set(unit_list[layer])
                for unit in units:
                    os.system("%s --caffe-root=%s --deploy-proto=%s --net-weights=%s --data-size='(224,224)' --push-layers %s --push-channel=%d --output-prefix=%s" % (os.path.join(currentdir, 'optimize_image.py'), args.caffe_root, args.net_prototxt, args.net_weights, layer, unit, os.path.join(args.outdir, layer, str(unit), 'opt_' + str(unit))))


if __name__ == '__main__':
    main()
