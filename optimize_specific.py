#! /usr/bin/env python2
"""
Created on Mon Feb 19 16:20:44 2018

@author: martin
"""

import os
import argparse
import json
import settings

def main():
    parser = argparse.ArgumentParser(description='executes optimize_image.py for specific layers/units')
    parser.add_argument('unit_list',    type = str, default = None, help = 'path to list of units to consider')
    parser.add_argument('--nmt_pkl',      type = str, default = os.path.join(settings.caffevis_outputs_dir, 'find_max_acts_output.pickled'), help = 'Which pickled NetMaxTracker to load.')
    parser.add_argument('--net_prototxt', type = str, default = settings.caffevis_deploy_prototxt, help = 'network prototxt to load')
    parser.add_argument('--net_weights',  type = str, default = settings.caffevis_network_weights, help = 'network weights to load')
    parser.add_argument('--datadir',      type = str, default = settings.static_files_dir, help = 'directory to look for files in')
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
                print layer

if __name__ == '__main__':
    main()
