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
    # parser.add_argument('--filelist',     type = str, default = settings.static_files_input_file, help = 'List of image files to consider, one per line. Must be the same filelist used to produce the NetMaxTracker!')
    parser.add_argument('--outdir',       type = str, default = settings.caffevis_outputs_dir, help = 'Which output directory to use. Files are output into outdir/layer/unit_%%04d/{maxes,deconv,backprop}_%%03d.png')
  
    parser.add_argument('-il', '--ignore_layers', nargs='*', default = [], help = 'ignore these layers')

    args = parser.parse_args()

    if args.unit_list != None:
        assert os.path.exists(args.unit_list)
        with open(args.unit_list, 'rt') as f:
            unit_list = json.load(f)

    config1 = os.path.join(currentdir, 'optimize_image.py') + " --decay 0 --blur-radius 0.5 --blur-every 4 --small-norm-percentile 50 --max-iter 500 --lr-policy progress --lr-params \"{'max_lr': 100.0, 'desired_prog': 2.0}\""
    config2 = os.path.join(currentdir, 'optimize_image.py') + " --decay 0.3 --blur-radius 0 --blur-every 0 --small-norm-percentile 20 --max-iter 750 --lr-policy constant --lr-params \"{'lr': 100.0}\""
    config3 = os.path.join(currentdir, 'optimize_image.py') + " --decay 0.0001 --blur-radius 1.0 --blur-every 4 --max-iter 1000 --lr-policy constant --lr-params \"{'lr': 100.0}\""
    config4 = os.path.join(currentdir, 'optimize_image.py') + " --decay 0 --blur-radius 0.5 --blur-every 4 --px-abs-benefit-percentile 90 --max-iter 1000 --lr-policy progress --lr-params \"{'max_lr': 100000000, 'desired_prog': 2.0}\""
    
    for layer in unit_list:
            if not os.path.exists(os.path.join(args.outdir, layer)) and not layer in args.ignore_layers:
                units = set(unit_list[layer])
                for unit in units:
                    os.system("%s  --caffe-root=%s --deploy-proto=%s --net-weights=%s --data-size='(224,224)' --push-layers %s --push-channel=%d --output-prefix=%s" % (config1, args.caffe_root, args.net_prototxt, args.net_weights, layer, unit, os.path.join(args.outdir, layer, str(unit), 'opt_c1_' + str(unit))))
                    os.system("%s  --caffe-root=%s --deploy-proto=%s --net-weights=%s --data-size='(224,224)' --push-layers %s --push-channel=%d --output-prefix=%s" % (config2, args.caffe_root, args.net_prototxt, args.net_weights, layer, unit, os.path.join(args.outdir, layer, str(unit), 'opt_c2_' + str(unit))))
                    os.system("%s  --caffe-root=%s --deploy-proto=%s --net-weights=%s --data-size='(224,224)' --push-layers %s --push-channel=%d --output-prefix=%s" % (config3, args.caffe_root, args.net_prototxt, args.net_weights, layer, unit, os.path.join(args.outdir, layer, str(unit), 'opt_c3_' + str(unit))))
                    os.system("%s  --caffe-root=%s --deploy-proto=%s --net-weights=%s --data-size='(224,224)' --push-layers %s --push-channel=%d --output-prefix=%s" % (config4, args.caffe_root, args.net_prototxt, args.net_weights, layer, unit, os.path.join(args.outdir, layer, str(unit), 'opt_c4_' + str(unit))))
                    # os.system("%s --decay 0.0001 --blur-radius 1.0 --blur-every 4 --lr-policy constant --caffe-root=%s --deploy-proto=%s --net-weights=%s --data-size='(224,224)' --push-layers %s --push-channel=%d --output-prefix=%s" % (os.path.join(currentdir, 'optimize_image.py'), args.caffe_root, args.net_prototxt, args.net_weights, layer, unit, os.path.join(args.outdir, layer, str(unit), 'opt_' + str(unit))))


if __name__ == '__main__':
    main()
