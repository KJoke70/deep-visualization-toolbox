#! /usr/bin/env python2
import matplotlib
matplotlib.use('Agg')

# add parent folder to search path, to enable import of core modules like settings
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import argparse
import numpy as np
import settings
from misc import get_files_list
from misc import mkdir_p

def main():
    
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('file1', type = str, help = 'First file with information to plot')
    parser.add_argument('file2', type = str, help = 'First file with information to plot')
    parser.add_argument('--outfile', type = str, default='comparison.jpg', help = 'First file with information to plot')
    parser.add_argument('--N', type = int, default = 10, help = 'note and save top N activations')
    parser.add_argument('--datadir', type = str, default = settings.static_files_dir, help = 'directory to look for files in')
    parser.add_argument('--outdir', type = str, default = settings.caffevis_outputs_dir, help = 'Which output directory to use. Files are output into outdir/layer/unit_%%04d/{max_histogram}.png')

    args = parser.parse_args()
    assert os.path.exists(args.file1)
    assert os.path.exists(args.file2)
    print args
    print args.file1

if __name__ == '__main__':
    main()
