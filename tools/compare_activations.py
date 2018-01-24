#! /usr/bin/env python2
import matplotlib
matplotlib.use('Agg')

# add parent folder to search path, to enable import of core modules like settings
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import argparse
import cPickle as pickle
import datetime
import matplotlib.pyplot as plt
import numpy as np
from misc import mkdir_p
from matplotlib.ticker import FormatStrFormatter

def main():
    execution_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('file1', type = str, help = 'First file with information to plot')
    parser.add_argument('file2', type = str, help = 'First file with information to plot')
    parser.add_argument('--N', type = int, default = None, help = 'compare top N, if None all within file will be used')
    parser.add_argument('--order', type = bool, default = True, help = 'consider order within top-N?' )
    parser.add_argument('--image_names', type = str, default = None, help = 'a list with all image filenames in the order they were used' )
    parser.add_argument('--outdir', type = str, default = os.path.join(currentdir, 'results', 'comp-' + execution_time), help = 'First file with information to plot')

    args = parser.parse_args()
    
    assert os.path.exists(args.file1)
    assert os.path.exists(args.file2)
    if not args.image_names == None:
        assert os.path.exists(args.image_names)
        with open(args.image_names) as f:
            image_names = f.readlines()
            image_names = [x.strip() for x in image_names]
    
    
    # pickle either contains {layer : {img_idx : [(unit_idx, activation_val), ...]}}
    # or {img_idx : {img_idx : [(unit_idx, activation_val), ...]}
    file1 = load_pickle(args.file1)
    file2 = load_pickle(args.file2)
    assert len(file1) == len(file2)
    
    # save command line parameters and execution-time in file
    save_execution_data(args, execution_time, os.path.join(args.outdir, 'execution_data.txt'))
    
    # Check if multiple layers need to be compared or just one
    compare_all_layers = False
    
    for l in file1:
        try:    
            int(l)
            top_n = len(file1[l])
            break
        except ValueError:
            compare_all_layers = True
            top_n = len(file1[l])
            break

    top_n = args.N if (args.N <= top_n and args.N > 0) else top_n
    
    generate_indices_plot(file1, file2, top_n, image_names, compare_all_layers, args.outdir)


def generate_indices_plot(data1, data2, top_n, image_names, mult_layers, outdir):
    #overall percentage of equal indices in the top_n, with and without considering order
    tops_order = []
    tops_no_order= []
    x_axis = np.arange(1, top_n + 1, 1)
    for i in x_axis:
        if mult_layers:
            # returns arrays of plots #TODO
            #tops_order.append(compare_index_multiple_layer(data1, data2, True, top_n, image_names))
            #tops_no_order.append(compare_index_multiple_layer(data1, data2, False, top_n, image_names))
            pass #TODO
        else:
            tops_order.append(compare_index_single_layer(data1, data2, True, i, image_names))
            tops_no_order.append(compare_index_single_layer(data1, data2, False, i, image_names))
    mkdir_p(outdir)
    
    min_y = min(tops_order + tops_no_order)
    max_y = max(tops_order + tops_no_order)
    abs_diff = abs(max_y - min_y)
    min_y -= abs_diff / float(top_n)
    max_y += abs_diff / float(top_n)
    y_axis = np.linspace(min_y, max_y, top_n, endpoint=True)
    
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.title('Equal units with order')
    plt.xticks(x_axis)
    plt.yticks(y_axis)
    plt.ylim(min_y, max_y)
    plt.xlabel('Top-N')
    plt.ylabel('precentage equal')
    plt.grid(True)
    plt.plot(x_axis, tops_order, 'ro')
    for i,j in zip(x_axis, tops_order):
        ax.annotate("%.3f" % j,xy=(i,j))
    plt.savefig(os.path.join(outdir, 'tops_order.png'))
    
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.title('Equal units without order')
    plt.xticks(x_axis)
    plt.yticks(y_axis)
    plt.ylim(min_y, max_y)
    plt.xlabel('Top-N')
    plt.ylabel('precentage equal')
    plt.grid(True)
    plt.plot(x_axis, tops_no_order, 'ro')
    for i,j in zip(x_axis, tops_no_order):
        ax.annotate("%.3f" % j,xy=(i,j))
    plt.savefig(os.path.join(outdir, 'tops_no_order.png'))
    
    #f2 = plt.figure(2)
    #plt.plot(x_axis, tops_no_order)
    #plt.savefig(os.path.join(outdir, 'top_no_order.png'))
#    plot_order = plt.plot(x_axis, tops_order)
#    plot_no_order = plt.plot(x_axis, tops_no_order)
    
    

def compare_index_multiple_layers(data1, data2, order, top_n, image_names):
    # {layer : {img_idx : [(unit_idx, activation_val), ...]}}
    pass #TODO

def compare_index_single_layer(data1, data2, order, top_n, image_names):
    # {img_idx : {img_idx : [(unit_idx, activation_val), ...]}
    percentage_per_image = []
    number_per_image = []
    percentage_sum = 0
    for img_idx in data1: 
        idxs1, vals1 = zip(*data1[img_idx])
        idxs2, vals2 = zip(*data2[img_idx])
        # array of indices should not contain duplicates
        assert not check_if_contains_duplicates(idxs1)
        assert not check_if_contains_duplicates(idxs2)
        equal_indices = compare_index_arrays(idxs1, idxs2, order, top_n)
        number_per_image.append(equal_indices)
        percentage = equal_indices / float(top_n)
        percentage_sum += percentage
        percentage_per_image.append(percentage)
    return percentage_sum / float(len(data1))

def compare_index_arrays(idxs1, idxs2, order, top_n):
    """ returns the number of equal elements in the first top_n entries of arr1 and arr2 """
    assert len(idxs1) == len(idxs2)
    equals = 0
    if not order:
        idxs1 = sorted(idxs1[:top_n])
        idxs2 = sorted(idxs2[:top_n])
    for i in xrange(top_n):
        if idxs1[i] == idxs2[i]:
            equals += 1

    return equals

def check_if_contains_duplicates(arr):
    return len(arr) != len(set(arr))

def load_pickle(filename):
    f = open(filename)
    data = pickle.load(f)
    f.close()
    return data

def save_execution_data(args, time, filename):
    dirname = os.path.dirname(filename)
    mkdir_p(dirname)
    
    with open(filename, 'wt') as f:
        f.write("date: %s\n" % time)
        for k in args.__dict__:
            f.write( "%s: %s\n" % (k, args.__dict__[k]))
        f.close()

if __name__ == '__main__':
    main()
