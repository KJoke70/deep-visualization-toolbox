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

#TODO split into to functions -> extract_data and make_plot
def generate_indices_plot(data1, data2, top_n, image_names, mult_layers, outdir): #TODO multilayer
    #overall percentage of equal indices in the top_n, with and without considering order
    if mult_layers: 
        nr_tops_ordered = dict()
        nr_tops = dict()
        indices_ordered = dict()
        indices = dict()

    else:
        nr_tops_ordered = []
        nr_tops = []
        indices_ordered = []
        indices = []
        
    x_axis = np.arange(1, top_n + 1, 1)
    for i in x_axis:
        if mult_layers:
            for l in data1:
                nr_tops_ordered[l] = []
                nr_tops[l] = []
                indices_ordered[l] = []
                indices[l] = []
                
                t_ord, idxs_ord = compare_index_single_layer(data1[l], data2[l], True, i, image_names)
                t, idxs = compare_index_single_layer(data1[l], data2[l], False, i, image_names)
                indices_ordered[l].append(idxs_ord)
                indices[l].append(idxs)  
                nr_tops_ordered[l].append(t_ord)
                nr_tops[l].append(t)
        else:
            t_ord, idxs_ord = compare_index_single_layer(data1, data2, True, i, image_names)
            t, idxs = compare_index_single_layer(data1, data2, False, i, image_names)
            indices_ordered.append(idxs_ord)
            indices.append(idxs)            
            nr_tops_ordered.append(t_ord)
            nr_tops.append(t)

    mkdir_p(outdir)

    # created axes and ylim for plot    
    if mult_layers:
        min_y = 1.0
        max_y = 0.0
        for l in nr_tops:
            temp_min = min(nr_tops_ordered[l] + nr_tops[l])
            temp_max = max(nr_tops_ordered[l] + nr_tops[l])
            if temp_min < min_y:
                min_y = temp_min
            if temp_max > max_y:
                max_y = temp_max
    else:
        min_y = min(nr_tops_ordered + nr_tops)
        max_y = max(nr_tops_ordered + nr_tops)
    abs_diff = abs(max_y - min_y)
    min_y -= abs_diff / float(top_n)
    max_y += abs_diff / float(top_n)
    y_axis = np.linspace(min_y, max_y, top_n, endpoint=True)

    
    def plot_data(y_data, filename, title, x_label = 'Top-N', y_label = 'precentage equal', x_data = x_axis, x_axis = x_axis, y_axis = y_axis, min_y = min_y, max_y = max_y):
        fig, ax = plt.subplots()
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        plt.title(title)
        plt.xticks(x_axis)
        plt.yticks(y_axis)
        plt.ylim(min_y, max_y)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        plt.plot(x_data, y_data, 'ro')
        for i,j in zip(x_data, y_data):
            ax.annotate("%.3f" % j,xy=(i,j))
        plt.savefig(filename, dpi=300)

    if mult_layers:
        for l in nr_tops_ordered:
            plot_data(nr_tops_ordered[l], os.path.join(outdir, l + '_nr_tops_ordered.png'), 'Equal units with considering order\nlayer: ' + l)
            plot_data(nr_tops[l], os.path.join(outdir, l + '_nr_tops.png'), 'Equal units without considering order\nlayer: ' + l)
    else:
        plot_data(nr_tops_ordered, os.path.join(outdir, 'nr_tops_ordered.png'), 'Equal units with considering order')
        plot_data(nr_tops, os.path.join(outdir, 'nr_tops.png'), 'Equal units without considering order')

    # TODO no duplicates?
    count = count_indices(indices_ordered)
    
    

            
def count_indices(indices):
    """ returns array with [{unit_idx : occurences}, ...] for [0;top_n[ """
    count = []
    for n in xrange(len(indices)):
        count.append(dict())
        for img_idx in xrange(len(indices[n])):
            for key in indices[n][img_idx]:
                if key in count:
                    count[n][key] += 1
                else:
                    count[n][key] = 1
    return count

def compare_index_single_layer(data1, data2, order, top_n, image_names):
    # {img_idx : {img_idx : [(unit_idx, activation_val), ...]}
    percentage_per_image = []
    number_per_image = []
    equal_indices = []
    percentage_sum = 0
    for img_idx in data1: 
#        idxs1, vals1 = zip(*data1[img_idx])
#        idxs2, vals2 = zip(*data2[img_idx])
        # array of indices should not contain duplicates
#        assert not check_if_contains_duplicates(idxs1)
#        assert not check_if_contains_duplicates(idxs2)
        
#        nr_equal_indices, equals = compare_index_arrays(idxs1, idxs2, order, top_n)
        nr_equal_indices, equals, values1, values2 = compare_index_arrays(data1[img_idx], data2[img_idx], order, top_n)
        
        equal_indices.append(equals)
        number_per_image.append(nr_equal_indices)
        percentage = nr_equal_indices / float(top_n)
        percentage_sum += percentage
        percentage_per_image.append(percentage)
    return percentage_sum / float(len(data1)), equal_indices

def compare_index_arrays(data1, data2, order, top_n):
    """ 
        returns 
         1.) the number of equal elements in the first top_n entries of arr1 and arr2
         2.) array of equal indices
    """
    assert len(data1) == len(data2)
    nr_equals = 0
    equals = []
    values1 = []
    values2 = []
    idxs1, vals1 = zip(*data1)
    idxs2, vals2 = zip(*data2)
    assert not check_if_contains_duplicates(idxs1)
    assert not check_if_contains_duplicates(idxs2)
    if not order: 
        for i in xrange(top_n):
            if idxs1[i] in idxs2:
                val2_loc = idxs2.index(idxs1[i])
                nr_equals += 1
                equals.append(idxs1[i])
                values1.append(vals1[i])
                values2.append(vals2[val2_loc])
    else:       
        for i in xrange(top_n):
            if idxs1[i] == idxs2[i]:
                nr_equals += 1
                equals.append(idxs1[i])
                values1.append(vals1[i])
                values2.append(vals2[i])

    return nr_equals, equals, values1, values2
#def compare_index_arrays(idxs1, idxs2, order, top_n):
#    """ 
#    returns 
#     1.) the number of equal elements in the first top_n entries of arr1 and arr2
#     2.) array of equal indices
#    """
#    assert len(idxs1) == len(idxs2)
#    nr_equals = 0
#    equals = []
#    if not order:
#        idxs1 = sorted(idxs1[:top_n])
#        idxs2 = sorted(idxs2[:top_n])
#    for i in xrange(top_n):
#        if idxs1[i] == idxs2[i]:
#            nr_equals += 1
#            equals.append(idxs1[i])
#
#    return nr_equals, equals

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
