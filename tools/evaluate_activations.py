#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:03:44 2018

@author: martin
"""

#! /usr/bin/env python2
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


    for l in file1:
        top_n = len(file1[l])
        break

    top_n = args.N if (args.N <= top_n and args.N > 0) else top_n

    extracted_data = extract_data(file1, file2, top_n)
    evaluate_data(extracted_data, top_n, args.outdir)

def evaluate_data(extracted_data, top_n, outdir):
    for l in extracted_data:
        plot_index_data(extracted_data[l]['percentages_o'], top_n, 'Equal units with considering order\nlayer: ' + l, os.path.join(outdir, l + '_nr_tops_ordered.png'))
        plot_index_data(extracted_data[l]['percentages_u'], top_n, 'Equal units without considering order\nlayer: ' + l, os.path.join(outdir, l + '_nr_tops_unordered.png'))
    pass #TODO

def plot_index_data(percentages, top_n, title, filename, min_y = 0.0, max_y = 1.0):

    x_label = 'Top-N'
    y_label = 'Portion of equal indices in Top-N'
    x_axis = np.arange(1, top_n + 1, 1)

    y_axis = np.linspace(min_y, max_y, top_n, endpoint=True)

    plt.clf()
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.title(title)
    plt.xticks(x_axis)
    plt.yticks(y_axis)
    plt.ylim(min_y, max_y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.plot(x_axis, percentages, 'ro')
    for i,j in zip(x_axis, percentages):
        ax.annotate("%.3f" % j,xy=(i,j))
    plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)

def plot_activation_difference(data, filename):
    pass #TODO

def extract_data(data1, data2, top_n, image_names = None):
    """
    data1, data2: {layer : {img_idx : [(unit_idx, activation_value), ...], ...}, ...}
    top_n: extract data for best top_n. Data must be pre-sorted
    image_names = None: currently no function #TODO

    returns a dictionary with all extraced data:
        percentage of equal indices (considering top-N order and without)
        unit-indices contained in the top-N in data1 and data2 (considering top-N order and without)
    """

    result = dict()


    for l in data1:
        percentages_ordered = []
        percentages_unordered = []
        equal_data_ordered = []
        equal_data_unordered = []
        result[l] = dict()

        for n in xrange(1, top_n + 1):
            perc_o, equal_data_o = compare_indices(data1[l], data2[l], True, n)
            perc_u, equal_data_u = compare_indices(data1[l], data2[l], False , n)

            percentages_ordered.append(perc_o)
            percentages_unordered.append(perc_u)
            equal_data_ordered.append(equal_data_o)
            equal_data_unordered.append(equal_data_u)

        result[l]['percentages_o'] = percentages_ordered
        result[l]['percentages_u'] = percentages_unordered
        result[l]['equal_ind_o'] = equal_data_ordered
        result[l]['equal_ind_u'] = equal_data_unordered

    return result


def compare_indices(data1, data2, order, top_n):
    """
    data1, data2: {img_idx : [(unit_idx, activation_value), ...],...}
    returns:
        - the average percentage of equal indices within the top-n ind data1
          and data2 depending on order (order = True: data1[idx] == data2[idx];
          order = False: data1[idx] in data2)
        - a dict containing the equal indices and corresponding values in data1 and data2
    """
    percentage_sum = 0
    equal_data = dict()
    for img_idx in data1:

        equal_data[img_idx] = dict()
        nr_equal_indices, equals, values1, values2 = compare_index_arrays(data1[img_idx], data2[img_idx], order, top_n)

        equal_data[img_idx]['number_equal'] = nr_equal_indices
        equal_data[img_idx]['equals'] = zip(equals, values1, values2) # -> (unit_idx, vals1, vals2)

        percentage = nr_equal_indices / float(top_n)
        percentage_sum += percentage
    return percentage_sum / float(len(data1)), equal_data

def compare_index_arrays(data1, data2, order, top_n):
    """
    data1, data2: [(unit_idx, value), ...]
        returns
         1.) the number of equal elements in the first top_n entries of arr1 and arr2
         2.) array of equal indices
         3.) the values corresponding to the equals in data1
         4.) the values corresponding to the equals in data2
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

#------------------------------------------------------------------------------------------------------------------------------------

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
