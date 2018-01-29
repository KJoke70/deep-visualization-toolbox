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
        plot_index_data(extracted_data[l]['percentages_o'], top_n, 'Equal Units (Considering Order)\nLayer: ' + l, os.path.join(outdir, l + '_perc_equal_tops_ordered.png'))
        plot_index_data(extracted_data[l]['percentages_u'], top_n, 'Equal Units\nLayer: ' + l, os.path.join(outdir, l + '_perc_equal_tops_unordered.png'))
        plot_activation_difference(extracted_data[l]['equal_ind_o'], top_n, 'Activations For Equal Units (Considering Order)\nLayer: ' + l, os.path.join(outdir, l + '_top_n_activation_values_ordered.png'))
        plot_activation_difference(extracted_data[l]['equal_ind_u'], top_n, 'Activations For Equal Units\nLayer: ' + l, os.path.join(outdir, l + '_top_n_activation_values_unordered.png'))
        plot_activation_difference2(extracted_data[l]['equal_ind_o'], top_n, 'Activations For Equal Units (Considering Order)\nLayer: ' + l, os.path.join(outdir, l + '_avg_activation_values_ordered.png'))
        plot_activation_difference2(extracted_data[l]['equal_ind_u'], top_n, 'Activations For Equal Units\nLayer: ' + l, os.path.join(outdir, l + '_avg_activation_values_unordered.png'))
    

def plot_index_data(percentages, top_n, title, filename, min_y=0.0, max_y=1.0):

    x_label = 'Top-N'
    y_label = 'Portion Of Equal Indices In Top-N'
    x_axis = np.arange(1, top_n + 1, 1)

    abs_diff = abs(max_y - min_y)
    if min_y > 0.0:
        min_y -= abs_diff / float(top_n)
    if max_y < 1.0:
        max_y += abs_diff / float(top_n)

    y_axis = np.linspace(min_y, max_y, top_n + 1, endpoint=True)


    plt.clf()
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.title(title)
    plt.xticks(x_axis)
    plt.yticks(y_axis)
    plt.ylim(min_y, max_y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.plot(x_axis, percentages, 'ro')
    for i, j in zip(x_axis, percentages): #TODO source
        ax.annotate("%.3f" % j, xy=(i, j))
    plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)

def plot_activation_difference(data, top_n, title, filename, bar_1='vgg', bar_2='vgg_flickrlogos'):

    count = 0
    sum1 = 0
    sum2 = 0
    diff_per_img = dict()
    total_diff = 0
    diffs = []

    for n in xrange(len(data)):
        for img_idx in data[n]:
            diff_per_img[img_idx] = []
            for info in data[n][img_idx]['equals']:
                diff = info[1] - info[2] #vgg - vgg_flickrlogos -> 
                sum1 += info[1]
                sum2 += info[2]
                count += 1
                total_diff += diff
                diff_per_img[img_idx].append(diff)
                diffs.append(diff)

    avg1 = sum1 / count
    avg2 = sum2 / count

    avg_diff = np.mean(diffs)

    d1 = [avg1, 0]
    d2 = [0, avg2]

    x_label = [bar_1, bar_2]
    x_axis = np.arange(len(x_label))
    y_label = 'Average Activation Value'

    width = 1

    plt.clf()
    fig, ax = plt.subplots()

    p1 = ax.bar(x_axis, d1, width, color='blue')
    p2 = ax.bar(x_axis, d2, width, color='red')

    plt.title(title)
    plt.ylabel('Average Activation In Both Network\'s Top-' + str(top_n) + ' Units')
    plt.xlabel('Network\nAverage Diff: %.3f' % avg_diff)
    
    max_y = max(avg1, avg2)
    max_y += 20 - (max_y % 10)
    y_ticks = np.linspace(0, max_y, 11, endpoint=True)
    ax.set_yticks(y_ticks, minor = False)
    ax.set_xticks(x_axis)
    ax.set_xticklabels(x_label, minor=False)
    
    for i, j in enumerate(d1):
        if j != 0:
            ax.annotate("%.3f" % j, xy=(i - 0.5, j + 3))
    for i, j in enumerate(d2):
        if j != 0:
            ax.annotate("%.3f" % j, xy=(i - 0.5, j + 3))
    
    plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)

def plot_activation_difference2(data, top_n, title, filename, bar_1='vgg', bar_2='vgg_flickrlogos'):

    width = 1
    plt.clf()
    fig, ax = plt.subplots()
    plt.title(title)
    plt.ylabel('Average Activation')
    plt.xlabel('Top-N')
    

    x_axis = np.arange(1, top_n + 1, 1)
    x_ticks = np.arange(0, top_n, 1)

    y1_vals = []
    y2_vals = []
    avg_diffs = []


    for i in x_axis:
        count = 0
        sum1 = 0
        sum2 = 0
        total_diff = 0
        for img_idx in data[i - 1]:
            for info in data[i - 1][img_idx]['equals'][:i]:
                diff = info[1] - info[2]
                sum1 += info[1]
                sum2 += info[2]
                count += 1
                total_diff += diff
        y1_vals.append(sum1 / float(count))
        y2_vals.append(sum2 / float(count))
        avg_diffs.append(total_diff / float(count))

    max_y = max(y1_vals + y2_vals)
    max_y += 20 - (max_y % 10)

    y_ticks = np.linspace(0, max_y, 11, endpoint=True)

    ax.set_yticks(y_ticks, minor=False)
    ax.set_xticks(x_ticks + width/2)
    ax.set_xticklabels(x_axis, minor=False)

    plt.legend(loc='upper right')
    if avg_diffs[top_n - 1] < 0:
        p2 = ax.bar(x_ticks, y2_vals, width, color='red', alpha=1)
        p1 = ax.bar(x_ticks, y1_vals, width, color='blue', alpha=1)
        ax.legend((p1[0], p2[0]), (bar_1, bar_2))
    else:
        p1 = ax.bar(x_ticks, y1_vals, width, color='blue', alpha=1)
        p2 = ax.bar(x_ticks, y2_vals, width, color='red', alpha=1)
        ax.legend((p1[0], p2[0]), (bar_1, bar_2))

    
    fontsize2use = 5
    for i, j in enumerate(y1_vals):
        ax.annotate("%.3f" % j, xy=(i - 0.501, j + 1), fontsize=fontsize2use)
    for i, j in enumerate(y2_vals):
        ax.annotate("%.3f" % j, xy=(i - 0.501, j + 1), fontsize=fontsize2use)
    
    plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)

def extract_data(data1, data2, top_n, image_names=None):
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
            perc_u, equal_data_u = compare_indices(data1[l], data2[l], False, n)

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
