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
from find_maxes.find_max_acts import pickle_to_text
from matplotlib.ticker import FormatStrFormatter

error_msgs = []

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

    if 'fc8_flickrlogos' in file2.keys():
        file2['fc8'] = file2.pop('fc8_flickrlogos')
    
    assert len(file1) == len(file2)

    min_n = 1000000
    for l in file1:
        for img_idx in file1[l]:
            if len(file1[l][img_idx]) < min_n:
                min_n = len(file1[l][img_idx])
            if len(file2[l][img_idx]) < min_n:
                min_n = len(file2[l][img_idx])
    top_n = min_n

    if not (args.N <= top_n and args.N > 0):
        print_error('N==%d not possible, using N==%d' % (args.N, top_n))
    top_n = args.N if (args.N <= top_n and args.N > 0) else top_n

    extracted_data = extract_data(file1, file2, top_n)
    evaluate_data(extracted_data, top_n, args.outdir)

    # save extracted_data as pickled-file
    save_pickle(extracted_data, os.path.join(args.outdir, 'extracted_data.pickled'))
    # save command line parameters and execution-time in file
    save_execution_data(args, top_n, execution_time, os.path.join(args.outdir, 'execution_data.txt'))

def evaluate_data(extracted_data, top_n, outdir):
    for l in extracted_data:
        plot_index_data(extracted_data[l]['percentages_o'], top_n, 'Equal Units (Considering Order)\nLayer: ' + l, os.path.join(outdir, l, l + '_perc_equal_tops_ordered.png'))
        plot_index_data(extracted_data[l]['percentages_u'], top_n, 'Equal Units\nLayer: ' + l, os.path.join(outdir, l, l + '_perc_equal_tops_unordered.png'))
        plot_activation_difference(extracted_data[l]['equal_ind_o'], top_n, 'Activations For Equal Units (Considering Order)\nLayer: ' + l, os.path.join(outdir, l, l + '_avg_activation_values_ordered.png'))
        plot_activation_difference(extracted_data[l]['equal_ind_u'], top_n, 'Activations For Equal Units\nLayer: ' + l, os.path.join(outdir, l, l + '_avg_activation_values_unordered.png'))
        plot_count_occurences(extracted_data[l]['combined_counts'], top_n, 'Distribution Of Activations\nLayer: ' + l, os.path.join(outdir, l, l + '_count_hist_top_' + str(top_n) + '.png'))

def plot_index_data(percentages, top_n, title, filename, min_y=0.0, max_y=1.0):
    """
    plots a graph to show what portion of unit indices remain the same within the top-N activations of both networks
    """

    dirname = os.path.dirname(filename)
    mkdir_p(dirname)

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
    fig, ax = plt.subplots(figsize=(8 * (top_n/10), 10))
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
    plt.close()

def plot_activation_difference(data, top_n, title, filename, bar_1='vgg', bar_2='vgg_flickrlogos'):
    """
    plots a bar chart comparing the average activation values of the 2 networks
    """

    dirname = os.path.dirname(filename)
    mkdir_p(dirname)

    width = 1
    plt.clf()
    fig, ax = plt.subplots(figsize=(8 * (top_n/10), 10))
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
        if count > 0:
            y1_vals.append(sum1 / float(count))
            y2_vals.append(sum2 / float(count))
            avg_diffs.append(total_diff / float(count))
        else:
            y1_vals.append(0)
            y2_vals.append(0)
            avg_diffs.append(0)

    max_y = max(y1_vals + y2_vals)
    max_y += 20 - (max_y % 10)

    y_ticks = np.linspace(0, max_y, 11, endpoint=True)

    ax.set_yticks(y_ticks, minor=False)
    ax.set_xticks(x_ticks + width/2)

    ax.set_xticklabels(x_axis, minor=False)

    if avg_diffs[top_n - 1] < 0:
        p2 = ax.bar(x_ticks, y2_vals, width, color='red', edgecolor='white', label = bar_2, alpha=0.5)
        p1 = ax.bar(x_ticks, y1_vals, width, color='blue', edgecolor='white', label = bar_1, alpha=0.5)
    else:
        p1 = ax.bar(x_ticks, y1_vals, width, color='blue', edgecolor='white', label = bar_1, alpha=0.5)
        p2 = ax.bar(x_ticks, y2_vals, width, color='red', edgecolor='white', label = bar_2, alpha=0.5)
    plt.legend(loc='upper right')
    
    for i, j in enumerate(y1_vals):
        ax.annotate(' %.3f' % j, xy=(i - 0.501, j), fontsize=8)
    for i, j in enumerate(y2_vals):
        ax.annotate(' %.3f' % j, xy=(i - 0.501, j), fontsize=8)
    
    for i, j in enumerate(y1_vals):
        if float("%.3f" % avg_diffs[i]) != 0.000:
            ax.annotate(' %.3f' % avg_diffs[i], xy=(i - 0.501, 0), fontsize=7, color='white')
    
    plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)
    plt.close()

#TODO check if correct
def plot_count_occurences(data, top_n, title, filename, legend_1='vgg', legend_2='vgg_flickrlogos', best=20):
    """
    plots a bar-chart showing how often specific units appear within the top-n activations
    """

    dirname = os.path.dirname(filename)
    mkdir_p(dirname)

    x_data, y_data1, y_data2 = data
    # https://stackoverflow.com/questions/13070461/get-index-of-the-top-n-values-of-a-list-in-python
    best_n_indices = sorted(range(len(y_data1)), key=lambda i: y_data1[i], reverse=True)[:best] + sorted(range(len(y_data2)), key=lambda i: y_data2[i], reverse=True)[:best]
    best_n_indices = sorted(list(set(best_n_indices)))
    
    x = []
    d1 = []
    d2 = []
    
    for ind in best_n_indices:
        x.append(str(x_data[ind]))
        d1.append(y_data1[ind])
        d2.append(y_data2[ind])
    x_ind = np.arange(len(x))
            
    fontsize = 6
    width = 1

    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 8 * (best/10)))
    ax.barh(x_ind, d1, width, color="blue", alpha = 0.5, label = legend_1, edgecolor='white')
    ax.barh(x_ind, d2, width, color="red", alpha = 0.5, label = legend_2, edgecolor='white')
    
    plt.title(title + '\n(best ' + str(best) + ')')
    plt.xlabel('Occurrences in top-'+ str(top_n) + ' activations')
    plt.ylabel('Unit')
    plt.legend(loc='upper right')
    
    ax.set_yticks(x_ind + width/2)
    ax.set_yticklabels(x, minor=False, fontsize=fontsize)
    
    for i, v in enumerate(d1):
        ax.text(v + 3, i - .3, str(v), fontsize=fontsize)
    for i, v in enumerate(d2):
        ax.text(v + 3, i + .1, str(v), fontsize=fontsize)
    
    plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)
    plt.close()

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

        count1 = count_indices(data1[l], top_n)
        count2 = count_indices(data2[l], top_n)


        result[l]['percentages_o'] = percentages_ordered
        result[l]['percentages_u'] = percentages_unordered
        result[l]['equal_ind_o'] = equal_data_ordered
        result[l]['equal_ind_u'] = equal_data_unordered
        result[l]['combined_counts'] = combine_counts(count1, count2)

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
        nr_equal_indices, equals, values1, values2 = compare_index_arrays(data1[img_idx][:top_n], data2[img_idx][:top_n], order, top_n)

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
    idxs1, vals1 = zip(*data1[:top_n])
    idxs2, vals2 = zip(*data2[:top_n])
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

def count_indices(data, top_n):
    """
    Counts the occurences of units in the top_n activations
    data: { img_idx : [(unit_idx, val), ... ]}
    returns: {unit_idx : count, ...}
    """
    count = dict()
    for img_idx in data:
        for unit, val in data[img_idx][:top_n]:
            if unit in count:
                count[unit] += 1
            else:
                count[unit] = 1

    return count

def combine_counts(count1, count2):
    """
    inserts {unit_idx : 0} in count1 and count2 if unit_idx is missing
    returns list of unit_idx and corresponding count values
    """
    for key in count1:
        if key not in count2:
            count2[key] = 0
            
    for key in count2:
        if key not in count1:
            count1[key] = 0

    x_data1, y_data1 = zip(*count1.items())
    x_data2, y_data2 = zip(*count2.items())
    
    comb1 = [(x,_) for _,x in sorted(zip(x_data1,y_data1))]
    comb2 = [(x,_) for _,x in sorted(zip(x_data2,y_data2))]
    
    
    y1, x1 = zip(*comb1)
    y2, x2 = zip(*comb2)    
    
    assert x1 == x2
    return (list(x1), list(y1), list(y2))
#------------------------------------------------------------------------------------------------------------------------------------
def print_error(msg):
    print msg
    error_msgs.append(msg)

def check_if_contains_duplicates(arr):
    return len(arr) != len(set(arr))

def load_pickle(filename):
    f = open(filename)
    data = pickle.load(f)
    f.close()
    return data

def save_pickle(data, filename):
    dirname = os.path.dirname(filename)
    mkdir_p(dirname)

    with open(filename, 'wb') as ff:
        pickle.dump(data, ff, -1)
    pickle_to_text(filename)

def save_execution_data(args, top_n, current_time, filename):
    dirname = os.path.dirname(filename)
    mkdir_p(dirname)

    with open(filename, 'wt') as f:
        f.write("date: %s\n" % current_time)
        for k in args.__dict__:
            f.write( "%s: %s\n" % (k, args.__dict__[k]))
        f.write("used N: %d" % top_n)
        f.write("\n")
        f.write("-"*30)
        f.write("\nerrors:\n")
        for i in xrange(len(error_msgs)):
            f.write("%s\n" % error_msgs[i])
        f.close()

if __name__ == '__main__':
    main()
