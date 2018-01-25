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
    
    for l in file1:
        top_n = len(file1[l])
        break

    top_n = args.N if (args.N <= top_n and args.N > 0) else top_n
    
    generate_indices_plot(file1, file2, top_n, image_names, args.outdir)

#TODO split into to functions -> extract_data and make_plot
def generate_indices_plot(data1, data2, top_n, image_names, outdir):
    #overall percentage of equal indices in the top_n, with and without considering order
    nr_tops_ordered = dict()
    nr_tops = dict()
    indices_ordered = dict()
    indices = dict()

    x_axis = np.arange(1, top_n + 1, 1)
    for l in data1:
        nr_tops_ordered[l] = []
        nr_tops[l] = []
        indices_ordered[l] = []
        indices[l] = []
        for n in x_axis:            
            t_ord, idxs_ord = compare_index_single_layer(data1[l], data2[l], True, n, image_names)
            t, idxs = compare_index_single_layer(data1[l], data2[l], False, n, image_names)
            
            indices_ordered[l].append(idxs_ord)
            indices[l].append(idxs)  
            nr_tops_ordered[l].append(t_ord)
            nr_tops[l].append(t)
            

    mkdir_p(outdir)

    # created axes and ylim for plot    
    min_y = 1.0
    max_y = 0.0
    for l in nr_tops:
        temp_min = min(nr_tops_ordered[l] + nr_tops[l])
        temp_max = max(nr_tops_ordered[l] + nr_tops[l])
        if temp_min < min_y:
            min_y = temp_min
        if temp_max > max_y:
            max_y = temp_max

    abs_diff = abs(max_y - min_y)
    min_y -= abs_diff / float(top_n)
    max_y += abs_diff / float(top_n)
    y_axis = np.linspace(min_y, max_y, top_n, endpoint=True)

    
    def plot_data(y_data, filename, title, x_label = 'Top-N', y_label = 'precentage equal', x_data = x_axis, x_axis = x_axis, y_axis = y_axis, min_y = min_y, max_y = max_y):
        plt.clf()        
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
        plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)

    def plot_count_histogram(x_data, y_data1, y_data2, filename, title, legend_1, legend_2, x_label = 'occurrences in top-'+ str(top_n) + ' activations', y_label = 'Unit', best = 50):

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
                
        fontsize2use = 6
        width = 1
        
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 8 * (best/10)))
        ax.barh(x_ind, d1, width, color="blue", alpha = 0.6, label = legend_1)
        ax.barh(x_ind, d2, width, color="red", alpha = 0.6, label = legend_2)
        
        plt.title(title + '\n(best ' + str(best) + ')')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc='upper right')
        
        ax.set_yticks(x_ind + width/2)
        ax.set_yticklabels(x, minor=False, fontsize=fontsize2use)
        
        for i, v in enumerate(d1):
            ax.text(v + 3, i - .3, str(v), fontsize=fontsize2use)
        for i, v in enumerate(d2):
            ax.text(v + 3, i + .1, str(v), fontsize=fontsize2use)
        
        plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)

    for l in nr_tops_ordered:
        plot_data(nr_tops_ordered[l], os.path.join(outdir, l + '_nr_tops_ordered.png'), 'Equal units with considering order\nlayer: ' + l)
        plot_data(nr_tops[l], os.path.join(outdir, l + '_nr_tops.png'), 'Equal units without considering order\nlayer: ' + l)

    # TODO no duplicates?
    count1 = count_indices(data1, top_n)
    count2 = count_indices(data2, top_n)
    
    for l in count1:
        hist_x, count1_y, count2_y = combine_counts(count1[l], count2[l])
        plot_count_histogram(hist_x, count1_y, count2_y, os.path.join(outdir, l + 'count_hist_top_' + str(top_n) + '.png'), 'distribution of activations', 'vgg', 'vgg_flickrlogos')


def count_indices(data, top_n):
    """ 
    Counts the occurences of units in the top_n activations
    data: {layer : { img_idx : [(unit_idx, val), ... ]}}
    returns: {layer : {unit_idx : count, ...}}
    """
    count = dict()
    for layer in data:
        count[layer] = dict()
        for img_idx in data[layer]:
            for unit, val in data[layer][img_idx][:top_n]:
                if unit in count[layer]:
                    count[layer][unit] += 1
                else:
                    count[layer][unit] = 1

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
    return list(x1), list(y1), list(y2)

def compare_counts(count1, count2):
    pass #TODO

def find_top_n(data, top_n = 10):
    """
    data: {layer : {unit_idx : count,...}}
    """
    pass #TODO
    

def compare_index_single_layer(data1, data2, order, top_n, image_names):
    # {img_idx : {img_idx : [(unit_idx, activation_val), ...]}
    percentage_per_image = []
    number_per_image = []
    equal_indices = []
    sums1 = []
    sums2 = []
    avgs1 = []
    avgs2 = []
    percentage_sum = 0
    for img_idx in data1: 
        nr_equal_indices, equals, values1, values2 = compare_index_arrays(data1[img_idx], data2[img_idx], order, top_n)
        sum1, sum2, avg1, avg2 = compare_values(data1[img_idx], data2[img_idx], top_n)

        
        equal_indices.append(equals)
        number_per_image.append(nr_equal_indices)
        percentage = nr_equal_indices / float(top_n)
        percentage_sum += percentage
        percentage_per_image.append(percentage)
    return percentage_sum / float(len(data1)), equal_indices

def compare_values(data1, data2, top_n):
    idxs1, vals1 = zip(*data1)
    idxs2, vals2 = zip(*data2)
    sum1 = sum(vals1[:top_n])
    sum2 = sum(vals2[:top_n])
    avg1 = np.mean(vals1[:top_n])
    avg2 = np.mean(vals2[:top_n])
    return sum1, sum2, avg1, avg2

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
