#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:03:44 2018

@author: martin
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# add parent folder to search path, to enable import of core modules like settings
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import argparse
import cPickle as pickle
import datetime
import logging
import json
import csv

#import copy

import numpy as np
from misc import mkdir_p
#from find_maxes.find_max_acts import pickle_to_text
from matplotlib.ticker import FormatStrFormatter

def main():
    execution_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('file1', type = str, help = 'First file with information to plot')
    parser.add_argument('file2', nargs='+', help = 'Second file with information to plot. Multiple files for comparison over time.')
    parser.add_argument('--N', type = int, default = None, help = 'compare top N, if None all within file will be used')
    parser.add_argument('--image_names', type = str, default = None, help = 'a list with all image filenames in the order they were used' )
    parser.add_argument('--outdir', type = str, default = os.path.join(currentdir, 'results', 'comp-' + execution_time), help = 'First file with information to plot')
    parser.add_argument('--iter_step', type = int, default = 300, help = 'iteration steps when multiple files2 are used')
    parser.add_argument('--save_pickled', type = bool, default = False, help = 'Whether to save extracted data as *.pickled. No pickled is saved by default')
    parser.add_argument('--save_json', type = bool, default = True, help = 'Whether to save Interesting units as json. Default is True')
    parser.add_argument('--no_eval', type = bool, default = False, help = 'Don\'t create plots if True')
    parser.add_argument('--compile_csv', type = bool, default = True, help = 'Save progress over iterations as csv table. Does nothing if len(file2) == 1')

    args = parser.parse_args()

    assert os.path.exists(args.file1)
    if len(args.file2) > 1:
        for p in args.file2:
            assert os.path.exists(p)
    if not args.image_names == None:
        assert os.path.exists(args.image_names)
        with open(args.image_names) as f:
            image_names = f.readlines()
            image_names = [x.strip() for x in image_names]

    mkdir_p(args.outdir)
    logging.basicConfig(filename=os.path.join(args.outdir, 'execution_log.log'), level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('START TIME: %s' % execution_time)
    logging.info('file1=%s' % args.file1)
    if len(args.file2) > 1:
        logging.info('number of files2: %d' % len(args.file2))
    for i in xrange(len(args.file2)):
        logging.info('files2[%i]=%s' % (i, args.file2[i]))
    logging.info('N=%d', args.N)
    logging.info('image_names=%s' % args.image_names)
    logging.info('outdir=%s' % args.outdir)
    logging.info('iter_step=%d' % args.iter_step)

    # pickle either contains {layer : {img_idx : [(unit_idx, activation_val), ...]}}
    # or {img_idx : {img_idx : [(unit_idx, activation_val), ...]}
    file1 = load_pickle(args.file1)
    files2 = []
    for p in args.file2:
        files2.append(load_pickle(p))

    # rename folders to match first file
    # TODO make configurable
    for i in xrange(len(files2)):
        if 'fc8_flickrlogos' in files2[i].keys():
            files2[i]['fc8'] = files2[i].pop('fc8_flickrlogos')
            assert len(file1) == len(files2[i])

    min_n = 1000000
    for l in file1:
        for img_idx in file1[l]:
            if len(file1[l][img_idx]) < min_n:
                min_n = len(file1[l][img_idx])

    for i in xrange(len(files2)):
        for l in files2[i]:
            for img_idx in files2[i][l]:
                if len(files2[i][l][img_idx]) < min_n:
                    min_n = len(files2[i][l][img_idx])
    top_n = min_n

    if not (args.N <= top_n and args.N > 0):
        logging.warning('N==%d not possible, using N==%d' % (args.N, top_n))
    top_n = args.N if (args.N <= top_n and args.N > 0) else top_n

    if len(files2) > 1:
        extracted_data = []
        interesting_units = []
        for i in xrange(len(files2)):
            logging.debug('extract_data: for iteration %d' % (i * args.iter_step))
            e_data, int_u = extract_data(file1, files2[i], top_n)
            extracted_data.append(e_data)
            interesting_units.append(int_u) #deepcopy?
    else:
        extracted_data, interesting_units = extract_data(file1, files2[0], top_n)
    if not args.no_eval:
        evaluate_data(extracted_data, top_n, args.outdir, args.iter_step)
    
    if len(files2) > 1 and bool(args.compile_csv):
        compile_csv(extracted_data, top_n, args.outdir, args.iter_step)

    # save extracted_data as pickled-file
    if args.save_pickled:
        save_pickle(extracted_data, os.path.join(args.outdir, 'extracted_data.pickled'))

    # save interesting units in json
    if args.save_json:
        if type(interesting_units) is list:
            for i in xrange(len(interesting_units)):
                save_json(interesting_units, os.path.join(args.outdir, str(i) + '_interesting_units.json'))
        else:
            save_json(interesting_units, os.path.join(args.outdir, 'interesting_units.json'))

def evaluate_data(extracted_data, top_n, outdir, iter_step=None):
    logging.debug('evaluate_data: start...')
    if type(extracted_data) == type(list()):
        layers = []
        for i in xrange(len(extracted_data)):
            for l in extracted_data[i]:
                layers.append(l)
            break

        assert iter_step != None
        logging.debug('evaluate_data: multiple-file-comparison')
        for l in layers:
            perc_data_o = retreive_percentage_data(extracted_data, l, 'percentages_o', top_n)
            perc_data_u = retreive_percentage_data(extracted_data, l, 'percentages_u', top_n)
            plot_index_data_per_iter(perc_data_o, top_n, 'Equal Units over time (Considering Order)\nLayer: ' + l, os.path.join(outdir, l + '_top' + str(top_n) + 'perc_equal_per_iters_ordered.png'), iter_step)
            plot_index_data_per_iter(perc_data_u, top_n, 'Equal Units over time\nLayer: ' + l, os.path.join(outdir, l + '_top' + str(top_n) + '_perc_equal_per_iters_unordered.png'), iter_step)
    else:
        logging.debug('evaluate_data: single-file-comparison')
        for l in extracted_data:
            logging.debug('evaluate_data: for layer %s' % l)
            plot_index_data(extracted_data[l]['percentages_o'], top_n, 'Equal Units (Considering Order)\nLayer: ' + l, os.path.join(outdir, l, l + '_perc_equal_tops_ordered.png'))
            plot_index_data(extracted_data[l]['percentages_u'], top_n, 'Equal Units\nLayer: ' + l, os.path.join(outdir, l, l + '_perc_equal_tops_unordered.png'))
            plot_activation_difference(extracted_data[l]['equal_ind_o'], top_n, 'Activations For Equal Units (Considering Order)\nLayer: ' + l, os.path.join(outdir, l, l + '_avg_activation_values_ordered.png'))
            plot_activation_difference(extracted_data[l]['equal_ind_u'], top_n, 'Activations For Equal Units\nLayer: ' + l, os.path.join(outdir, l, l + '_avg_activation_values_unordered.png'))
            plot_activation_difference_curve(extracted_data[l]['equal_ind_o'], top_n, 'Activation-Difference For Equal Units (Considering Order)\nLayer: ' + l, os.path.join(outdir, l, l + '_avg_activation_diffs_ordered.png'))
            plot_activation_difference_curve(extracted_data[l]['equal_ind_u'], top_n, 'Activation-Difference For Equal Units\nLayer: ' + l, os.path.join(outdir, l, l + '_avg_activation_diffs_unordered.png'))
            plot_activation_averages(extracted_data[l]['averages'], top_n, 'Activation Averages\nLayer: ' + l, os.path.join(outdir, l, l + '_avg_activation_averages.png'))

            for n in xrange(top_n):
                plot_count_occurences(extracted_data[l]['combined_counts'][n], n + 1, 'Distribution Of Activations\nLayer: ' + l, os.path.join(outdir, l, l + '_count_hist_top_' + str(n + 1) + '.png'))
    logging.debug('evaluate_data: end.')

def plot_index_data_per_iter(percentages, top_n, title, filename, iter_step, min_y=0.0, max_y=1.0):
    """
    plots a graph to show what portion of unit indices remain the same within the top-N activations of both networks
    """
    logging.debug('plot_index_data_per_iter: start...')
    logging.debug('plot_index_data_per_iter: top-n=%d; title=%s; filename=%s; iter_step=%d min_y=%f; max_y=%f' % (top_n, title, filename, iter_step, min_y, max_y))

    dirname = os.path.dirname(filename)
    mkdir_p(dirname)
    
    x_label = 'Iterations'
    y_label = 'Portion Of Equal Indices In Top-' + str(top_n)
    x_axis = np.arange(iter_step, iter_step * len(percentages) + 1, iter_step)

    max_y = max(percentages) + 0.1
    min_y = min(percentages) - 0.1

    
    plt.clf()
    fig, ax = plt.subplots(figsize=(8 * (top_n/10), 12))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.title(title)
    plt.xticks(x_axis)
    plt.ylim(min_y, max_y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)

    plt.plot(x_axis, percentages, 'o')
    for i, j in zip(x_axis, percentages): #TODO source
        ax.annotate("%.3f" % j, xy=(i, j))

    plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    logging.debug('plot_index_data_per_iter: end')

def plot_index_data(percentages, top_n, title, filename, min_y=0.0, max_y=1.0):
    """
    plots a graph to show what portion of unit indices remain the same within the top-N activations of both networks
    """

    logging.debug('plot_index_data: start...')
    logging.debug('plot_index_data: top-n=%d; title=%s; filename=%s; min_y=%f; max_y=%f' % (top_n, title, filename, min_y, max_y))
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
    logging.debug('plot_index_data: end')

def plot_activation_difference(data, top_n, title, filename, bar_1='vgg', bar_2='vgg_flickrlogos'):
    """
    plots a bar chart comparing the average activation values of the 2 (same indices) networks
    """

    logging.debug('plot_activation_difference: start...')
    logging.debug('plot_activation_difference: top-n=%d; title=%s, filename=%s; bar_1=%s; bar_2=%s' % (top_n, title, filename, bar_1, bar_2))
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
                diff = info[2] - info[1]
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
    logging.debug('plot_activation_difference: end')

def plot_activation_difference_curve(data, top_n, title, filename, bar_1='vgg', bar_2='vgg_flickrlogos'):
    """
    plots a curve comparing the average activation values of the 2 networks
    """

    logging.debug('plot_activation_difference_curve: start')
    logging.debug('plot_activation_difference_curve: top-n=%d; title=%s, filename=%s; bar_1=%s; bar_2=%s' % (top_n, title, filename, bar_1, bar_2))
    dirname = os.path.dirname(filename)
    mkdir_p(dirname)

    width = 1
    plt.clf()
    fig, ax = plt.subplots(figsize=(7 * (top_n/10), 10))
    plt.title(title)
    plt.ylabel('Average Difference')
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
                diff = info[2] - info[1]
                sum1 += info[1]
                sum2 += info[2]
                count += 1
                total_diff += diff
        if count > 0:
            avg_diffs.append(total_diff / float(count))
        else:
            avg_diffs.append(0)

    max_y = max(avg_diffs)
    max_y = 0 if max_y < 0 else max_y + 5 - (max_y % 5)
    min_y = min(avg_diffs)
    min_y -= min_y % 5

    y_ticks = np.linspace(min_y, max_y, 11, endpoint=True)
    plt.ylim(min_y, max_y)

    ax.set_yticks(y_ticks, minor=True)

    ax.set_xticks(x_ticks + width/2)
    ax.set_xticklabels(x_axis, minor=False)

    p = ax.plot(x_ticks, avg_diffs, '.r-', label='Difference %s - %s' % (bar_2, bar_1))
    plt.legend(loc='upper right')
    
    for i, j in enumerate(avg_diffs):
        ax.annotate('%.3f' % j, xy=(i - 0.501, j + 0.01), fontsize=8)
    
    plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    logging.debug('plot_activation_difference_curve: end.')

def plot_activation_averages(data, top_n, title, filename, bar_1='vgg', bar_2='vgg_flickrlogos'):
    """
    plots a bar chart comparing the average activation values of the 2 networks
    """

    logging.debug('plot_activation_averages: start...')
    logging.debug('plot_activation_averages: top-n=%d; title=%s, filename=%s; bar_1=%s; bar_2=%s' % (top_n, title, filename, bar_1, bar_2))
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

    y1_vals = data[0]
    y2_vals = data[1]
    avg_diffs = []

    for i in xrange(top_n):
        diff = y1_vals[i] - y2_vals[i]
        avg_diffs.append(diff)

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
    logging.debug('plot_activation_averages: end')
#TODO check if correct
def plot_count_occurences(data, top_n, title, filename, legend_1='vgg', legend_2='vgg_flickrlogos', best=20):
    """
    plots a bar-chart showing how often specific units appear within the top-n activations
    """

    logging.debug('plot_count_occurences: start...')
    logging.debug('plot_count_occurences: top-n=%d; title=%s, filename=%s; legend_1:%s; legend_2:%s; best=%d' % (top_n, title, filename, legend_1, legend_2, best))
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
    logging.debug('plot_count_occurences: end.')

def extract_data(data1, data2, top_n, image_names=None):
    """
    data1, data2: {layer : {img_idx : [(unit_idx, activation_value), ...], ...}, ...}
    top_n: extract data for best top_n. Data must be pre-sorted
    image_names = None: currently no function #TODO

    returns a dictionary with all extraced data:
        percentage of equal indices (considering top-N order and without)
        unit-indices contained in the top-N in data1 and data2 (considering top-N order and without)
    """
    logging.debug('extract_data: starting...')
    result = dict()
    interesting_units = dict()

    for l in data1:
        logging.debug('extract_data: for layer %s' % l)
        percentages_ordered = []
        percentages_unordered = []
        equal_data_ordered = []
        equal_data_unordered = []
        result[l] = dict()
        interesting_units[l] = []

        for n in xrange(1, top_n + 1):
            logging.debug('extract_data: percentages and equal indices for top-%d for layer %s' % (n, l))
            perc_o, equal_data_o = compare_indices(data1[l], data2[l], True, n)
            perc_u, equal_data_u = compare_indices(data1[l], data2[l], False, n)
            

            percentages_ordered.append(perc_o)
            percentages_unordered.append(perc_u)
            equal_data_ordered.append(equal_data_o)
            equal_data_unordered.append(equal_data_u)

        logging.debug('extract_data: counting indices for layer %s' % l)
        count1 = count_indices(data1[l], top_n)
        count2 = count_indices(data2[l], top_n)
        logging.debug('extract_data: average activations for layer %s' % l)
        avgs1, avgs2 = top_n_average(data1[l], data2[l], top_n)

        combined_counts = combine_counts(count1, count2)
        _, y_data1, y_data2 = combined_counts[top_n - 1]
        best_n_indices = sorted(range(len(y_data1)), key=lambda i: y_data1[i], reverse=True)[:20] + sorted(range(len(y_data2)), key=lambda i: y_data2[i], reverse=True)[:20]
        best_n_indices = sorted(list(set(best_n_indices)))
        interesting_units[l][:] = best_n_indices

        result[l]['percentages_o'] = percentages_ordered
        result[l]['percentages_u'] = percentages_unordered
        result[l]['equal_ind_o'] = equal_data_ordered
        result[l]['equal_ind_u'] = equal_data_unordered
        result[l]['combined_counts'] = combine_counts(count1, count2)
        result[l]['averages'] = (avgs1, avgs2)

    logging.debug('extract_data: end.')
    return result, interesting_units

def retreive_percentage_data(data, layer, key, top_n):
    """
    data: [{layer : {key : [...]}}, ...]
    key is either percentages_o or percentages_u
    returns list of the percentage of equal indices for every entry in data
    """
    result = []
    for i in xrange(len(data)):
        result.append(data[i][layer][key][top_n-1])
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
    count = []
    for n in xrange(top_n):
        count.append(dict())
        for img_idx in data:
            for unit, val in data[img_idx][:n + 1]:
                if unit in count[n]:
                    count[n][unit] += 1
                else:
                    count[n][unit] = 1

    return count

def combine_counts(count1, count2):
    """
    inserts {unit_idx : 0} in count1 and count2 if unit_idx is missing
    returns list of unit_idx and corresponding count values
    """
    assert len(count1) == len(count2)
    for n in xrange(len(count1)):
        for key in count1[n]:
            if key not in count2[n]:
                count2[n][key] = 0
    
    for n in xrange(len(count2)):
        for key in count2[n]:
            if key not in count1[n]:
                count1[n][key] = 0

    result = []
    for n in xrange(len(count1)):
        x_data1, y_data1 = zip(*count1[n].items())
        x_data2, y_data2 = zip(*count2[n].items())
        
        comb1 = [(x,_) for _,x in sorted(zip(x_data1,y_data1))]
        comb2 = [(x,_) for _,x in sorted(zip(x_data2,y_data2))]
        
        y1, x1 = zip(*comb1)
        y2, x2 = zip(*comb2)    
    
        assert x1 == x2
        result.append((list(x1), list(y1), list(y2)))
    return result

def top_n_average(data1, data2, top_n):
    """
    data1, data2: {img_idx : [(unit_idx, activation_value), ...],...}
    returns the average activation for top1 to top-n
        avgs1[*] -> avgs for top-1 to top-N for data1
        avgs2[*] -> avgs for top-1 to top-N for data2
    """
    logging.debug('top_n_average: start...')
    logging.debug('top_n_average: top_n=%d' % top_n)

    avgs1 = []
    avgs2 = []
    for n in xrange(1, top_n + 1):
        sum_total1 = 0.0
        sum_total2 = 0.0
        avg1 = 0.0
        avg2 = 0.0
        for img_idx in data1:
            sum_img1 = 0
            sum_img2 = 0
            for i in xrange(len(data1[img_idx][:n])):
                sum_img1 += data1[img_idx][i][1]
                sum_img2 += data2[img_idx][i][1]
            sum_total1 += sum_img1/float(n)
            sum_total2 += sum_img2/float(n)
        avg1 = sum_total1 / float(len(data1))
        avg2 = sum_total2 / float(len(data2))
        avgs1.append(avg1)
        avgs2.append(avg2)

    logging.debug('top_n_average: end.')
    return avgs1, avgs2

#TODO complete function
def top_n_activations(data1, data2, top_n):
    """
    data1, data2: {img_idx : [(unit_idx, activation_value), ...],...}
    Returns a list of the top-n average activations over all images
    """
    assert len(data1) == len(data2)

    # {unit_idx : [sum, count]}

    count1 = []
    count2 = []
    avgs1 = []
    avgs2 = []

    for n in xrange(top_n):
        count1.append(dict())
        count2.append(dict())
        for img_idx in data1:
            for i in xrange(len(data1[img_idx][:n + 1])):
                key = data1[img_idx][i][0]
                act = data1[img_idx][i][1]
                if key in avgs1[n]:
                    count1[n][key][0] += act
                    count1[n][key][1] += 1
                else:
                    count1[n][key] = [act, 1]
            for j in xrange(len(data2[img_idx][:n + 1])):
                key = data2[img_idx][i][0]
                act = data2[img_idx][i][1]
                if key in count2[n]:
                    count2[n][key][0] += act
                    count2[n][key][1] += 1
                else:
                    count2[n][key] = [act, 1]

    for i in xrange(len(count1)):
        avgs1.append(list())
        avgs2.append(list())
        # [(unit_idx, avg), ]
        for key in count1[i]:
            pass #TODO

#------------------------------------------------------------------------------------------------------------------------------------
def sorted_avgs(data, top_n):
    x_axis = np.arange(1, top_n + 1, 1)
    y2_vals = []
    for i in x_axis:
        count = 0
        sum2 = 0
        for img_idx in data[i - 1]:
            for info in data[i - 1][img_idx]['equals'][:i]:
                sum2 += info[2]
                count += 1
        if count > 0:
            y2_vals.append(sum2 / float(count))
        else:
            y2_vals.append(0)
    
    return y2_vals

def compile_csv(data, top_n, outdir, iter_step):
    assert type(data) == type(list()), "Extracted data is not a list"
    path = os.path.join(outdir, 'csv' + str(top_n))
    mkdir_p(path)
    # keys:
    #   percentages_o. percentages_u
    #   equal_ind_o, equal_ind_u
    #   combined_counts
    #   averages
    

    layers = []
    for i in xrange(len(data)):
        for l in data[i]:
            layers.append(l)
        break

    percs_u = dict()
    percs_o = dict()
    #count = dict()
    avgs = dict()
    avgs_u = dict()
    avgs_o = dict()
    for l in layers:
        percs_o[l] = []
        percs_u[l] = []
        avgs[l] = []
        avgs_u[l] = []
        avgs_o[l] = []
        for i in xrange(len(data)): #i is iteration 
            percs_o[l].append(data[i][l]['percentages_o'][:top_n]) # [t1, t2, t3][t1_2, t2_2, t3_3], ...
            percs_u[l].append(data[i][l]['percentages_u'][:top_n])
            avgs[l].append([x[1] for x in data[i][l]['averages'][:top_n]])          
            avgs_o[l].append(sorted_avgs(data[i][l]['equal_ind_o'], top_n))
            avgs_u[l].append(sorted_avgs(data[i][l]['equal_ind_u'], top_n))
        
        save_csv(percs_o[l], iter_step, os.path.join(path, l + '_equals_ordered_top' + str(top_n) + '.csv'))
        save_csv(percs_u[l], iter_step, os.path.join(path, l + '_equals_unordered_top' + str(top_n) + '.csv'))
        save_csv(avgs[l], iter_step, os.path.join(path, l + '_avgs_top' + str(top_n) + '.csv'))
        save_csv(avgs_o[l], iter_step, os.path.join(path, l + '_avgs_ordered_top' + str(top_n) + '.csv'))
        save_csv(avgs_u[l], iter_step, os.path.join(path, l + '_avgs_unordered_top' + str(top_n) + '.csv'))
    



def save_csv(data, iter_step, filename):
    with open(filename, 'wb') as csvf:
        writer=csv.writer(csvf, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for i in xrange(len(data)):
            writer.writerow([(i+1)*iter_step] + list(data[i]))
    
    
#-----------------------------------------------------------------------------------------------------------------
def check_if_contains_duplicates(arr):
    return len(arr) != len(set(arr))

def load_pickle(filename):
    #{layer : {img_idx : [(unit_idx, activation_val), ...]}}
    logging.info('reading %s' % filename)
    try:
        f = open(filename)
        data = pickle.load(f)
        f.close()
    except IOError:
        logging.critical('failed to open %s' % filename)
        raise
    except:
        logging.critical('%s does not contain pickled data' % filename)    
    
    logging.info('layers: ' + repr([x for x in data.iterkeys()]))
    return data

def save_pickle(data, filename):
    logging.debug('save_pickle: start...')
    logging.debug('save_pickle: saving extracted data to %s' % filename)
    dirname = os.path.dirname(filename)
    mkdir_p(dirname)

    with open(filename, 'wb') as ff:
        pickle.dump(data, ff, -1)
    #pickle_to_text(filename)
    logging.debug('save_pickle: end.')

def save_json(data, filename):
    logging.debug('save_json: start...')
    logging.debug('save_json: saving json data to %s' % filename)
    dirname = os.path.dirname(filename)
    mkdir_p(dirname)
    with open(filename, 'wt') as f:
        json.dump(data, f)
    logging.debug('save_json: end.')

if __name__ == '__main__':
    main()
