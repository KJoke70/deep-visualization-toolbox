#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:30:44 2018

@author: martin

Script to merge json files with the content:
    {k1 : [...], k2 : [...], ...}
"""

import json
import os.path
import argparse

def main():
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('jsons', nargs='+', help = 'json files to be merged')
    parser.add_argument('--outfile', type = str, default = 'merged.json', help = 'Filename for output file')

    args = parser.parse_args()

    if len(args.jsons) == 1:
        print 'Only one json file entered. No output.'
        exit()
    
    jsons = loadJSONs(args.jsons)
    merged = mergeDicts(jsons)
    saveJSON(merged, args.outfile)

def loadJSONs(files):
    assert len(files) > 1, "need files to load."
    for f in files:
        assert os.path.exists(f), "file %s does not exist." % (f)
    jsons = []
    for f in files:
        with open(f, 'rt') as j:
            jsons.append(json.load(j))
    
    return jsons

def saveJSON(data, filename):
    with open(filename, 'wt') as f:
        json.dump(data, f)

def mergeDicts(*arg):
    """
    function to merge dicts in  the format {k1 : [...], k2 : [...], ...}
    returns merged dict of key:list pairs with no duplicates
    """
    assert len(arg) > 0, "can't merge without dicts."
    if len(arg) > 1:
        result = dict()
        for j in arg:
            for k in j.iterkeys():
                if k in result:
                    result[k] = list(set(result[k] + j[k]))
                else:
                    result[k] = list(set(j[k]))
        return result
    else:
        return arg[0]

if __name__ == '__main__':
    main()
