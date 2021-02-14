#!/usr/bin/env python
# coding=utf-8

import os
import sys


def func(target, source):
    
    table = {}
    # print(target, source)
    with open(source,'r') as fid:
        for line in fid:
            path = line.strip()
            name = path.split('/')[-1] 
            table[name] = path
    with open(target,'r') as fid:
        for line in fid:
            name = line.strip()
            if name in table and os.path.isfile(path):
                path = table[name]
                print(path)

def main():
    if len(sys.argv) != 3:
        exit(-1)
    func(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
    main()
