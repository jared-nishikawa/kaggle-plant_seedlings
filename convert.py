#!/usr/bin/python

import png
import sys

def to_row(data, label):
    row = []
    for dat in data:
        if dat == '\x00':
            row.append('0')
        else:
            row.append('1')
    row = ','.join(row)
    row += ',' + label
    return row

def make_train_csv(filename):
    spl = filename.split('/')
    label = spl[-2]
    data,w,h = png.process(filename)

    # Do a pixel check
    threshold = sum(map(ord, data))
    if threshold < 20:
        exit("Not enough white pixels")

    row = to_row(data, label)
    with open('my_csv/train.csv','a') as f:
        f.write(row + '\n')

def make_test_csv(filename):
    spl = filename.split('/')
    base = spl[-1]
    data,w,h = png.process(filename)

    row = to_row(data, base)
    with open('my_csv/test.csv','a') as f:
        f.write(row + '\n')

if __name__ == '__main__':
    if not sys.argv[1:]:
        sys.ext("Need filename")
    #make_train_csv(sys.argv[1])
    make_test_csv(sys.argv[1])

    
