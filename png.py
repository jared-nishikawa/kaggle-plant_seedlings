#!/usr/bin/python

import sys
import zlib
import struct
import binascii
import numpy as np

def crc(data):
    c = binascii.crc32(data)
    c &= 0xffffffff
    return struct.pack('>I', c)

def bytes2num(BYTES):
    total = 0
    for byte in BYTES:
        o = ord(byte)
        total <<= 8
        total += o
    return total

def parse(raw):
    header = raw[:8]
    tail = raw[8:]
    BYTES = ''
    while tail:
        tail,NEW,TYPE = get_chunk(tail)
        if TYPE == "IDAT":
            BYTES += NEW
    #print len(BYTES)
    return BYTES

def get_IHDR(chunk):
    assert len(chunk) == 13
    W = bytes2num(chunk[:4])
    H = bytes2num(chunk[4:8])
    depth = bytes2num(chunk[8])
    ctype = bytes2num(chunk[9])
    cmethod = bytes2num(chunk[10])
    fmethod = bytes2num(chunk[11])
    imethod = bytes2num(chunk[12])
    print "Width:", W
    print "Height:", H
    print "Bit depth:", depth
    print "Color type:", ctype
    print "Compression method:", cmethod
    print "Filter method:", fmethod
    print "Interlace method:", imethod
    return W,H,depth,ctype,cmethod,fmethod,imethod

def get_chunk(tail):
    LENGTH = bytes2num(tail[:4])
    TYPE = tail[4:8]
    DATA = tail[8:8+LENGTH]
    CRC = tail[8+LENGTH:12+LENGTH]
    CRC_check = crc(TYPE+DATA)

    assert CRC == CRC_check

    #print "Type:", TYPE
    #print "Length:", LENGTH
    if TYPE == "IHDR":
        #get_IHDR(DATA)
        NEWDAT = ''
    else:
        NEWDAT = DATA
    return tail[12+LENGTH:], NEWDAT, TYPE

def read_image(name):
    with open(name) as f:
        return f.read()

def bits2bytes(bits):
    if len(bits) != 8:
        return ''
    return chr(int(bits,2))

def get_info(raw):
    header = raw[:8]
    tail = raw[8:]
    while tail:
        LENGTH = bytes2num(tail[:4])
        TYPE = tail[4:8]
        if TYPE == "IHDR":
            DATA = tail[8:8+LENGTH]
            return get_IHDR(DATA)
            break
        tail,NEW,TYPE = get_chunk(tail)

def unfilter(filtered_lines, channels):
    unfiltered_lines = []
    for ind,line in enumerate(filtered_lines):
        f_type = ord(line[0])
        #print f_type
        line = line[1:]
        if f_type == 0:
            new_line = map(ord, line)
        elif f_type == 1:
            new_line = []
            for byte in line:
                A = new_line[-channels] if len(new_line) >= channels else 0
                new_byte = (ord(byte) + A)%256
                new_line.append(new_byte)
        elif f_type == 2:
            new_line = []
            for k,byte in enumerate(line):
                A = new_line[-channels] if len(new_line) >= channels else 0
                B = ord(unfiltered_lines[ind-1][k]) if ind>0 else 0
                new_byte = (ord(byte) + B)%256
                new_line.append(new_byte)
        elif f_type == 3:
            exit("Unimplemented filter type!")
        elif f_type == 4:
            new_line = []
            for k,byte in enumerate(line):
                A = new_line[-channels] if len(new_line) >= channels else 0
                B = ord(unfiltered_lines[ind-1][k]) if ind>0 else 0
                C = ord(unfiltered_lines[ind-1][k-channels]) if ind>0 and k>=channels else 0
                p = A + B - C
                distances = [abs(A-p), abs(B-p), abs(C-p)]
                m = min(distances)
                if distances[0] == m:
                    X = A
                elif distances[1] == m:
                    X = B
                else:
                    X = C
                new_byte = (ord(byte) + X)%256
                new_line.append(new_byte)
        else:
            exit("Invalid filter type")

        unfiltered_lines.append(''.join(map(chr,new_line)))

    return unfiltered_lines

def get_channels(ctype):
    if ctype == 6:
        return 4
    elif ctype == 2:
        return 3
    elif ctype == 0:
        return 1
    else:
        exit("Unhandled color type")


def process(filename):
    raw = read_image(filename)
    W,H,depth,ctype,cmethod,fmethod,imethod = get_info(raw)
    BYTES = parse(raw)
    # BYTES is the result of pixels being filtered, then compressed
    
    # First decompress
    decomp = zlib.decompress(BYTES)

    # Now, defilter.
    # Filter type: 0
    # With the None filter, the scanline is transmitted unmodified
    # it is only necessary to insert a filter type byte before the data.
    channels = get_channels(ctype)
    sz = channels*W + 1


    flines = [decomp[sz*i: sz*(i + 1)] for i in range(H)]
    #print list(flines[0][1:])

    uflines = unfilter(flines, channels)

    # Now, DATA is unfiltered
    DATA = ''.join(uflines)
    return DATA, W, H

def make_png(rgb_data, w, h, ctype):
    # png magic number
    header = '\x89PNG\r\n\x1a\n'

    # 13 byte IHDR
    # Width - 4 bytes
    # Height - 4 bytes
    # depth = 1 byte
    # color type = 1 byte
    # compression method = 1 byte
    # filter method = 1 byte
    # interlace method = 1 byte
    W = struct.pack('>I', w)
    H = struct.pack('>I', h)

    # Color type 0 = Grayscale
    # Color type 2 = Truecolor
    # Color type 6 = Truecolor + alpha
    depth = chr(8)
    channels = get_channels(ctype)
    ctype = chr(ctype)
    cmethod = chr(0)
    fmethod = chr(0)
    imethod = chr(0)
    ihdr = W + H + depth + ctype + cmethod + fmethod + imethod
    c = crc("IHDR" + ihdr)
    ihdr = struct.pack('>I', 13) + "IHDR" + ihdr + c

    tail = ''

    lines = []
    # filter
    while rgb_data:
        line = rgb_data[:channels*w]
        rgb_data = rgb_data[channels*w:]
        # ftype = 0
        lines.append(chr(0) + line)
    filtered_data = ''.join(lines)
    compressed = zlib.compress(filtered_data)

    png_data = ''
    # make chunks
    while compressed:
        #print len(compressed)
        if len(compressed) >= 65536:
            chunk_data = compressed[:65536]
            compressed = compressed[65536:]
        else:
            chunk_data = compressed
            compressed = ''
        length = struct.pack('>I', len(chunk_data))
        chunk = length + "IDAT" + chunk_data
        crc_check = crc("IDAT" + chunk_data)
        png_data += chunk + crc_check
    return header + ihdr + png_data

def luma(r,g,b):
    m = max(r,b)
    g -= m
    threshold = 5
    if g > threshold:
        return 255
    else:
        return 0
    #return min(int(0.299*r + 0.587*g + 0.114*b),255)

def grayscale(rgb_data, channels=3):
    R = rgb_data[::channels]
    G = rgb_data[1::channels]
    B = rgb_data[2::channels]
    
    Z = [chr(luma(ord(R[i]), ord(G[i]), ord(B[i]))) for i in range(len(R))]
    return ''.join(Z)

def resize(rgb_data, w,h, target_w, target_h):
    lines = [rgb_data[w*i:w*(i+1)] for i in range(h)]
    new_lines = []
    for j in range(target_h):
        new_line = ''
        approx_y = int(j*float(h)/target_h)
        for i in range(target_w):
            approx_x = int(i*float(w)/target_w)
            new_line += lines[approx_y][approx_x]
        new_lines.append(new_line)
    return ''.join(new_lines)

if __name__ == '__main__':
    if not sys.argv[1:]:
        sys.exit("Need filename")
    filename = sys.argv[1]
    rgb_data,w,h = process(filename)
    gray = grayscale(rgb_data)

    gray = resize(gray, w,h, 32, 32)
    w = h = 32

    test_data = make_png(gray, w, h, 0)
    with open('resized/' + filename,'w') as f:
        f.write(test_data)



