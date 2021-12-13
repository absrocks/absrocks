#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os


# parameters

def my_print( lns, spacing = 3 ):
    widths = [max(len(value) for value in column) + spacing
              for column in zip(*lns)]
    proc_seqf = open('processed_seq.txt','a')
    for line in lns:
        pretty = ''.join('%-*s' % item for item in zip(widths, line))
        print(pretty) # debugging print
        proc_seqf.write(pretty + '\n')
    return
