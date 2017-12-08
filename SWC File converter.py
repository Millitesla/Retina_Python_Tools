#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 23:41:53 2017

@author: ruff

SWC File converter
Redefines first point as Soma and rest of points as Dendrites
"""
import os
import glob

#convert swc files so it can be read by neurom
os.chdir('/Users/ruff/OneDrive/Retina Python Tools/DataV2')
files = glob.glob(os.curdir + '/InData/Morphology' + '/*.swc')
for item in files:
    with open(item, 'r') as f, open(item[0:-4] + 'edit.swc', 'w') as ff:
        lines = f.readlines()    
        for i, line in enumerate(lines):
            row = line.split(' ')
            if i > 5:
                if i == 6:
                    row[1] = '1'
                    row[-2] = '1.0'
                else:
                    row[-2] = '1.0'
                    row[1] = '3'      
            ff.write(' '.join(row))
            
#For removing 2 Points of 3Point Soma in Suembuel Cells            
os.chdir('/Users/ruff/OneDrive/Retina Python Tools/DataV2')
files = glob.glob(os.curdir + '/Suembuel Cells' + '/*.swc')
for item in files:
    with open(item, 'r') as f, open(item[0:-4] + 'edit.swc', 'w') as ff:
        lines = f.readlines()    
        for i, line in enumerate(lines):
            row = line.split(' ')
            if i > 0:
                if i == 1:
                    row[1] = '3'
                if i == 2:
                    row[1] = '3'
                else:
                    row[1] = '3'      
            ff.write(' '.join(row))
