# -*- coding: utf-8 -*-
"""to be oriented horizontal and GCL should face down

@author: ruff
"""

import pandas as pd
import glob
import math
import matplotlib.pyplot as plt
import os
import numpy as np
import pylab as p
from scipy import misc

os.chdir('/Users/ruff/OneDrive/Retina Python Tools/Data')
#files = glob.glob(os.curdir + '/*exp84_16_Stratify*')
files = glob.glob(os.curdir + '/*exp84_16p3*' + '*Cell2_4*')

#Created on Sat Oct  1 14:52:00 2016
#Plots Stratification profile of 8bit tiff images
#Images have 
#Final output dataframe
Output = pd.DataFrame()

#for item in files:
#    tempdata = misc.imread(item)

def Stratify2D (filelist):
    output = pd.DataFrame()
    for item in filelist:
        data = misc.imread(item, flatten = True)
        print(data)
        #normalize to maximum value for each column if you want to see how homogeneous stratification is along x axis
        #data = data/data.max(axis=0)
        #Calculate mean for each row
        data = np.mean(data, axis=1)
        data = data/data.max(axis=0)
        data = pd.DataFrame(data) # convert to dataframe for concatenating
        plt.figure()
        plt.imshow(data, cmap='viridis', interpolation='nearest', aspect='auto')
        p.savefig(os.curdir + '/Figures/' + item + '.pdf', frameon= False, transparent=True)
        output = pd.concat([output, data], axis=1, ignore_index=True)    
    plt.figure()
    plt.imshow(output, cmap='viridis', interpolation='nearest', aspect= 'auto')
    p.savefig(os.curdir + '/Figures/' + 'all.pdf', frameon= False, transparent= True)
    
Stratify2D(files)
