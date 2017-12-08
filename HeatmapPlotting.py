#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:38:40 2017

@author: ruff
Heatmap plotting
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import pylab as p

#Main directory
os.chdir('/Users/ruff/OneDrive/FoldingDataPython/Data')

#Function for heatmap
def Heatmapping (data, index, columns, values, filename):
    data = data
    data = data[[index,columns,values]]
    data = pd.pivot_table(data, index=index, columns=columns, values = values)
    #fig=plt.figure()
    #fig=sns.factorplot(x='Age',y='FLRT3+/dsRed', data = data, kind = 'bar')
    
     #Draw the heatmap with the mask and correct aspect ratio
    fig=plt.figure()
    fig.add_subplot(111, aspect='equal')
    fig=sns.heatmap(data,linewidths=.7, yticklabels=True, annot=True, vmin=0, vmax=35)
    plt.title(values)
    fig=plt.yticks(rotation=0)
    fig=plt.xticks(rotation=0)
    #plt.ylabel(pd.DataFrame(data.index.get_values()), rotation=0)
    
    fig=plt.tight_layout() #prohibits labels outside figure boarders
    p.savefig(os.curdir + '/Figures/' + filename + '.pdf', frameon= True, transparent= False, dpi=300, bbox_inches='tight')

    
files = glob.glob(os.curdir + '/*ColocalizationFlrt3dsRed_Alldata*')
for item in files:
    data = pd.read_excel(item, delim_whitespace=True, sheetname= 'Sheet1')
    #Call function
    Heatmapping(data, 'MarkerAge','n', 'FLRT3+/dsRed', 'heatmap')

    