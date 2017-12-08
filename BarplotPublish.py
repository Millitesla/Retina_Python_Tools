#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:15:23 2017

Making publishable barplots with standard error, p-value stars and plotting single data points
Index: In index you specify which properties it should summarize to one common index, default it averages data
@author: ruff
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
def BarplotPublish (data, index, values, filename):
    data = data 
    datapivot = pd.pivot_table(data, index=index) #Averages data so you have only single experiments left (averages e.g over slices of one brain)
    
    fig=plt.figure(figsize=[3,7])
    #fig.add_subplot(111, aspect='equal')
    #fig=plt.style.use('seaborn-pastel')
    fig=sns.set_style('ticks')
    #ax=sns.axes_style({'axes.grid':True, 'axes.facecolor':'white', 'axes.linewidth':10})
    
    fig=sns.barplot(data=datapivot, x='Age', y=values, capsize=.2, errwidth=1, errcolor='black')
    fig=sns.stripplot(data=datapivot, x='Age', y=values, jitter=0.1, color='black')
    
    fig=sns.despine(top=True, trim=False)
        
    fig=plt.tight_layout() #prohibits labels outside figure boarders
    #p.savefig(os.curdir + '/Figures/' + filename + '.svg', frameon= True, transparent= False, dpi=300, bbox_inches='tight')

    
files = glob.glob(os.curdir + '/*ColocalizationFlrt3dsRed_Alldata*')
for item in files:
    data = pd.read_excel(item, delim_whitespace=True, sheetname= 'Sheet1')
    #Call function
    BarplotPublish(data, ['Experiment', 'Brain'], 'FLRT3+/dsRed', 'ColocbarpotFLRT3')
    
files = glob.glob(os.curdir + '/*ColocalizationFlrt1DAPI_Alldata*')
for item in files:
    data = pd.read_excel(item, delim_whitespace=True, sheetname= 'Sheet1')
    #Call function
    BarplotPublish(data, ['Experiment','Brain'], 'FLRT1+/DAPI', 'ColocbarpotFLRT1')
    
#SUM Type1 divided by SUM Type2

