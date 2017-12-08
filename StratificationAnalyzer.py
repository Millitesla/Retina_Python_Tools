#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 18:28:07 2017
Stratification Analyzer
@author: ruff
"""


import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
import glob
import math
import matplotlib.pyplot as plt
import os
from scipy import misc
import pylab as p

os.chdir('/Users/ruff/OneDrive/Retina Python Tools/DataV2')

def StratifyAnalysis (data, filename):
    '''
    Takes Series of images with stratification pattern (binary) and Sums each row, bins into 10, Sums with all other images 
    of same series (=same cell) and plots an image using heatmap
    '''
    binnum = 5
    allallstratify = pd.DataFrame()
    for cell in range(len(data)):
        
        images = glob.glob(os.curdir + '/InFigures/' + '*' + data['SQL_ID1'][cell] + ' ' + '*')
        allstratify=pd.DataFrame(columns={'stratify'}, index=np.linspace(0,binnum-1,binnum)).fillna(0)
        for image in images:
            stratify = pd.DataFrame()
            depth = pd.DataFrame()
            stratify = misc.imread(image, flatten = True)
            stratify = pd.DataFrame(np.sum(stratify, axis=1)).iloc[::-1]
            stratify = stratify.reset_index(drop=True)
            stratify = stratify.rename(columns = {0:'stratify'})
            depth = pd.DataFrame(np.linspace(0, 1, len(stratify)))
            depth = depth.rename(columns={0:'depth'})
            stratify['depth']=depth.depth # add depth column to stratify
            stratify.depth=stratify.depth*binnum
            stratify.depth = stratify.depth.astype(int) #convert to integer for subsequent sumation
            stratify['depth'].replace(to_replace=binnum, value=binnum-1, inplace=True, method='pad', axis=None)
            stratify = pd.pivot_table(data=stratify, columns=stratify.depth , aggfunc='sum').transpose()
            stratifymean = stratify.mean(axis=0) #calculate average of all rows
            allstratify['stratify'] = stratify.stratify + allstratify.stratify
            #print(stratify)
        allstratify = allstratify.stratify/allstratify.stratify.max()#normalize #here you can decide if you want maximum value to be 100% or all values relative
        allstratify = allstratify.rename(columns={'stratify':image[12:30]}) #rename column to cell
        allallstratify[image[12:30]] = allstratify  
    allallstratifymean = pd.DataFrame(allallstratify.mean(axis=1)) #calculate average of all rows
    
    #Select only cells with maximum in Layer 5 or 4,3,2,1 to sort cells
#    allallstratify5 = allallstratify[allallstratify.Layer5==1]
#    allallstratify4 = allallstratify[allallstratify.Layer4==1]
#    allallstratify3 = allallstratify[allallstratify.Layer3==1]
#    allallstratify2 = allallstratify[allallstratify.Layer2==1]
#    allallstratify1 = allallstratify[allallstratify.Layer1==1]    
#    allallstratifysorted = allallstratify1.append([allallstratify2, allallstratify3, allallstratify4, allallstratify5])
    
    #return(allallstratify)
    fig=plt.figure()
    fig=plt.imshow(allallstratify, cmap='viridis', interpolation='nearest', aspect=0.4)
    plt.grid(False)
    plt.title(image[12:30])
    p.savefig(os.curdir + '/Figures/'  +filename+image[12:30] + '.png', frameon= False, transparent=False)
    
#    fig=plt.figure()
#    fig=plt.imshow(allallstratifymean, cmap='viridis', interpolation='nearest', aspect=0.4)
#    plt.grid(False)
#    p.savefig(os.curdir + '/Figures/' + filename+ image[12:30] + 'average'+'.png', frameon= False, transparent=False)
#        #return allstratify
        
# Plot all RGCs
data = pd.read_excel(os.curdir + '/Alldata_Project_Retina.xlsx', sheetname='CellMorphology')
data = data[data['Year'].isin([2017])]
data = data[data['Experiment'].isin([14,22,26])]
data = data[data['Sub_Type'].isin(['RGC'])]
data = data.reset_index(drop=True)

output = StratifyAnalysis(data, 'RGCs ')

# Plot all Amacrines
data = pd.read_excel(os.curdir + '/Alldata_Project_Retina.xlsx', sheetname='CellMorphology')
data = data[data['Year'].isin([2017])]
data = data[data['Experiment'].isin([14,22,26])]
data = data[data['Sub_Type'].isin(['Amacrine'])]
data=data.reset_index(drop=True)

output = StratifyAnalysis(data, 'Amacrines ')


#K-means Clustering analysis
# Sorting dataframe










