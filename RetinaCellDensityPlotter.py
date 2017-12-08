# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 23:55:56 2016

@author: Tobias Ruff

Function: Calculation of local densities of dots (cells) in 2D array. Local density defined by sampleradius.
Adds new density column to input dataframe.
Plots density color coded dots.
"""

import pandas as pd
import glob
import math
import matplotlib.pyplot as plt
import os
import numpy as np
import pylab as p
import seaborn as sns

os.chdir('/Users/ruff/OneDrive/Retina Python Tools/Data')
#Conversion = pd.read_csv(os.curdir + '/Conversion_exp55_16 pixel um mac.csv', sep=';', index_col='Index') # Use only in case you read in pixel coordinates

def dotdensity2D (tempdata, cx, cy,celltype,types, sampleradius, colormap, dotsize, transparancy, minimum, maximum, filename, numsize, color, limit, invert):
    '''
    Parameters
    ----------
    minimum: lowest value for value 0 of colormap cmap
    maximum: highest value for value 1 of colormap cmap
    types: boolean. Decide if you want to take all (False) or onlye one type ()
    limit: boolean. Decide if you want to limit plot area
    Returns
    ------
    Result: Density of cells within radius sampleradius in cells/mm^2
    '''
    mx=pd.DataFrame()
    my=pd.DataFrame()
    radius=pd.DataFrame()
    densities=pd.DataFrame()
    #Slicing out the defined cell type in Type column
    if 'Type' in tempdata\
    and types == True:
        tempdata = tempdata[tempdata.Type == celltype]
    else:
        pass
    
    mx =pd.DataFrame.transpose(pd.DataFrame(np.array(tempdata.loc[:,cx]) + np.zeros([len(tempdata),len(tempdata)]))) 
    #Create dataframe with by boradcasting one column over row number of columns to get nxn matrix
    my =pd.DataFrame.transpose(pd.DataFrame(np.array(tempdata.loc[:,cy]) + np.zeros([len(tempdata),len(tempdata)])))
    radius=(((mx - pd.DataFrame.transpose(mx)).apply(np.square)) + ((my - pd.DataFrame.transpose(my)).apply(np.square))).apply(np.sqrt) #Differenz zwischen matrix und transponierter matrix
    densities = ((radius<sampleradius).sum(axis=1))/(math.pi*((sampleradius/1000)*(sampleradius/1000))) # check which radius are below threshold and count them per row. divide by radius defined area  
    #tempdata.loc[:,'density'] = densities 
    
    fig=plt.figure()
    plt.style.use('seaborn-white')
    ax = fig.add_subplot(111, aspect='equal') # make aspect ratio equal
    ax.grid(False) 
    if invert == True:
        plt.scatter(tempdata.loc[:,cx], -tempdata.loc[:,cy], c=densities, cmap=colormap,  edgecolors='none',alpha=transparancy, s=dotsize)#, vmin=minimum, vmax=maximum)
    else: 
        plt.scatter(tempdata.loc[:,cx], tempdata.loc[:,cy], c=densities, cmap=colormap,  edgecolors='none',alpha=transparancy, s=dotsize)#, vmin=minimum, vmax=maximum)

    plt.colorbar()
    plt.xlabel('µm', fontsize=numsize)
    plt.ylabel('µm', fontsize=numsize)
    plt.xticks(fontsize=numsize, rotation=45)
    plt.yticks(fontsize=numsize)
#    if limit == True:    
#        plt.xlim(0,1300)
#        plt.ylim(0,1300)
    plt.tight_layout() #prohibits labels outside figure boarders
    p.savefig(os.curdir + '/Figures/' + filename +'dotdensity2D.png', frameon= False, transparent= True) 
#    plt.figure()    
#    sns.distplot(densities, hist=True, kde = False, bins = 10)  
#    plt.xlabel('Cells/$mm^2$')
#    plt.ylabel('# of Cells')
 
#files = glob.glob(os.curdir+ '/*exp84_16**wholemount*.csv')
#for item in files:
#    data=pd.read_csv(item, sep=';', skip_blank_lines=True)
#    dotdensity2D(data, 'X(micron)', 'Y(micron)', 1, False, 100, 'viridis', 5, 0.9, 0,1, item + 'FLRT3+ RGCs', 23, 'white', False, True)
#    
    #data.XM = data.XM*float(Conversion.loc[item])/1000 #Use only for conversion
    #data.YM = data.YM*float(Conversion.loc[item])/1000 #Use only for conversion
    dotdensity2D(data, 'X(micron)', 'Y(micron)', 1, True, 80, 'viridis', 10, 0.8, 0,300, item + 'FLRT3+ RGC', 23, 'white', True, False)
    dotdensity2D(data, 'X(micron)', 'Y(micron)', 2, True, 80, 'viridis', 10, 0.8, 0,300, item + 'FLRT3+ only', 23, 'white', True, False)
    dotdensity2D(data, 'X(micron)', 'Y(micron)', 4, True, 80, 'viridis', 10, 0.8, 0,300, item + 'RBPMS', 23, 'white', True, False)
    
#files = glob.glob(os.curdir + '/*exp9_14_Results*.csv')    
#for item in files:
#    data = pd.read_csv(item, sep=';', skip_blank_lines=True)
#    dotdensity2D(data, 'X', 'Y', 1, True, 60, 'viridis', 20, 0.8, 0,300, item + 'ChAT', 23, 'white', True, False)
    
#files = glob.glob(os.curdir + '/*exp10_14 Brn3a*.csv')    
#for item in files:
#    data = pd.read_csv(item, sep=';', skip_blank_lines=True)
#    dotdensity2D(data, 'X', 'Y', 1, False, 60, 'viridis', 20, 0.8, 0,300, item + 'Brn3a', 23, 'white', True, False)    
    
files = glob.glob(os.curdir + '/*exp84_16_p3_P60_wholemount*.csv')      
for item in files:
    data = pd.read_csv(item, sep=';', skip_blank_lines=True)
    #data.XM = data.XM*float(Conversion.loc[item])/1000 #Use only for conversion
    #data.YM = data.YM*float(Conversion.loc[item])/1000 #Use only for conversion
    dotdensity2D(data, 'X(micron)', 'Y(micron)', 1, True, 80, 'viridis', 10, 0.8, 0,300, item + 'FLRT3+ RGC', 23, 'white', True, False)
    dotdensity2D(data, 'X(micron)', 'Y(micron)', 2, True, 80, 'viridis', 10, 0.8, 0,300, item + 'FLRT3+ only', 23, 'white', True, False)

