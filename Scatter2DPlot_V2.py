#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:35:29 2017

@author: ruff
"""

import pandas as pd
import glob
import math
import matplotlib.pyplot as plt
import os
import numpy as np
import pylab as p
import seaborn as sns

os.chdir('/Users/ruff/OneDrive/Retina Python Tools/DataV2')

def Scatter2D_Plot_V2 (data, filename, limit, color):
    ''' column_value: Give array of values that you want to select from dataframe eg. if you want types 1 and 2 then: [1,2]'''

    fig=plt.figure()
    ax = fig.add_subplot(111, aspect='equal') # make aspect ratio equal  
    ax.grid(True)
    plt.xlim(0,limit)
    plt.ylim(0,limit)
    plt.scatter(x=data['X(micron)'], y=data['Y(micron)'], edgecolors='None', c=color, s=1)
    plt.tight_layout() #prohibits labels outside figure boarders
    p.savefig(os.curdir + '/Figures/' + filename + 'scatter2D.pdf', frameon= True, transparent= True) 

#
data = pd.read_excel(os.curdir + '/Alldata_Project_Retina.xlsx', sheetname='CellCounter')
data = data[data['Year'].isin([2016])]
data = data[data['Experiment'].isin([90])]
data = data[data['Staining'].isin(['RBPMS CART'])]

type1 = data[data['Type'].isin([1])]
Scatter2D_Plot_V2 (type1, 'exp90_16 Calb+ RGC', limit=6000, color = 'coral')

type2 = data[data['Type'].isin([2])]
Scatter2D_Plot_V2 (type2, 'exp90_16 Calb- RGC', limit=6000, color = 'lightsalmon')

type3 = data[data['Type'].isin([3])]
Scatter2D_Plot_V2 (type3, 'exp90_16 Calb+ Amacrine', limit=6000, color = 'royalblue')

type4 = data[data['Type'].isin([4])]
Scatter2D_Plot_V2 (type4, 'exp90_16 Calb- Amacrine', limit=6000, color = 'dodgerblue') 

type12 = data[data['Type'].isin([1,2])]
Scatter2D_Plot_V2 (type1, 'exp90_16 Flrt3+ RGCs 1D5', limit=6000, color = 'dodgerblue')

type34 = data[data['Type'].isin([3,4])]
Scatter2D_Plot_V2 (type1, 'exp90_16 Flrt3+ Amacrines 1D5', limit=6000, color = 'salmon')
   
#Scatter2D_Plot_V2 (data, [2016], [90], 'Type', (1,2), 'exp90_16_', limit=6000)
#Scatter2D_Plot_V2 (data, [2017], [9], 'Type', (1), 'exp9_17_', limit=4500)
#Scatter2D_Plot_V2 (data, [2014], [9], 'Type', (1), 'exp9_14_', limit=1000)

data = pd.read_excel(os.curdir + '/Alldata_Project_Retina.xlsx', sheetname='CellCounter')
data = data[data['Year'].isin([2016])]
data = data[data['Experiment'].isin([90])]
data = data[data['Staining'].isin(['RBPMS CART'])]

type1 = data[data['Type'].isin([1])]
Scatter2D_Plot_V2 (type1, 'exp90_16_FLRT3+ CART+ RGCs', limit=6000, color = 'coral')

type2 = data[data['Type'].isin([2])]
Scatter2D_Plot_V2 (type2, 'exp90_16_FLRT3+ CART neg RGCs', limit=6000, color = 'lightsalmon')

type2 = data[data['Type'].isin([1, 2])]
Scatter2D_Plot_V2 (type2, 'exp90_16_all FLRT3+ RGCs from CART', limit=6000, color = 'salmon')

type4 = data[data['Type'].isin([4])]
Scatter2D_Plot_V2 (type4, 'exp90_16_FLRT3+ Amacrines', limit=6000, color = 'dodgerblue') 

#Plotting Founder line 2G10
data = pd.read_excel(os.curdir + '/Alldata_Project_Retina.xlsx', sheetname='CellCounter')
data = data[data['Year'].isin([2017])]
data = data[data['Experiment'].isin([19])]
data = data[data['Founder_line'].isin(['2G10'])]
data = data[data['Staining'].isin(['RBPMS CART'])]

type12 = data[data['Type'].isin([1,2])]
Scatter2D_Plot_V2 (type12, 'exp19_17 Flrt3+ RGCs 2G10', limit=6000, color = 'dodgerblue')

type34 = data[data['Type'].isin([3,4])]
Scatter2D_Plot_V2 (type34, 'exp19_17 Flrt3+ Amacrines 2G10', limit=6000, color = 'salmon')


#Plotting Founder line 1G1
data = pd.read_excel(os.curdir + '/Alldata_Project_Retina.xlsx', sheetname='CellCounter')
data = data[data['Year'].isin([2017])]
data = data[data['Experiment'].isin([20])]
data = data[data['Founder_line'].isin(['1G1'])]
data = data[data['Staining'].isin(['RBPMS CART'])]

type12 = data[data['Type'].isin([1,2])]
Scatter2D_Plot_V2 (type12, 'exp20_17 Flrt3+ RGCs 1G1', limit=6000, color = 'dodgerblue')

type34 = data[data['Type'].isin([3,4])]
Scatter2D_Plot_V2 (type34, 'exp20_17 Flrt3+ Amacrines 1G1', limit=6000, color = 'salmon')


