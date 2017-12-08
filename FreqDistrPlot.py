# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:50:00 2016

Frequency distribution plotter
Accepts different Types = conditions

isType: True or False depending if you have Type column or not
@author: ruff
"""
import matplotlib.pyplot as plt
import pylab as p
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import glob
import math
import random
from scipy.stats import gaussian_kde
from scipy.stats import norm
import seaborn as sns


def FreqDistr (filename, isType,types,typenamecol, ycol,typecol,binnumber, name, color, histogram, kdens, ylabel):
    os.chdir('/Users/ruff/OneDrive/Retina Python Tools/DataV2')
    filelist = glob.glob(os.curdir + '/*'+filename+'*')
    for item in filelist:
        data = pd.read_csv(item, sep=';', skip_blank_lines=True)
        if isType == True:
            fig=plt.figure(figsize=(15, 10))
            fig=plt.style.use('seaborn-white')
            sns.set_context("talk",font_scale=3.5, rc={"lines.linewidth": 2.5})
            #sns.set_style('ticks', {'text.color': color, 'ytick.color': 'white', 'xtick.color': color, 'xlabel': color, 'ylabel':color}) 
            sns.set_palette('colorblind')                           
            for i in types:
                ax=sns.distplot(data.loc[data.loc[:,typecol]==i,ycol], bins=binnumber, hist=histogram, rug=False, kde=kdens, norm_hist=False,
                             label=data.loc[data.loc[:,'Type']==i,'Typename'].iloc[1], hist_kws=dict(alpha=0.5), kde_kws={"shade": True})
                plt.xlabel("Soma size" + " [$\mu$$m^2$]", color = color)
                plt.ylabel(ylabel, color = color)
                plt.yticks(color=color)
                plt.xticks(color=color)
                plt.legend()
#                vals = ax.get_yticks()
#                ax.set_yticklabels(['{:3.1f}'.format(x*100) for x in vals])
                for axis in ['bottom','left']: #Set axis color and linewidth
                    ax.spines[axis].set_linewidth(1)
                    ax.spines[axis].set_color(color) 
                sns.despine()                
                #sns.set_style('ticks', {'axes.facecolor': 'white', 'text.color': 'white', 'ytick.color': 'white', 'xtick.color': 'white'})                
                
        else:
            plt.figure(figsize=(15,10))
            sns.set_context("talk",font_scale=3.5, rc={"lines.linewidth": 2.5})
            sns.set_palette('colorblind')                           
            plt.xlabel(ycol + "[$\mu$$m^2$]", color = color)
            plt.ylabel("Kernel density", color = color)            
            sns.distplot(data.loc[:,ycol], bins=binnumber, hist=True, rug=False, kde=True, label=data.loc[data.loc[:,'Type']==1,'Typename'].iloc[1]) 
#            vals = ax.get_yticks()
#            ax.set_yticklabels(['{:3.1f}'.format(x*100) for x in vals])
            plt.yticks(color=color)
            plt.xticks(color=color)
            plt.legend()
            sns.despine
        plt.tight_layout()
        p.savefig(os.curdir + '/Figures/' + item + ycol + name + '.png',  frameon=False, transparent=True)
           
        #else: # Plot the whole column

FreqDistr('exp84_16_Cellsize', True, [3,4],'Typename', 'Area', 'Type', 20, 'RGCP30 and P60_hist', 'black', True, False, "# of cells")
FreqDistr('exp84_16_Cellsize', True, [1,2,3,5],'Typename', 'Area', 'Type', 20, 'P30_hist', 'black', True, False, "# of cells")

FreqDistr('exp84_16_Cellsize', True, [3,4],'Typename', 'Area', 'Type', 20, 'RGCP30 and P60_kde', 'black', False, True, "Kernel density")
FreqDistr('exp84_16_Cellsize', True, [1,2,3,5],'Typename', 'Area', 'Type', 20, 'P30_kde', 'black', False, True, "Kernel density")

#FREQUENCY DISTRIBUTION ANALYZER#######
## Use only code from here for quick frequency distribution analysis of area
data = pd.read_excel(os.curdir + '/Alldata_Project_Retina.xlsx', sheetname='CellMorphology')
data = data[data['Year'].isin([2017])]
data = data[data['Experiment'].isin([14,22,26])]
data = data[data['Sub_Type'].isin(['RGC'])]
data=data.reset_index(drop=True)

fig=plt.figure()
fig=plt.style.use('seaborn-white')
ax=sns.distplot(data.Area, bins=5, hist=False, rug=False, kde=True, norm_hist=False, hist_kws=dict(alpha=0.5), kde_kws={"shade": True})

data = pd.read_excel(os.curdir + '/Alldata_Project_Retina.xlsx', sheetname='CellSize')
data = data[data['Year'].isin([2016])]
data = data[data['Experiment'].isin([89])]
data = data[data['Type'].isin([2])]
data=data.reset_index(drop=True)

fig=plt.figure()
fig=plt.style.use('seaborn-white')
ax=sns.distplot(data.Area, bins=5, hist=True, rug=False, kde=True, norm_hist=False, hist_kws=dict(alpha=0.5), kde_kws={"shade": True})
