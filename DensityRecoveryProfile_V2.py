# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 14:06:37 2016

Version 1
Date: 11.09.2016
Description: 
1.Project all points to zero point
2.Count density of points in x,y matrix that are within radius r

@author: ruff tobias
"""
import appnope
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
import seaborn as sns

os.chdir('/Users/ruff/OneDrive/Retina Python Tools/DataV2')

def DensRecovProf (inputdata, XM, YM, rmax, binsize, filename, numsize, colors, xySelect):
    '''
    !!! ATTENTION: SOMEHOW FUNCTION IS MANIPULATING ORIGINAL DATA INPUT and not giving right output!!!
    Parameters
    ----------
    tempdata: pandas Dataframe with coordinates
    XM: column with X-coordinate of cells
    YM: column with Y-coordinate of cells
    celltype: select which type of cell you want to analyze
    types: boolean. Decide if you want to take all (False) or only one subtype of cell for calculation(True)
    rmax: specifies maximum radius for profile analysis
    binsize: specifies size of each anulus for calculating density
    numsize: size of fonts (title, axis, xylabel)
    xSelectMin/Max: Min/Max x value of square to select from retina
    ySelectMin/Max: Min/Max y value of square to select from retina
    xySelect: Dictionary with xmin, xmax, ymin, ymax values for square to select from retina
    Results
    --------
    Density recovery profile
    Spatial autocorrelogram
    '''
    outDataRand = pd.DataFrame()
    outData = pd.DataFrame() #dataframe where calculations from all square selections are summarized for plotting
    tempdata = pd.DataFrame()
    #Select Squares on retina with x and y values
    for d in range(len(xySelect['xmax'])):
        tempdata = inputdata
        tempdata = tempdata[(tempdata['X(micron)']>=xySelect['xmin'][d]) & (tempdata['X(micron)']<=xySelect['xmax'][d])]
        tempdata = tempdata[(tempdata['Y(micron)']>=xySelect['ymin'][d]) & (tempdata['Y(micron)']<=xySelect['ymax'][d])]
        tempdata.reset_index(inplace=True, drop=True)
        
        #Here you are manipulating input dataframe!!!!   One should avoid that!!!!
        tempdata.loc[:,XM]=tempdata.loc[:,XM]-min(tempdata.loc[:,XM])#Normalize first to bring origin of image crop to start at 0.0
        tempdata.loc[:,YM]=tempdata.loc[:,YM]-min(tempdata.loc[:,YM])#Normalize first to bring origin of image crop to start at 0.0
                   
                    
        #Create Random number X and Y
        randX = pd.DataFrame()
        randX.loc[:,'randX'] = np.random.random_sample(len(tempdata))*(tempdata.loc[:,XM].max())
        randY = pd.DataFrame()    
        randY.loc[:,'randY'] = np.random.random_sample(len(tempdata))*(tempdata.loc[:,YM].max())  
        
        a = max(tempdata.loc[:,YM])-min(tempdata.loc[:,YM])# side length of image calculated by ymax and ymin in array
        maxy = a-rmax #maximum height of box for center points
        b = max(tempdata.loc[:,XM])-min(tempdata.loc[:,XM])# side length of image calculated by xmax and xmin in array
        maxx = b-rmax #maximum length of box for center points
        #Select all center points for projection (if points are within x=rmax;b-rmax and y=rmax;a-rmax) and use them to
        #project point of table and correlating points within radius rmax to position rmax, rmax and transfer to new table
        normdata = pd.DataFrame()
        for i in range(len(tempdata)):
            tempnormdata = pd.DataFrame()
            if  tempdata.loc[i,XM] > rmax and tempdata.loc[i,XM] < maxx and tempdata.loc[i,YM] > rmax and tempdata.loc[i,YM] < maxy: #Select data points within range and take x,y to shift all other points 
                tempnormdata.loc[:,XM] = tempdata.loc[:,XM] - (tempdata.loc[i,XM]) #normalize points by subtracting 
                tempnormdata.loc[:,YM] = tempdata.loc[:,YM] - (tempdata.loc[i,YM]) #normalize points by subtracting 
                tempnormdata.loc[:,'randX'] = randX.loc[:,'randX'] - (randX.loc[i,'randX'])
                tempnormdata.loc[:,'randY'] = randY.loc[:,'randY'] - (randY.loc[i,'randY'])           
                #Add to output dataframe normdata
                normdata = normdata.append(tempnormdata)
                normdata.reset_index(drop=True, inplace=True)
            else:
                pass          
            
        #Calculate density of each Annulus 
        #Calculate radius for each point and use loc function to insert it in place
        normdata.loc[:,'radius']=(normdata.loc[:,XM]**2+normdata.loc[:,YM]**2)**0.5 
        normdata.loc[:,'randRadius']=(normdata.loc[:,'randX']**2+normdata.loc[:,'randY']**2)**0.5 
        #Remove radius with value 0
        normdataRand = pd.DataFrame()
        normdataRand = normdata.loc[normdata.loc[:,'randRadius']>0,:]
        normdata = normdata.loc[normdata.loc[:,'radius']>0,:]
        #determine range of bins defined by rmax
        binRange = np.arange(0,rmax,binsize)
        #bin the radius into defined bins
        hist=np.histogram(normdata.radius, bins=binRange)
        randhist=np.histogram(normdataRand.randRadius, bins=binRange)
        histcounts = hist[0]
        histcountsRand = randhist[0]
            
        #Calculate total number of cells in analyzed area
        bindensity=pd.DataFrame()
        bindensityRand=pd.DataFrame()
        for i in range(len(binRange)-1):    
            bindensity.loc[i,'bindensity'] = histcounts[i]/((math.pi*(((i+1)*(binsize/1000))**2)-(math.pi*((i*(binsize/1000))**2)))*len(tempdata))
            bindensityRand.loc[i,'bindensityRand'] = histcountsRand[i]/((math.pi*(((i+1)*(binsize/1000))**2)-(math.pi*((i*(binsize/1000))**2)))*len(tempdata))
        
        
        bindensity=bindensity.reset_index(drop=True)
        bindensityRand = bindensityRand.reset_index(drop=True)
        meanRand = bindensityRand.bindensityRand.mean(axis=0)
        
        #Normalize
        avgdensity = bindensity.loc[:, 'bindensity'].mean()
        avgdensityRand = bindensityRand.loc[:, 'bindensityRand'].mean()
        #print (avgdensity)
        bindensity.loc[:,'bindensity'] = bindensity.loc[:,'bindensity']/avgdensity
        bindensityRand.loc[:,'bindensityRand'] = bindensityRand.loc[:,'bindensityRand']/avgdensityRand


        #Append all data into one dataframe
        outData.loc[:,d] = bindensity 
        outDataRand.loc[:,d] = bindensityRand
    
    fig=plt.figure()
    plt.style.use('seaborn-white')
    #ax = fig.add_subplot(111) # make aspect ratio equal
    #ax.grid(False) 
    #plt.title('Density recovery profile ' + filename, fontsize=numsize)
    #plt.bar(bindensity.loc[1:int((rmax/binsize)), 'index'], bindensity.loc[1:int((rmax/binsize)),'bindensity'])
    fig=sns.barplot(data=outData.transpose(), errwidth=1, color=colors, estimator=np.mean, capsize=0.2, ci=68)
    plt.ylabel('Normalized Density', fontsize=numsize)
    plt.ylim(0,3.6)
    plt.xticks(fontsize=numsize)
    plt.yticks(fontsize=numsize)
    plt.xlabel('Distance [*10µm]', fontsize=numsize)
    plt.tight_layout()
    p.savefig(os.curdir + '/Figures/' + filename+ 'DensRecovProf.png', frameon= False, transparent= False)

        
#        #make new dataframe with xy coordinates only within radius for Retina Cell Denisty Plotter to check data
#        normdatafiltered = pd.DataFrame()
#        normdatafiltered= normdata.loc[normdata.loc[:,'radius']<rmax,:]
#        normdatafiltered= normdatafiltered.loc[normdatafiltered.loc[:,'radius']>0,:].reset_index(drop=True)
#        fig=plt.figure()
#        ax = fig.add_subplot(111, aspect='equal') # make aspect ratio equal
#        ax.grid(False) 
#        plt.scatter(normdatafiltered.loc[:,XM], normdatafiltered.loc[:,YM], edgecolors='none', s=2, color=colors, alpha=0.7)
#        plt.title('Autocorrelogram ' + filename, fontsize=numsize)
#        plt.xticks(fontsize=numsize, rotation=90)
#        plt.yticks(fontsize=numsize)
#        plt.ylabel('µm', fontsize=numsize)
#        plt.xlabel('µm', fontsize=numsize)  
    return [outData, outDataRand]

#Select Experiment from dataframe
data = pd.read_excel(os.curdir + '/Alldata_Project_Retina.xlsx')
data = data[data['Year'].isin([2016])]
data = data[data['Experiment'].isin([90])]
data = data[data['Sub_Type'].isin(['CART'])]

type1 = DensRecovProf(data[data['Type'].isin([1])], 'X(micron)', 'Y(micron)', 180,10, 'exp90_16_FLRT3+ CART+ RGCs',23, 'coral', {'xmin':[2914,1068,700,3096],'xmax':[4389,2434,2677,4342], 'ymin':[1474,1197,3187,3226],'ymax':[2850,2062,4143,4143]})
type2 = DensRecovProf(data[data['Type'].isin([2])], 'X(micron)', 'Y(micron)', 180,10, 'exp90_16_FLRT3+ CART neg RGCs',23, 'lightsalmon', {'xmin':[2914,1068,700,3096],'xmax':[4389,2434,2677,4342], 'ymin':[1474,1197,3187,3226],'ymax':[2850,2062,4143,4143]})
type12 = DensRecovProf(data[data['Type'].isin([1,2])], 'X(micron)', 'Y(micron)', 180,10, 'exp90_16_all FLRT3+ RGCs from CART',23, 'salmon', {'xmin':[2914,1068,700,3096],'xmax':[4389,2434,2677,4342], 'ymin':[1474,1197,3187,3226],'ymax':[2850,2062,4143,4143]})
type4 = DensRecovProf(data[data['Type'].isin([4])], 'X(micron)', 'Y(micron)', 180,10, 'exp90_16_FLRT3+ Amacrines',23, 'dodgerblue', {'xmin':[2914,1068,700,3096],'xmax':[4389,2434,2677,4342], 'ymin':[1474,1197,3187,3226],'ymax':[2850,2062,4143,4143]})
#
data = pd.read_excel(os.curdir + '/Alldata_Project_Retina.xlsx')
data = data[data['Year'].isin([2016])]
data = data[data['Experiment'].isin([90])]
data = data[data['Sub_Type'].isin(['Calbindin'])]

type1 = DensRecovProf(data[data['Type'].isin([1])], 'X(micron)', 'Y(micron)', 180,10, 'exp90_16_FLRT3+ Calbindin+ RGCs',23, 'coral', {'xmin':[882, 2773, 328, 1678, 1894],'xmax':[2032, 3913, 1747, 3295, 2763], 'ymin':[1375, 1206, 2802, 4282, 2885],'ymax':[2296, 3005, 3671, 5168, 4191]})
type2 = DensRecovProf(data[data['Type'].isin([2])], 'X(micron)', 'Y(micron)', 180,10, 'exp90_16_FLRT3+ Calbindin neg RGCs',23, 'lightsalmon', {'xmin':[882, 2773, 328, 1678, 1894],'xmax':[2032, 3913, 1747, 3295, 2763], 'ymin':[1375, 1206, 2802, 4282, 2885],'ymax':[2296, 3005, 3671, 5168, 4191]})
type12 = DensRecovProf(data[data['Type'].isin([1,2])], 'X(micron)', 'Y(micron)', 180,10, 'exp90_16_all FLRT3+ RGCs from Calbindin',23, 'salmon', {'xmin':[882, 2773, 328, 1678, 1894],'xmax':[2032, 3913, 1747, 3295, 2763], 'ymin':[1375, 1206, 2802, 4282, 2885],'ymax':[2296, 3005, 3671, 5168, 4191]})
type3 = DensRecovProf(data[data['Type'].isin([3])], 'X(micron)', 'Y(micron)', 180,10, 'exp90_16_FLRT3+ Calbindin+ Amacrine',23, 'royalblue', {'xmin':[882, 2773, 328, 1678, 1894],'xmax':[2032, 3913, 1747, 3295, 2763], 'ymin':[1375, 1206, 2802, 4282, 2885],'ymax':[2296, 3005, 3671, 5168, 4191]})
type4 = DensRecovProf(data[data['Type'].isin([4])], 'X(micron)', 'Y(micron)', 180,10, 'exp90_16_FLRT3+ Calbindin neg Amacrine',23, 'dodgerblue', {'xmin':[882, 2773, 328, 1678, 1894],'xmax':[2032, 3913, 1747, 3295, 2763], 'ymin':[1375, 1206, 2802, 4282, 2885],'ymax':[2296, 3005, 3671, 5168, 4191]})

#fig=sns.barplot(data=type1[0].transpose(), errwidth=1)
#fig=sns.barplot(data=type2.transpose(), errwidth=1)
#fig=sns.barplot(data=type3.transpose(), errwidth=1)
##
data = pd.read_excel(os.curdir + '/Alldata_Project_Retina.xlsx')
data = data[data['Year'].isin([2014])]
data = data[data['Experiment'].isin([9])]
#
amacrinectrl = DensRecovProf(data[data['Type'].isin([1])], 'X(micron)', 'Y(micron)', 180,10, 'exp9_14 Amacrine Chat',23, 'salmon', {'xmin':[1],'xmax':[1000], 'ymin':[1],'ymax':[1000]})
#plt.figure()
#fig=sns.barplot(data=amacrinectrl[0].transpose(), errwidth=1)

#1G1
#Plotting Founder line 2G10
data = pd.read_excel(os.curdir + '/Alldata_Project_Retina.xlsx', sheetname='CellCounter')
data = data[data['Year'].isin([2017])]
data = data[data['Experiment'].isin([19])]
data = data[data['Founder_line'].isin(['2G10'])]
data = data[data['Staining'].isin(['RBPMS CART'])]

type12 = data[data['Type'].isin([1,2])]
DensRecovProf(type12, 'X(micron)', 'Y(micron)', 180,10, 'exp19_17_FLRT3+ RGCs 2G10',23, 'dodgerblue', {'xmin':[851],'xmax':[2283], 'ymin':[3468],'ymax':[4355]})
type34 = data[data['Type'].isin([3,4])]
DensRecovProf(type34, 'X(micron)', 'Y(micron)', 180,10, 'exp19_17_FLRT3+ Amacrine 2G10',23, 'coral', {'xmin':[851],'xmax':[2283], 'ymin':[3468],'ymax':[4355]})

#2G10
#Plotting Founder line 1G1
data = pd.read_excel(os.curdir + '/Alldata_Project_Retina.xlsx', sheetname='CellCounter')
data = data[data['Year'].isin([2017])]
data = data[data['Experiment'].isin([20])]
data = data[data['Founder_line'].isin(['1G1'])]
data = data[data['Staining'].isin(['RBPMS CART'])]

type12 = data[data['Type'].isin([1,2])]
DensRecovProf(type12, 'X(micron)', 'Y(micron)', 180,10, 'exp20_17_FLRT3+ RGCs 1G1',23, 'dodgerblue', {'xmin':[1388],'xmax':[2564], 'ymin':[3479],'ymax':[4577]})
type34 = data[data['Type'].isin([3,4])]
DensRecovProf(type34, 'X(micron)', 'Y(micron)', 180,10, 'exp20_17_FLRT3+ Amacrine 1G1',23, 'coral', {'xmin':[1388],'xmax':[2564], 'ymin':[3479],'ymax':[4577]})




