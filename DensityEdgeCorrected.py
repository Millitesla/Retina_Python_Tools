#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 20:52:30 2016

@author: ruff
Returns Edge corrected densities of dots. If 
"""

import pandas as pd
import glob
import math
import matplotlib.pyplot as plt
import os
import numpy as np
import pylab as p
from scipy import misc

os.chdir('/Users/filepath')

'''Select Experimental data from big dataframe'''

data = pd.read_excel(os.curdir + '/Alldata_Project_Retina.xlsx')
coordinates = data
coordinates = coordinates.loc[data['Year'].isin([2016])]
coordinates = coordinates.loc[data['Experiment'].isin([90])]
coordinates = coordinates.loc[data['Type'].isin([1,2])]
coordinates.reset_index(inplace=True, drop=True)

#Get Binary file out of dataframe    
binary = misc.imread(os.curdir + '/InFigures/' + coordinates.loc[1,'Associated_File1'])
binary = binary/255#normalize to make binary
    
#Input for function
Radius = 120 #in um
ConversionFactor = 0.7207960
RadPixl = Radius/ConversionFactor

'''Calculate number of cells within InRadius around each cell'''
mx = np.array(coordinates['X']) #take X coordinates into np array
mx_transpose = np.tile(mx, (np.shape(mx)[0],1)) #broadcast columns and transpose
mx_normal = np.transpose(mx_transpose) #broadcast them over #of rows
delta_mx = mx_normal - mx_transpose #calculate difference from every cell to any other cell

my = np.array(coordinates['Y']) #take X coordinates into np array
my_transpose = np.tile(my, (np.shape(my)[0],1)) #broadcast columns and transpose
my_normal = np.transpose(my_transpose) #broadcast them over #of rows
delta_my = my_normal - my_transpose #calculate difference from every cell to any other cell

distances = ((delta_mx)**2+(delta_my)**2)**0.5 #calculate distances from each cell to each other cell
sum_distances = (distances<RadPixl).sum(axis=1) # count how many cells are within Radius of each cell

'''Calculate valid area for each cell to calculate correct density'''
Retina = binary #Array generated from binary image of Retina
xDim = np.shape(Retina)[1]#Determine Xdimension of image
yDim = np.shape(Retina)[0]#Determine Ydimension of image
Area = np.zeros(np.shape(mx)[0])# Create empty numpy array to fill in areas

for s in range(np.shape(coordinates)[0]):
    #print ("\r", str(np.round(s*100./(len(coordinates)-1),1)), "%"), 
    sys.stdout.write("\r{0}".format((float(s)/(len(coordinates)-1))*100))
    sys.stdout.flush()
    x = int(mx[s]) # xKoordinate des Punktes
    y = int(my[s]) # yKoordinate des Punktes
    
    xMin = x - RadPixl
    xMax = x + RadPixl
    yMin = y - RadPixl
    yMax = y + RadPixl
    
    if xMin < 0:
        xMin = 0
    else: 
        pass
    if yMin < 0:
        yMin = 0
    else:
        pass
    if xMax < 0:
        xMax = 0
    else:
        pass
    if yMax < 0:
        yMax = 0
    
    if xMax > xDim:
        xMax = xDim
    else:
        pass
    if yMax > yDim:
        yMax = yDim
    else:
        pass      #special case: Dot/cell/coordinate is at the edge of field
    
    xRange = np.linspace(xMin, xMax, xMax+1-xMin)# array from xMin to xMax
    yRange = np.linspace(yMin, yMax, yMax+1-yMin)# array from yMin to yMax
    
    Cell = np.zeros((yDim, xDim))#Create empty Numpy Array for cell
    for i in xRange:
        for j in yRange:
            TempRadius = ((x-i)**2 + (y-j)**2)**0.5
            if TempRadius <= RadPixl:
                Cell[j-1,i-1] = 1 #set Arrayposition Cell i,j to 1)
            else:
                pass#set Arrayposition i,j to 0
    CellArea = Cell*Retina # Multiply Cell Array with Retina Array to get only 1 in cells where both arrays contain 1
    TempArea = np.sum(CellArea)*(ConversionFactor)**2#Calculate Area: Divide Product of Count of SubtractedCell* conversion factor / by area calculated from r^2pi   
    Area[s] = TempArea
Area = Area*0.000001 #Convert Area to mm^2

#Calculate Densities
Densities = sum_distances/Area

fig=plt.figure()
plt.style.use('seaborn-white')
ax = fig.add_subplot(111, aspect='equal') # make aspect ratio equal
ax.grid(False) 
plt.scatter(mx,my, c=Densities, cmap='viridis',  edgecolors='none',alpha=1, s=10)#, vmin=minimum, vmax=maximum)
plt.colorbar()
numsize=23
plt.xlabel('µm', fontsize=numsize)
plt.ylabel('µm', fontsize=numsize)
plt.xticks(fontsize=numsize, rotation=45)
plt.yticks(fontsize=numsize)

plt.tight_layout() #prohibits labels outside figure boarders
p.savefig(os.curdir + '/Figures/'  + 'exp90_16.svg', frameon= False, transparent= True)


fig=plt.figure()
plt.imshow(Retina)


