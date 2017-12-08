# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 22:40:23 2016

Neuro Morphology analysis tool

@author: ruff
"""
from __future__ import print_function
import seaborn as sns
import neurom as nm
import pandas as pd
import glob
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
import pylab as p
from scipy import misc

from neurom.core.dataformat import COLS
import neurom as nm
from neurom import geom
from neurom.fst import sectionfunc
from neurom.core import Tree
from neurom.core.types import tree_type_checker, NEURITES
from neurom import morphmath as mm
from neurom import viewer
import brian2tools as b2
from brian2 import * #change line width in brian2tools morphology.py file

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

#from RetinaAnalysisTools.MorphAnalysis import StratifyAnalysis
import sys
sys.setrecursionlimit(10000) #otherwise error in plotting with brian2tools

#Don't forget to convert swc files so it can be read by neurom
###################################################
 # Functions for neuron morphology analysis#########        
def sec_len(sec):
    '''Return the length of a section'''
    return nm.morphmath.section_length(sec.points)

def n_points(sec):
    '''number of points in a section'''
    n = len(sec.points)
    # Non-root sections have duplicate first point
    return n if sec.parent is None else n - 1
   
# Determine Stratification of neurons
def StratifyAnalysis (data, filename):
    '''
    Takes Series of images with stratification pattern (binary) and Sums each row, bins into 10, Sums with all other images 
    of same series (=same cell) and plots an image using heatmap
    '''
    
    binnum = 10
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
            allstratify['stratify'] = stratify.stratify + allstratify.stratify
        allstratify = allstratify.stratify/allstratify.stratify.max()#normalize
        allstratify = pd.DataFrame(allstratify) #convert to dataframe
        allstratify = allstratify.transpose()
        #allstratify['SQL_ID1'] = image[12:30]
        allallstratify = pd.concat([allallstratify, allstratify])
    #allallstratify = allallstratify.rename(columns = {0.0:'Layer1',1.0:'Layer2',2.0:'Layer3',3.0:'Layer4',4.0:'Layer5'})
    allallstratify = allallstratify.rename(columns = {0.0:'Layer1',1.0:'Layer2',2.0:'Layer3',3.0:'Layer4',4.0:'Layer5',5.0:'Layer6',6.0:'Layer7',7.0:'Layer8',8.0:'Layer9',9.0:'Layer10'})#Use with 10 layers
    allallstratify = allallstratify.reset_index(drop=True) 
    
    PcaData = pd.DataFrame()                    #Make empty DataFrame for Principle component analysis
    PcaData['SQL_ID1'] = data.loc[:,'SQL_ID1']  #Add Filename SQL_ID in first column
    PcaData['SQL_ID4'] = data.loc[:,'SQL_ID4']
    PcaData['SomaSize'] = data.loc[:,'Area']    #Add Soma Size
    PcaData = pd.concat([PcaData, allallstratify], axis=1)#Add Stratification
    PcaData = PcaData.set_index(PcaData.SQL_ID1, drop=False)
    
    # Neuron Morphology analysis starts here:
    Tracefiles = glob.glob(os.curdir + '/InData/Morphology/*edit.swc') #load all SWC files
    for i in range(len(PcaData)):
        '''Datafile durchgehen und falls swc File exisists Wert berechnen und einfuegen'''
        Sql1 = PcaData.loc[PcaData.index[i],'SQL_ID1']
        Sql4 = PcaData.loc[PcaData.index[i],'SQL_ID4']
        File = os.curdir + '/InData/Morphology/' +Sql1 + Sql4
        if File in Tracefiles: #Wenn swc File exisitiert
            nrn = nm.load_neuron(File)         
            PcaData.loc[PcaData.index[i],'TotalNeuriteLength']=sum(sec_len(s) for s in nm.iter_sections(nrn))
            #PcaData.loc[PcaData.index[i],'NeuriteSurfaceArea']=sum(nm.morphmath.segment_area(s) for s in nm.iter_segments(nrn))  
            PcaData.loc[PcaData.index[i],'BifurcationPoints']=sum(1 for _ in nm.iter_sections(nrn, iterator_type=Tree.ibifurcation_point))
            PcaData.loc[PcaData.index[i],'Terminations']=nm.get('number_of_terminations', nrn)

            #PcaData.loc[PcaData.index[i],'MaxBranchOrder']=max(sectionfunc.branch_order(s) for s in nm.iter_sections(nrn))
            #PcaData.loc[PcaData.index[i],'GeomBoundBox']=geom.bounding_box(nrn)
            # Add Sholl analysis
            
#    #### Do Principle component analysis    
#    PcaData.fillna(inplace=True, method='backfill')
#    X = PcaData[PcaData.columns[3:13]].values
#    X = scale(X)           
#    Pca = PCA(n_components = len(datam))
#    X = Pca.fit_transform(X)
#    
#    #colors = ['navy', 'turquoise', 'darkorange']
#    
#    #plt.figure(figsize=(8, 8))
#    fig=plt.figure()
#    #fig.patch.set_facecolor('white')
#    
#    #ax = fig.add_subplot(111, projection='3d')
#    ax=plt.subplot(1,2,1)
#    ax.scatter(XPca[:,0], X[:,1], lw=0.25, s=4)
#    ax.set_facecolor('white')
#    ax.grid(b=False)

    
    ####SORT DATA according to stratification pattern and then according to # of terminations##########
    PcaData10 = PcaData[PcaData.Layer10==1]
    PcaData9 = PcaData[PcaData.Layer9==1]
    PcaData8 = PcaData[PcaData.Layer8==1]
    PcaData7 = PcaData[PcaData.Layer7==1]
    PcaData6 = PcaData[PcaData.Layer6==1]    
    
    PcaData5 = PcaData[PcaData.Layer5==1]
    PcaData4 = PcaData[PcaData.Layer4==1]
    PcaData3 = PcaData[PcaData.Layer3==1]
    PcaData2 = PcaData[PcaData.Layer2==1]
    PcaData1 = PcaData[PcaData.Layer1==1]    
    #PcaData = PcaData1.append([PcaData2, PcaData3, PcaData4, PcaData5])
    
    PcaData10.sort(columns = 'TotalNeuriteLength', axis=0, inplace=True, ascending=False)
    PcaData9.sort(columns = 'TotalNeuriteLength', axis=0, inplace=True, ascending=False)
    PcaData8.sort(columns = 'TotalNeuriteLength', axis=0, inplace=True, ascending=False)
    PcaData7.sort(columns = 'TotalNeuriteLength', axis=0, inplace=True, ascending=False)
    PcaData6.sort(columns = 'TotalNeuriteLength', axis=0, inplace=True, ascending=False)
    PcaData5.sort(columns = 'TotalNeuriteLength', axis=0, inplace=True, ascending=False)
    PcaData4.sort(columns = 'TotalNeuriteLength', axis=0, inplace=True, ascending=False)
    PcaData3.sort(columns = 'TotalNeuriteLength', axis=0, inplace=True, ascending=False)
    PcaData2.sort(columns = 'TotalNeuriteLength', axis=0, inplace=True, ascending=False)
    PcaData1.sort(columns = 'TotalNeuriteLength', axis=0, inplace=True, ascending=False)


    PcaData = PcaData1.append([PcaData2, PcaData3, PcaData4, PcaData5, PcaData6, PcaData7, PcaData8, PcaData9, PcaData10])#Use for 10 bins
    ###################################################
    
    ###########################Plotting Figures#################################
    Tracefiles = glob.glob(os.curdir + '/InData/Morphology/*edit.swc') #load all SWC files
    # Set the font dictionaries (for plot title and axis titles)
    #fig, ax = plt.subplots(10,10, sharex=True, sharey=True) 
    fig=plt.figure()
    fig.patch.set_facecolor('k')
    colors=['red','darksalmon','white','darkgrey','cyan','dodgerblue','green','lightgreen','magenta','pink']
    for i in range(len(PcaData)):
        Sql1 = PcaData.loc[PcaData.index[i],'SQL_ID1']
        Sql4 = PcaData.loc[PcaData.index[i],'SQL_ID4']
        File = os.curdir + '/InData/Morphology/' + Sql1 + Sql4
        ax=plt.subplot(10,10,i+1) 
        ax.set_xlim([0,300])
        ax.set_ylim([0,300])
        ax.set_facecolor('k')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(b=False)
        cell = str(i)
        #ax.set_title(File[23:38] + ' Cell # ' + cell, fontsize=3, color='dimgrey') #full title
        ax.set_title('Cell# ' + cell, fontsize=4, color='white')#cell number only
        
        
        if File in Tracefiles:
            #nrn = nm.load_neuron(File)
            print(File)
            morpho=Morphology.from_file(File)
            if PcaData.loc[PcaData.index[i],'Layer1']==1:
                color = colors[0]
                
            if PcaData.loc[PcaData.index[i],'Layer2']==1:
                color = colors[1]
                
            if PcaData.loc[PcaData.index[i],'Layer3']==1:
                color = colors[2]
                
            if PcaData.loc[PcaData.index[i],'Layer4']==1:
                color = colors[3]                
                
            if PcaData.loc[PcaData.index[i],'Layer5']==1:
                color = colors[4]
                
            if PcaData.loc[PcaData.index[i],'Layer6']==1:
                color = colors[5]
                
            if PcaData.loc[PcaData.index[i],'Layer7']==1:
                color = colors[6]
                
            if PcaData.loc[PcaData.index[i],'Layer8']==1:
                color = colors[7]
                
            if PcaData.loc[PcaData.index[i],'Layer9']==1:
                color = colors[8]
                
            if PcaData.loc[PcaData.index[i],'Layer10']==1:
                color = colors[9]
                
#            x=round(i-(np.floor(i/10)*10)) #X coordinate for subplot
#            y=round(np.ceil(i/10)-1)        #Y coordinate for subplot
            #nm.view.view.plot_neuron(ax[x,y], nrn, soma_outline=False, linewidth=0.001, alpha=1, color='k') #Blue BRain Project plotting
            b2.plot_morphology(morpho, plot_3d=False, show_diameter=False, show_compartments=False, colors=(color,color))            
            #b2.plot_dendrogram(morpho)
            
    p.savefig(os.curdir + '/Figures/'  +'Morphology '+ filename +'.pdf', frameon= True, transparent=False, facecolor='k')

#            b2.plot_morphology(morpho, axes = ax[x,y], plot_3d=False, show_diameter=False, show_compartments=False, colors=('k','k'))
#            ax[x,y].grid(b=False)
#            ax[x,y].set_facecolor('white')
#            ax[x,y].set_title(File[-21:-39])
#            ax[x,y].set_xlim([0,300])
#            ax[x,y].set_ylim([0,300])
            #ax[x,y].title(File[-21:-39])
            #ax=plt.subplot(10,10,i) 
            #b2.plot_morphology(morpho, plot_3d=False, show_compartments=False, show_diameter=False)
            


#            matplotlib.rcParams.update({'font.size': 5})
            #neurom.view.view.plot_dendrogram(ax[x,y], nrn, show_diameters=False)
            
    return(PcaData)


#Add more colors for each sublayer

#Filter for celltype (choose either RGC or Amacrine) #############################
os.chdir('/Users/ruff/OneDrive/Retina Python Tools/DataV2') 
                     
data = pd.read_excel(os.curdir + '/Alldata_Project_Retina.xlsx', sheetname='CellMorphology')
data = data[data['Year'].isin([2017])]
data = data[data['Experiment'].isin([14,22,26,27])]
data = data[data['Sub_Type'].isin(['RGC'])]
data = data.reset_index(drop=True)

RgcData = StratifyAnalysis (data, 'RGCs')

data = pd.read_excel(os.curdir + '/Alldata_Project_Retina.xlsx', sheetname='CellMorphology')
data = data[data['Year'].isin([2017])]
data = data[data['Experiment'].isin([14,22,26,27])]
data = data[data['Sub_Type'].isin(['Amacrine'])]
data = data.reset_index(drop=True)

AmacData = StratifyAnalysis (data, ' Amacrines')   #Stratification properties
                           
#Load Sholl Data
sholl = pd.read_csv(os.curdir + '/InData/Sholl/Intersections all neurons.txt', delimiter=',',error_bad_lines=False )                          
##############################################

####PCA Analysis###########################################
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

RgcData.fillna(inplace=True, method='backfill')
datam = RgcData.columns[10:13]
X = RgcData[datam].values
X = scale(X)           
           
Pca = PCA(n_components = len(datam))
XPca = Pca.fit_transform(X)

#colors = ['navy', 'turquoise', 'darkorange']

#plt.figure(figsize=(8, 8))
fig=plt.figure()
#fig.patch.set_facecolor('white')

#ax = fig.add_subplot(111, projection='3d')
ax=plt.subplot(1,2,1, projection='3d')
ax.scatter(XPca[:,0], XPca[:,1],XPca[:,2], lw=0.25, s=4)
ax.set_facecolor('white')
ax.grid(b=False)


kmean = KMeans(n_clusters=3, max_iter=1000, random_state=1000).fit_predict(X)

ax=plt.subplot(1,2,2)
ax.scatter(X[:, 0], X[:, 1], c=kmean, cmap=plt.cm.hot, s=4)
ax.set_facecolor('white')
ax.grid(b=False)


plt.show()
#Retrieve Principle most important features for most important components
###########################################################  



#####Frequency distribution of SomaSize Plotted with cntrl data############
##RGC control data
RgcCtrl = pd.read_excel(os.curdir + '/Alldata_Project_Retina.xlsx', sheetname='CellSize')
RgcCtrl = RgcCtrl[RgcCtrl['Year'].isin([2016])]
RgcCtrl = RgcCtrl[RgcCtrl['Experiment'].isin([89])]
RgcCtrl = RgcCtrl[RgcCtrl['Type'].isin([2])]
RgcCtrl = RgcCtrl.reset_index(drop=True)

##Amacrine control data
AmacCtrl = pd.read_excel(os.curdir + '/Alldata_Project_Retina.xlsx', sheetname='CellSize')
AmacCtrl = AmacCtrl[AmacCtrl['Year'].isin([2016])]
AmacCtrl = AmacCtrl[AmacCtrl['Experiment'].isin([89])]
AmacCtrl = AmacCtrl[AmacCtrl['Type'].isin([1])]
AmacCtrl = AmacCtrl.reset_index(drop=True)

#####Plotting data###############################################################################
    ###plotting RGC data
plt.figure()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.001, hspace=None)

ax1=plt.subplot(182)
ax1.grid(False)
plt.title('Soma', rotation=90, y=1.09) #y defines distance of title
plt.imshow(RgcData[['SomaSize']], cmap='Reds' ,vmin=0, vmax=400, aspect=0.5)

ax2=plt.subplot(183)
ax2.grid(False)
plt.title('NLen', rotation=90, y=1.09)
plt.imshow(RgcData[['TotalNeuriteLength']], cmap='Blues' ,vmin=0, vmax=5000, aspect=0.5)

ax2=plt.subplot(184)
ax2.grid(False)
plt.title('Term', rotation=90, y=1.09)
plt.imshow(RgcData[['Terminations']], cmap='Greens' ,vmin=0, vmax=100, aspect=0.5)

ax2=plt.subplot(185)
ax2.grid(False)
plt.title('Bifur', rotation=90, y=1.09)
plt.imshow(RgcData[['BifurcationPoints']], cmap='Purples' ,vmin=0, vmax=100, aspect=0.5)

ax4=plt.subplot(181)
ax4.grid(False)
plt.title('Strat', rotation=90, y=1.09)
plt.imshow(RgcData[['Layer1','Layer2','Layer3','Layer4','Layer5','Layer6','Layer7','Layer8','Layer9','Layer10']], cmap='Reds', interpolation='nearest', aspect=0.5)

ax3=plt.subplot(188)
ax3.grid(False)
plt.title('RGC Size', rotation=0, y=1.09)
plt.style.use('seaborn-white')
ax3=sns.distplot(RgcCtrl.Area, bins=5, hist=False, rug=False, kde=True, norm_hist=False, hist_kws=dict(alpha=0.5), kde_kws={"shade": True})
ax3=sns.distplot(RgcData.SomaSize, bins=5, hist=False, rug=False, kde=True, norm_hist=False, hist_kws=dict(alpha=0.5), kde_kws={"shade": True}, color='r')   
    ###plotting Amacrine data
plt.figure()
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=None)

ax1=plt.subplot(182)
ax1.grid(False)
plt.title('Soma')
plt.imshow(AmacData[['SomaSize']], cmap='Reds' ,vmin=0, vmax=400, aspect=0.5)

ax2=plt.subplot(183)
ax2.grid(False)
plt.title('NLen')
plt.imshow(AmacData[['TotalNeuriteLength']], cmap='Blues' ,vmin=0, vmax=5000, aspect=0.5)

ax2=plt.subplot(184)
ax2.grid(False)
plt.title('Term')
plt.imshow(AmacData[['Terminations']], cmap='Greens' ,vmin=0, vmax=100, aspect=0.5)

ax2=plt.subplot(185)
ax2.grid(False)
plt.title('Bifur')
plt.imshow(AmacData[['BifurcationPoints']], cmap='Purples' ,vmin=0, vmax=100, aspect=0.5)

ax4=plt.subplot(181)
ax4.grid(False)
plt.title('Strat')
plt.imshow(AmacData[['Layer1','Layer2','Layer3','Layer4','Layer5','Layer6','Layer7','Layer8','Layer9','Layer10']], cmap='Reds', interpolation='nearest', aspect=0.5)

ax5=plt.subplot(188)
ax5.grid(False)
plt.title('Amac Size')
plt.style.use('seaborn-white')
ax5=sns.distplot(AmacCtrl.Area, bins=5, hist=False, rug=False, kde=True, norm_hist=False, hist_kws=dict(alpha=0.5), kde_kws={"shade": True})
ax5=sns.distplot(AmacData.SomaSize, bins=5, hist=False, rug=False, kde=True, norm_hist=False, hist_kws=dict(alpha=0.5), kde_kws={"shade": True}, color='r')

#Use for 10 bins #plt.imshow(PcaDataSorted[['Layer1','Layer2','Layer3','Layer4','Layer5', 'Layer6','Layer7','Layer8','Layer9','Layer10']], cmap='viridis', interpolation='nearest', aspect=1.8) 
p.savefig(os.curdir + '/Figures/'  +'RGC Stratification'+'.png', frameon= False, transparent=False)


#################################################################################################

from neurom import viewer

fig, ax = viewer.draw(nrn)
fig.show()
fig, ax = viewer.draw(nrn, mode='3d') # valid modes '2d', '3d', 'dendrogram'
fig.show()