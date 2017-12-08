#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 18:19:13 2017

@author: ruff
"""

import seaborn as sns
import neurom as nm
import pandas as pd
import glob
import math
import matplotlib.pyplot as plt
import os
import numpy as np
import pylab as p
from scipy import misc
from __future__ import print_function

from neurom.core.dataformat import COLS
import neurom as nm
from neurom import geom
from neurom.fst import sectionfunc
from neurom.core import Tree
from neurom.core.types import tree_type_checker, NEURITES
from neurom import morphmath as mm
import numpy as np
#Don't forget to convert swc files so it can be read by neurom

os.chdir('/Users/ruff/OneDrive/Retina Python Tools/DataV2') 

data = pd.read_excel(os.curdir + '/Alldata_4D.xlsx')

plt.figure()
sns.boxplot(x='Pivot', y='Relative_Size', data=data, whis=np.inf, width=0.2, color='c')
sns.stripplot(x='Pivot', y='Relative_Size', data=data, jitter=True, size=4, color='.1', linewidth=0)

sns.boxplot(x='Pivot', y='Relative_Weight', data=data, whis=np.inf, width=0.2, color='r')
sns.stripplot(x='Pivot', y='Relative_Weight', data=data, jitter=True, size=4, color='.1', linewidth=0)

sns.boxplot(x='Injection?', y='Relative_Size', data=data, whis=np.inf, width=0.2, color='c')
sns.stripplot(x='Injection?', y='Relative_Size', data=data, jitter=True, size=4, color='.1', linewidth=0)

sns.boxplot(x='Injection?', y='Relative_Weight', data=data, whis=np.inf, width=0.2, color='r')
sns.stripplot(x='Injection?', y='Relative_Weight', data=data, jitter=True, size=4, color='0.1', linewidth=0)
