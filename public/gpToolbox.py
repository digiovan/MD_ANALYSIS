
# coding: utf-8

# In[1]:

print('Version 0.01.')
print('Use it at your risk. In case please report any bug to digiovan@cern.ch.')

# imports taken from Guido's toolbox
get_ipython().magic('matplotlib inline')
import os
import glob
import scipy.io
import datetime   
import pickle     
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.dates as md
import matplotlib
import pandas as pnd
import platform 
import math
import sys
import time
from IPython.display import Image, display
from scipy.optimize import curve_fit
#!git clone https://github.com/rdemaria/pytimber.git
#sys.path.insert(0,'/eos/user/s/sterbini/MD_ANALYSIS/public/pytimber/pytimber')
import pytimber;
#pytimber.__file__
try:
    import seaborn as sns
except:
    print('If you want to use "seaborn" package, install it from a SWAN terminal "pip install --user seaborn"')


# In[2]:

# dictionary containing the dispersion as measured in 2016
dispersion_measured = {}

# rings
dispersion_measured['r1'] = {}
dispersion_measured['r2'] = {}
dispersion_measured['r3'] = {}
dispersion_measured['r4'] = {}

# planes
dispersion_measured['r1']['H'] = {}
dispersion_measured['r1']['V'] = {}

dispersion_measured['r2']['H'] = {}
dispersion_measured['r2']['V'] = {}

dispersion_measured['r3']['H'] = {}
dispersion_measured['r3']['V'] = {}

dispersion_measured['r4']['H'] = {}
dispersion_measured['r4']['V'] = {}

# populate the dictionary

## ring1
dispersion_measured['r1']['H']['WS']  = -1.3619   # m
dispersion_measured['r1']['H']['SG1'] = -1.5515   # m
dispersion_measured['r1']['H']['SG2'] = -0.052911 # m 
dispersion_measured['r1']['H']['SG3'] = +1.4290   # m

dispersion_measured['r1']['V']['WS']  = -0.0919   # m
dispersion_measured['r1']['V']['SG1'] = +0.32782  # m
dispersion_measured['r1']['V']['SG2'] = -0.0262   # m 
dispersion_measured['r1']['V']['SG3'] = -0.39807  # m


## ring2
dispersion_measured['r2']['H']['WS']  = -1.3884   # m
dispersion_measured['r2']['H']['SG1'] = -1.2368   # m
dispersion_measured['r2']['H']['SG2'] = +0.038622 # m 
dispersion_measured['r2']['H']['SG3'] = +1.2936   # m

dispersion_measured['r2']['V']['WS']  = -0.0114   # m
dispersion_measured['r2']['V']['SG1'] = +0.25384  # m
dispersion_measured['r2']['V']['SG2'] = +0.062838 # m 
dispersion_measured['r2']['V']['SG3'] = -0.12573  # m


## ring3
dispersion_measured['r3']['H']['WS']  = -1.3634   # m
dispersion_measured['r3']['H']['SG1'] = -1.1798   # m
dispersion_measured['r3']['H']['SG2'] = +0.039221 # m 
dispersion_measured['r3']['H']['SG3'] = +1.2243   # m

dispersion_measured['r3']['V']['WS']  = -0.0299   # m
dispersion_measured['r3']['V']['SG1'] = +0.029354 # m
dispersion_measured['r3']['V']['SG2'] = +0.027425 # m 
dispersion_measured['r3']['V']['SG3'] = +0.04821  # m


## ring4
dispersion_measured['r4']['H']['WS']  = -1.3827   # m
dispersion_measured['r4']['H']['SG1'] = -1.5313   # m
dispersion_measured['r4']['H']['SG2'] = -0.14812  # m 
dispersion_measured['r4']['H']['SG3'] = +1.1998   # m

dispersion_measured['r4']['V']['WS']  = -0.1101   # m
dispersion_measured['r4']['V']['SG1'] = +0.034784 # m
dispersion_measured['r4']['V']['SG2'] = +0.14163  # m 
dispersion_measured['r4']['V']['SG3'] = +0.30977  # m

#print dispersion_measured
# an image is available at ![measured dispersion in 2016](/eos/user/d/digiovan/MD_ANALYSIS/public/img/dispersion_measured_in_2016.png)


# In[3]:

class gpToolbox:
    
    @staticmethod    
    def Gaussian(x, A, mu, sig):
        """Gaussian(x, A, mu, sig)"""
        return A/np.sqrt(2*np.pi)/sig*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        #return A*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    @staticmethod    
    # from Guido's toolbox
    def Gaussian_5_parameters(x, c, m, A, mu, sig):
        """gaussian_5_parameter(x, c, m, A, mu, sig)"""
        return c+m*x+A/np.sqrt(2*np.pi)/sig*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    
    
    @staticmethod
    # thanks to http://stackoverflow.com/questions/19206332/gaussian-fit-for-python
    def makeGaussianFit(X,Y):
        i = np.where(Y>0)

        X=X[i].astype(float)
        Y=Y[i].astype(float)
        
        # correction for weighted arithmetic mean
        mean = sum(X * Y) / sum(Y)
        sigma = np.sqrt(sum(Y * (X - mean)**2) / sum(Y))

        popt,pcov = curve_fit(gpToolbox.Gaussian, X, Y, p0=[max(Y), mean, sigma])

        return popt, pcov
    
    @staticmethod        
    #thanks to Hannes, but modified
    def makeGaussianFit_5_parameters(X,Y):     
        i = np.where( (X>min(X)+1e-3) & (X<max(X)-1e-3) )
        X = X[i]
        Y = Y[i]

        i = np.where(Y>0)
        X = X[i].astype(float)
        Y = Y[i].astype(float)

        indx_max = np.argmax(Y)
        mu0 = X[indx_max]
        window = 2*100
        x_tmp = X[indx_max-window:indx_max+window]
        y_tmp = Y[indx_max-window:indx_max+window]
        offs0 = min(y_tmp)
        ampl = max(y_tmp)-offs0
        x1 = x_tmp[np.searchsorted(y_tmp[:window], offs0+ampl/2)]
        x2 = x_tmp[np.searchsorted(-y_tmp[window:], -offs0+ampl/2)]
        FWHM = x2-x1
        sigma0 = np.abs(2*FWHM/2.355)
        ampl *= np.sqrt(2*np.pi)*sigma0
        slope = 0
        popt,pcov = curve_fit(gpToolbox.Gaussian_5_parameters,X,Y,p0=[offs0,slope,ampl,mu0,sigma0])
        return popt,pcov
    

