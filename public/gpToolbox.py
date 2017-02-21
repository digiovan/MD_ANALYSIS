
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

        # the next 3 lines are my personal modification. Increased robustness to my data...
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
    

    @staticmethod        
    # thanks to the mitical Guido. Modified the linspace part
    def computeTransverseEmittance(WS_position_um,
                                   WS_profile_arb_unit,
                                   off_momentum_distribution_arb_unit,
                                   deltaP_P,
                                   betaGammaRelativistic,
                                   betaOptical_m,
                                   Dispersion_m):

        x_inj=WS_position_um/1000;
        y_inj=WS_profile_arb_unit;
        
        
        popt,pcov = gpToolbox.makeGaussianFit_5_parameters(x_inj,y_inj)
        y_inj_1=gpToolbox.Gaussian_5_parameters(x_inj,popt[0],popt[1],popt[2],popt[3],popt[4])
        y_inj_2=y_inj-popt[0]-popt[1]*x_inj
        x_inj_2=x_inj-popt[3]
        
        # define a reasonable range for the interpolation
        limit = 5 * popt[4]
        #print limit 
        x_inj_3=np.linspace(-limit,limit,1000)
        # uncomment for hardcoded range
        #x_inj_3=np.linspace(-40,40,1000);

        y_inj_3=scipy.interpolate.interp1d(x_inj_2,y_inj_2)(x_inj_3)
        y_inj_4=y_inj_3/np.trapz(y_inj_3,x_inj_3)
        y_inj_5=(y_inj_4+y_inj_4[::-1])/2

        WS_profile_step1_5GaussianFit=y_inj_1
        WS_profile_step2_dropping_baseline=y_inj_2
        WS_profile_step3_interpolation=y_inj_3
        WS_profile_step4_normalization=y_inj_4
        WS_profile_step5_symmetric=y_inj_5
        WS_position_step1_centering_mm=x_inj_2;
        WS_position_step2_interpolation_mm=x_inj_3;
        Dispersion_mm=Dispersion_m*1000

        Dispersive_position_step1_mm=deltaP_P*Dispersion_mm
        Dispersive_profile_step1_normalized=off_momentum_distribution_arb_unit/np.trapz(off_momentum_distribution_arb_unit,Dispersive_position_step1_mm)
        Dispersive_position_step2_mm=WS_position_step2_interpolation_mm
        Dispersive_step2_interpolation=scipy.interpolate.interp1d(Dispersive_position_step1_mm,Dispersive_profile_step1_normalized,bounds_error=0,fill_value=0)(Dispersive_position_step2_mm)
        Dispersive_step3_symmetric=(Dispersive_step2_interpolation+Dispersive_step2_interpolation[::-1])/2

        def myConvolution(WS_position_step2_interpolation_mm,sigma):
            myConv=np.convolve(Dispersive_step3_symmetric, gpToolbox.Gaussian(WS_position_step2_interpolation_mm,1,0,sigma), 'same')
            myConv/=np.trapz(myConv,WS_position_step2_interpolation_mm)
            return myConv

        def myError(sigma):
            myConv=np.convolve(Dispersive_step3_symmetric, gpToolbox.Gaussian(WS_position_step2_interpolation_mm,1,0,sigma), 'same')
            myConv/=np.trapz(myConv,WS_position_step2_interpolation_mm)
            aux=myConv-WS_profile_step5_symmetric
            return np.std(aux), aux, myConv

        popt,pcov = curve_fit(myConvolution,WS_position_step2_interpolation_mm,WS_profile_step5_symmetric,p0=[1])
        sigma=popt;
        emittance=sigma**2/betaOptical_m*betaGammaRelativistic
        return {'emittance_um':emittance,'sigma_mm':sigma,'WS_position_mm':WS_position_step2_interpolation_mm, 'WS_profile': WS_profile_step5_symmetric, 'Dispersive_position_mm':Dispersive_position_step2_mm, 'Dispersive_profile':Dispersive_step3_symmetric,
               'convolutionBackComputed':myConvolution(WS_position_step2_interpolation_mm,sigma),
               'betatronicProfile':gpToolbox.Gaussian(WS_position_step2_interpolation_mm,1,0,sigma)
               }

