"""
------------------------------------------------------
Python Script that two plots of the fit results:
1. Model spectrum & its components and the IRS spectrum
2. Mass fractions of dust species

RUNNING:
$ python3 PlotResults.py [starname] [output directory]
[starname]  = name of starsystem to be fitted
[output directory] = folder in which MultiNest data is created

E.g.
python3 FunctionsFittingRens.py AA-Tau /home/your/folders/MNPPD/Output/long/AA-Tau/full_example/

"""

import pandas as pd
import sys
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy import interpolate
from scipy.stats import chisquare
from timeit import default_timer as timer
from scipy.optimize import curve_fit
import functions as fn
from model import Model, Spectral

s = Spectral(sys.argv[1])
s_norm = Spectral(sys.argv[1])
s_norm.cut_spectrum(end=17)
s.cut_spectrum(begin=5.24)
if sys.argv[1] =='DF-Tau':     #Removal of wavelengths with outlying fluxes
    s.cut_spectrum(begin=5.54)
    s_norm.cut_spectrum(begin=5.54)

MN_path = sys.argv[2]
print(MN_path)
if (MN_path.find('long') != -1):

    print(sys.argv[2])

    """new data retrieval from Analyzer"""
    Inner_Rim = pd.DataFrame(np.array([fn.get_1sigma(sys.argv[2],'C1','short'),fn.get_1sigma(sys.argv[2],'Trim_min','short')
                            ,fn.get_1sigma(sys.argv[2],'qrim','short')]),columns=['Mean','Sigma','sh'])
    Inner_Rim = Inner_Rim['Mean']
    """--------------------------------"""
    print(Inner_Rim)
    n_params = 20
    s.cut_spectrum(17, 35)
    region = 'long'
else:
    n_params = 23
    s.cut_spectrum(end=17)
    print('short wave spectrum selected',s.spitzer['wavelength'])
    region = 'short'
    print("Short wavelengths")

factor = s_norm.spitzer['flux'].max()/s.spitzer['flux'].max()
print(s.spitzer['wavelength'])
s.setunity()
s.rebin_starfit()
s.read_opacity()
s.restore_index()
MN_path = sys.argv[2]
MN_output = fn.get_all1sigma(sys.argv[2],n_params,region)
# MN_output = pd.read_csv(MN_path,sep='\s+',skiprows=2,nrows=n_params,index_col=False,usecols=[1,2])
# MN_output.columns = ['Mean','Sigma']
print(MN_output)
statistic = pd.read_csv(MN_path +"/test-stats.dat",sep='\s+',nrows=1)
loglike = round(float(statistic.columns[5]))
parameters_values = MN_output['Mean']

try:
    parameters_values= pd.concat([Inner_Rim,parameters_values],ignore_index=True)
except:
    print("shortwave")
print(parameters_values)
# quit()
start = timer()
f = Model(s.spitzer['wavelength'],s.opacities)
f.define_parameters(parameters_values)

print(s.spitzer['wavelength'].max())
if s.spitzer['wavelength'].max()>=17:
    print(factor)
    f.longwave_rim(factor,s_norm.spitzer['wavelength'])
    print('I am in the long wavelength part of the script!')
    f.compute_longwave()
else:
    f.compute()
print(timer()-start)

"""---------------------------------------------"""
"""PLOTS"""
fig,axes = plt.subplots(2,1,figsize=(8,8),gridspec_kw={'height_ratios': [6, 2]})

axes[0].errorbar(s.spitzer['wavelength'],s.spitzer['flux'],s.spitzer['sigma'],label='Spitzer',color='black')

axes[0].plot(s.spitzer['wavelength'], s.stellar,label='star',color='green')
axes[0].plot(s.spitzer['wavelength'], f.components['midplane'],label='Midplane')
axes[0].plot(s.spitzer['wavelength'], f.components['inner_rim'],label='Inner Rim')
axes[0].plot(s.spitzer['wavelength'], f.components['atmosphere'],label = 'MODEL dust',color='orange')

axes[0].plot(s.spitzer['wavelength'], f.components['disk_model']+s.stellar,linewidth=2, label = 'MODEL',color='r')

axes[1].plot(s.spitzer['wavelength'],(s.spitzer['flux']-f.components['disk_model']-s.stellar)/s.spitzer['flux'],label = 'Residual',color='black')

axes[0].set_ylabel('$\\frac{F}{F_{max}}$',size=20)
axes[1].set_xlabel('$\lambda$ ($\mu$m)')
axes[1].set_ylabel('$\\frac{\Delta F}{F}$')
axes[0].legend()
axes[1].legend()
axes[0].set_title('Log-Likelihood: ' + str(loglike),fontsize=10 )
if n_params == 23:
    axes[0].set_title('5-17 $\mu m$' )
else:
    axes[0].set_title('17-35 $\mu m$' )
plt.subplots_adjust(hspace=0)
plt.show()
fig.savefig(sys.argv[2] + '/fitfig.png', dpi=fig.dpi)

"""Dust Composition plot"""
kap = [0.1*3,2*3,5*3,0.1*2.8,2*2.8,5*2.8,0.1*3.2,2*3.2,0.1*2.8,2*2.8,0.1*2.2,2*2.2,5*2.2]
dust_parameters,dust_sigma_min, dust_sigma_plus = fn.dust_sigma(MN_output.iloc[-13:])
print(MN_output.iloc[-13:])
fig2,axes = plt.subplots(1,1,figsize=(8,8))
olivine = axes.bar(['.1 ','2 ','5 '],dust_parameters[0:3],yerr=[dust_sigma_min[0:3],dust_sigma_plus[0:3]]
            ,color='r',label='Olivine')
pyroxene = axes.bar([' .1 ',' 2 ',' 5 '],dust_parameters[3:6],yerr=[dust_sigma_min[3:6],dust_sigma_plus[3:6]]
            ,color='b',label='Pyroxene')
forsterite = axes.bar(['.1','2'],dust_parameters[6:8],yerr=[dust_sigma_min[6:8],dust_sigma_plus[6:8]]
            ,color='g',label='Forsterite')
enstatite = axes.bar([' .1',' 2'],dust_parameters[8:10],yerr=[dust_sigma_min[8:10],dust_sigma_plus[8:10]]
            ,color='y',label='Enstatite')
silica = axes.bar([' .1  ',' 2  ',' 5  '],dust_parameters[10:],yerr=[dust_sigma_min[10:],dust_sigma_plus[10:]]
            ,color='purple',label='Silica')

axes.set_ylim(bottom=0,top=1)
axes.set_ylabel('Mass fraction')
axes.set_xlabel('Grain size ($\mu$m)')
axes.set_title(sys.argv[1])
axes.set_title('Log-Likelihood: ' + str(loglike),fontsize=10)
axes.legend(fontsize=20)
plt.xticks(fontsize=15)
plt.show()
fig2.savefig(sys.argv[2] + '/dustspecies.png', dpi=fig.dpi)

MassPD = pd.DataFrame(columns=['mass','sigmamin','sigmaplus'])
MassPD['mass'],MassPD['sigmamin'],MassPD['sigmaplus'] = dust_parameters,dust_sigma_min,dust_sigma_plus
MassPD.to_csv(sys.argv[2]+'/mass_fractions.csv')
