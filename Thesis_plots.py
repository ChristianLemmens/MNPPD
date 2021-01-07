from os import listdir
import os
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functions as fn
import sys
path = '/home/chris/Documents/Thesis/jupbooks/daniele/JulyRun/short/'
path_long = '/home/chris/Documents/Thesis/jupbooks/daniele/JulyRun/long/'
# path = path_long
stars = listdir(path)
stars = sorted(stars)
stars.remove('HD135344B')
stars.remove('HD144432')
stars.remove('DG-Tau')
stars.remove('SR21')
# print(stars)

def grainsize_Fo_En2():
    """
    Plot the ratio of grain sizes 2.0 : 0.1 micron of
    forsterite (y-axis) against enstatite (x-axis)

    THESIS: Figure 5.1.3 (a)
    """
    l_once,s_once=0,0
    for star in stars:
        marker = 'o'
        xuplims = False
        uplims = False
        try:
            MN_output_short = pd.read_csv(path + star + "/full/parameters1sigma.csv",skiprows=10,index_col=False)
            MN_output_short.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            dust_parameters,parameters_sigma_min,parameters_sigma_plus = fn.dust_sigma(MN_output_short)
            crystal_params = dust_parameters[6:10]
            enstatite_size = crystal_params[9] / crystal_params[8]
            forsterite_size = crystal_params[7] / crystal_params[6]
            enstatite_plus = (parameters_sigma_plus[8]/dust_parameters[8] + parameters_sigma_plus[9]/dust_parameters[9]) *enstatite_size
            enstatite_min = (parameters_sigma_min[8]/dust_parameters[8] + parameters_sigma_min[9]/dust_parameters[9]) *enstatite_size
            forsterite_plus = (parameters_sigma_plus[6]/dust_parameters[6] + parameters_sigma_plus[7]/dust_parameters[7]) *forsterite_size
            forsterite_min = (parameters_sigma_min[6]/dust_parameters[6] + parameters_sigma_min[7]/dust_parameters[7])*forsterite_size

            plot =plt.errorbar(enstatite_size,forsterite_size,xerr=[[enstatite_min],[enstatite_plus]],yerr=[[forsterite_min],[forsterite_plus]],marker='o',markersize=5,color='blue',elinewidth=0.3)
            if s_once == 0:
                plot.set_label('5-17 $\mu m$')
                s_once+=1

            print(star,forsterite_size)
        except Exception as e:
            print(star , e)
            print('plakje')
        marker = 'o'
        xuplims = False
        uplims = False
        try:
            MN_output_short = pd.read_csv(path_long + star + "/full/parameters1sigma.csv",skiprows=7,index_col=False)
            MN_output_short.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            dust_parameters,parameters_sigma_min,parameters_sigma_plus = fn.dust_sigma(MN_output_short)
            crystal_params = dust_parameters[6:10]
            enstatite_size = crystal_params[9] / crystal_params[8]
            forsterite_size = crystal_params[7] / crystal_params[6]
            enstatite_plus = (parameters_sigma_plus[8]/dust_parameters[8] + parameters_sigma_plus[9]/dust_parameters[9]) *enstatite_size
            enstatite_min = (parameters_sigma_min[8]/dust_parameters[8] + parameters_sigma_min[9]/dust_parameters[9]) *enstatite_size
            forsterite_plus = (parameters_sigma_plus[6]/dust_parameters[6] + parameters_sigma_plus[7]/dust_parameters[7]) *forsterite_size
            forsterite_min = (parameters_sigma_min[6]/dust_parameters[6] + parameters_sigma_min[7]/dust_parameters[7])*forsterite_size
            # if enstatite_size - enstatite_min < 1e-2:
            #     marker = '|'
            #     xuplims = True
            #     enstatite_size += enstatite_plus
            plot =plt.errorbar(enstatite_size,forsterite_size,xuplims=xuplims,uplims=uplims,marker=marker,markersize=5,xerr=[[enstatite_min],[enstatite_plus]],yerr=[[forsterite_min],[forsterite_plus]],color='red',elinewidth=0.3)

            if l_once == 0:
                plot.set_label('17-35 $\mu m$')
                l_once+=1
        except Exception as e:
            print(star , e)
            print('plakje')
    # plt.xlabel('F$_{24}$/F$_8$')
    plt.plot(plt.xlim(),plt.xlim(),label='d$_{fo}$ = d$_{en}$')
    plt.plot(plt.xlim(),[1,1],linestyle='dashdot',color = 'black',label = 'M$_{0.1 \mu m}$ = M$_{2.0 \mu m}$' )
    plt.plot([1,1],plt.ylim(),linestyle='dashdot',color = 'black')
    plt.xlabel('Enstatite',size=15)
    plt.ylabel('Forsterite',size=15)
    plt.xscale('log')
    plt.yscale('log')
    plt.tick_params(labelsize=15)
    plt.subplots_adjust(bottom=0.15,left = 0.15)
    # plt.xlim(0.1,2)
    # plt.ylim(0.1,2)
    plt.legend(loc=3)
    plt.title('M$_{2.0 \mu m}$ / M$_{0.1 \mu m}$',size=20)

    plt.show()

def grainsize_Ol_Py():
    """
    Plot the ratio of grain sizes 5.0 : 0.1 micron of
    Olivine (y-axis) against Pyroxene (x-axis)

    NOT IN THESIS
    """
    l_once,s_once=0,0
    for star in stars:
        marker = 'o'
        xuplims = False
        uplims = False
        try:
            MN_output_short = pd.read_csv(path + star + "/full/parameters1sigma.csv",skiprows=10,index_col=False)
            MN_output_short.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            dust_parameters,parameters_sigma_min,parameters_sigma_plus = fn.dust_sigma(MN_output_short)
            enstatite_size = dust_parameters[5] / dust_parameters[3]
            forsterite_size = dust_parameters[2] / dust_parameters[0]
            enstatite_plus = (parameters_sigma_plus[3]/dust_parameters[3] + parameters_sigma_plus[5]/dust_parameters[5]) *enstatite_size
            enstatite_min = (parameters_sigma_min[3]/dust_parameters[3] + parameters_sigma_min[5]/dust_parameters[5]) *enstatite_size
            forsterite_plus = (parameters_sigma_plus[0]/dust_parameters[0] + parameters_sigma_plus[2]/dust_parameters[2]) *forsterite_size
            forsterite_min = (parameters_sigma_min[0]/dust_parameters[0] + parameters_sigma_min[2]/dust_parameters[2])*forsterite_size

            plot = plt.errorbar(enstatite_size,forsterite_size,xerr=[[enstatite_min],[enstatite_plus]],yerr=[[forsterite_min],[forsterite_plus]],marker='o',markersize=5,color='blue',elinewidth=0.3)
            if s_once == 0:
                plot.set_label('5-17 $\mu m$')
                s_once+=1

            print(star,forsterite_size)
        except Exception as e:
            print(star , e)
            print('plakje')
        marker = 'o'
        xuplims = False
        uplims = False
        try:
            MN_output_short = pd.read_csv(path_long + star + "/full/parameters1sigma.csv",skiprows=7,index_col=False)
            MN_output_short.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            dust_parameters,parameters_sigma_min,parameters_sigma_plus = fn.dust_sigma(MN_output_short)
            enstatite_size = dust_parameters[5] / dust_parameters[3]
            forsterite_size = dust_parameters[2] / dust_parameters[0]
            enstatite_plus = (parameters_sigma_plus[3]/dust_parameters[3] + parameters_sigma_plus[5]/dust_parameters[5]) *enstatite_size
            enstatite_min = (parameters_sigma_min[3]/dust_parameters[3] + parameters_sigma_min[5]/dust_parameters[5]) *enstatite_size
            forsterite_plus = (parameters_sigma_plus[0]/dust_parameters[0] + parameters_sigma_plus[2]/dust_parameters[2]) *forsterite_size
            forsterite_min = (parameters_sigma_min[0]/dust_parameters[0] + parameters_sigma_min[2]/dust_parameters[2])*forsterite_size
# if enstatite_size - enstatite_min < 1e-2:
            #     marker = '|'
            #     xuplims = True
            #     enstatite_size += enstatite_plus
            plot =plt.errorbar(enstatite_size,forsterite_size,xuplims=xuplims,uplims=uplims,marker=marker,markersize=5,xerr=[[enstatite_min],[enstatite_plus]],yerr=[[forsterite_min],[forsterite_plus]],color='red',elinewidth=0.3)

            if l_once == 0:
                plot.set_label('17-35 $\mu m$')
                l_once+=1
        except Exception as e:
            print(star , e)
            print('plakje')
    # plt.xlabel('F$_{24}$/F$_8$')
    plt.plot(plt.xlim(),plt.xlim(),label='M$_{5-17}$ = M$_{17-35}$')
    plt.plot(plt.xlim(),[1,1],linestyle='dashdot',color = 'black',label = 'M$_{0.1 \mu m}$ = M$_{2.0 \mu m}$' )
    plt.plot([1,1],plt.ylim(),linestyle='dashdot',color = 'black')
    plt.xlabel('Pyroxene',size=15)
    plt.ylabel('Olivine',size=15)
    plt.xscale('log')
    plt.yscale('log')
    plt.tick_params(labelsize=15)
    plt.subplots_adjust(bottom=0.15,left = 0.15)
    # plt.xlim(0.1,2)
    # plt.ylim(0.1,2)
    plt.legend(loc=3)
    plt.title('M$_{5.0 \mu m}$ / M$_{0.1 \mu m}$',size=20)

    plt.show()

def grainsize_in_out():
    """
    Plots a sum of mass fractions with
    y = long wavelength value
    x = short wavelength value

    Currently set to sum of 5 micron amorphous grains

    THESIS: Figure 5.3.1 a-d
    """
    l_once,s_once=0,0
    sizelist2 = [2,5,12] # 5micron grains
    # sizelist2 = [6,7,8,9] # crystalline grains
    # sizelist2 = [6,7] # forsterite grains
    sizelist2 = [10,11,12] # SiO2 grains
    for star in stars:
        marker, xuplims, yuplims = 'o',False,False
        try:
            MN_output_shortlim = pd.read_csv(path + star + "/full/parameters3sigma.csv",skiprows=10,index_col=False)
            MN_output_shortlim.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            MN_output_short = pd.read_csv(path + star + "/full/parameters1sigma.csv",skiprows=10,index_col=False)
            MN_output_short.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            dust_parameters,parameters_sigma_min,parameters_sigma_plus = fn.dust_sigma(MN_output_short)
            kaas,parameterslim_min,parameterslim_plus = fn.dust_sigma(MN_output_shortlim)

            params_short = dust_parameters[sizelist2].sum()
            params_short_min = parameters_sigma_min[sizelist2].sum()
            params_short_plus = parameters_sigma_plus[sizelist2].sum()
            params_shortlim_min = parameterslim_min[sizelist2].sum()
            params_shortlim_plus = parameterslim_plus[sizelist2].sum()

            MN_output_longlim = pd.read_csv(path_long + star + "/full/parameters3sigma.csv",skiprows=7,index_col=False)
            MN_output_longlim.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            MN_output_long = pd.read_csv(path_long + star + "/full/parameters1sigma.csv",skiprows=7,index_col=False)
            MN_output_long.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            dust_parameters,parameters_sigma_min,parameters_sigma_plus = fn.dust_sigma(MN_output_long)
            kaas,parameterslim_min,parameterslim_plus = fn.dust_sigma(MN_output_longlim)

            params_long = dust_parameters[sizelist2].sum()
            params_long_min = parameters_sigma_min[sizelist2].sum()
            params_long_plus = parameters_sigma_plus[sizelist2].sum()
            params_longlim_min = parameterslim_min[sizelist2].sum()
            params_longlim_plus = parameterslim_plus[sizelist2].sum()
            if params_short - params_shortlim_min < 0.005:
                xuplims = True
                params_short += params_shortlim_plus
                marker='|'
            if params_long - params_longlim_min < 0.005:
                yuplims = True
                params_long += params_longlim_plus
                marker='_'
                if xuplims == True:
                    marker = '+'
            if star in ['TW-Hya','PDS70','LkCa15']:
                marker = 'D'
            if params_long > params_short:
                print(star + ": Might be transitional?")
            plot = plt.errorbar(params_short,params_long,xuplims=xuplims,uplims=yuplims, xerr = [[params_short_min],[params_short_plus]],yerr = [[params_long_min],[params_long_plus]],marker=marker,markersize=5,elinewidth=0.3,color='black')
        except Exception as e:
            print(star , e)

    plt.plot([0.01,1],[0.01,1],label='M$_{5-17}$ = M$_{17-35}$')
    plt.xlabel('5-17 $\mu m$',size=15)
    plt.ylabel('17-35 $\mu m$',size=15)
    plt.xscale('log')
    plt.yscale('log')
    plt.tick_params(labelsize=15)
    plt.subplots_adjust(bottom=0.15,left = 0.15)
    plt.legend()
    plt.title('Mass fraction 5 $\mu$m grains',size=15)

    plt.show()
    return


def grainsize_flaring():
    """
    Plots grainsizes vs. flaring parameter of IRS
    y = IRS flaring parameter
    x = crystalline mass fraction

    THESIS: Figure 5.1.3 (b)
    """
    l_once,s_once=0,0
    sizelist = [6,7]
    sizelist2 = [6,7,8,9]
    for star in stars:
        spitzer_path = '/home/chris/Documents/Thesis/Diana/CSV/' + star + '.csv'
        spitzer = pd.read_csv(spitzer_path,usecols=['wavelength','flux','sigma'])
        F13 = spitzer[(spitzer['wavelength'] > 12.5) & (spitzer['wavelength'] < 13.5)]
        F30 = spitzer[(spitzer['wavelength'] > 30) & (spitzer['wavelength'] < 31)]
        flaring = np.mean(F30['flux'])/np.mean(F13['flux'])

        try:
            MN_output_shortlim = pd.read_csv(path + star + "/full/parameters3sigma.csv",skiprows=10,index_col=False)
            MN_output_shortlim.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            MN_output_short = pd.read_csv(path + star + "/full/parameters1sigma.csv",skiprows=10,index_col=False)
            MN_output_short.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            dust_parameters,parameters_sigma_min,parameters_sigma_plus = fn.dust_sigma(MN_output_short)
            kaas,parameterslim_min,parameterslim_plus = fn.dust_sigma(MN_output_shortlim)

            params_short = dust_parameters[sizelist2].sum()
            params_short_min = parameters_sigma_min[sizelist2].sum()
            params_short_plus = parameters_sigma_plus[sizelist2].sum()
            params_shortlim_min = parameterslim_min[sizelist2].sum()
            params_shortlim_plus = parameterslim_plus[sizelist2].sum()
            # plt.errorbar(params_short,flaring,xerr=[[forsterite2_short_min],[forsterite2_short_plus]],marker='o',color='blue',elinewidth=0.3)
            print(star, flaring,params_short)
            if params_short > 0.25:
                print(star + " crystallinity larger than 0.25")
            print(params_short - params_shortlim_min)
            if params_short - params_shortlim_min<0.01:
                plot = plt.errorbar(params_short+params_short_plus,flaring,xerr=[[params_short_min],[params_short_plus]],marker="|",color='blue',elinewidth=0.3,xuplims=True)
            else:
                plot = plt.errorbar(params_short,flaring,xerr=[[params_short_min],[params_short_plus]],marker='o',markersize=5,color='blue',elinewidth=0.3)
            if s_once == 0:
                plot.set_label('5-17 $\mu m$')
                s_once+=1
        except Exception as e:
            print(star , e)
            print('Watch out!! no shortwave data!!!')
        try:
            # MN_output_short = pd.read_csv(path_long + star + "/full/parameters3sigma.csv",skiprows=7,index_col=False)
            # MN_output_short.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            # dust_parameters,parameters_sigma_min,parameters_sigma_plus = fn.dust_sigma(MN_output_short)
            # forsterite_long = dust_parameters[sizelist].sum()
            # forsterite_long_min = parameters_sigma_min[sizelist].sum()
            # forsterite_long_plus = parameters_sigma_plus[sizelist].sum()
            # forsterite2_long = dust_parameters[sizelist2].sum()
            # forsterite2_long_min = parameters_sigma_min[sizelist2].sum()
            # forsterite2_long_plus = parameters_sigma_plus[sizelist2].sum()
            # # if
            # plt.errorbar(forsterite2_long,flaring,xerr=[[forsterite2_long_min],[forsterite2_long_plus]],marker='o',color='red',elinewidth=0.3)
            MN_output_shortlim = pd.read_csv(path_long + star + "/full/parameters3sigma.csv",skiprows=7,index_col=False)
            MN_output_shortlim.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            MN_output_short = pd.read_csv(path_long + star + "/full/parameters1sigma.csv",skiprows=7,index_col=False)
            MN_output_short.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            dust_parameters,parameters_sigma_min,parameters_sigma_plus = fn.dust_sigma(MN_output_short)
            kaas,parameterslim_min,parameterslim_plus = fn.dust_sigma(MN_output_shortlim)

            params_short = dust_parameters[sizelist2].sum()
            params_short_min = parameters_sigma_min[sizelist2].sum()
            params_short_plus = parameters_sigma_plus[sizelist2].sum()
            params_shortlim_min = parameterslim_min[sizelist2].sum()
            params_shortlim_plus = parameterslim_plus[sizelist2].sum()
            # plt.errorbar(params_short,flaring,xerr=[[forsterite2_short_min],[forsterite2_short_plus]],marker='o',color='blue',elinewidth=0.3)
            print(star, flaring,params_short)
            if params_short > 0.25:
                print(star + " crystallinity larger than 0.25")
            print(params_short - params_shortlim_min)
            if params_short - params_shortlim_min<0.01:
                plot = plt.errorbar(params_short+params_short_plus,flaring,xerr=[[params_short_min],[params_short_plus]],marker="|",color='red',elinewidth=0.3,xuplims=True)
            else:
                plot = plt.errorbar(params_short,flaring,xerr=[[params_short_min],[params_short_plus]],marker='o',markersize=5,color='red',elinewidth=0.3)
            if l_once == 0:
                plot.set_label('17-35 $\mu m$')
                l_once+=1
        except Exception as e:
            print(star , e)
            print('Watch out!! no longwave data!!!')

    # plt.xlabel('F$_{24}$/F$_8$')
    # plt.plot([0.1,2],[0.1,2])
    # plt.plot([0.01,1],[0.01,1])
    plt.ylabel('F$_{30}$/F$_{13}$',size=15)
    plt.xlabel('M$_{cryst}$ / M$_{tot}$',size=15)
    plt.xscale('log')
    plt.yscale('log')
    plt.tick_params(labelsize=15)
    plt.subplots_adjust(bottom=0.15,left = 0.15)
    plt.xlim(0.01,0.5)
    # plt.ylim(0.01,0.5)
    plt.legend()
    # plt.title('Mass fraction 5.0 $\mu m$ grains',size=15)

    plt.show()
    return

# grainsize_Fo_En2()
# grainsize_Ol_Py()
# in_out_Forsterite()
# grainsize_in_out()
# grainsize_flaring()
