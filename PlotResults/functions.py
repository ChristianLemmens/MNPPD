import pandas as pd
import sys
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy import interpolate
from scipy.stats import chisquare
from timeit import default_timer as timer
from sklearn import preprocessing

def get_1sigma(outputpath,param,shortlong):
    """
    star        : name of star (str)
    param       : name of parameter (str)
    shortlong   : 'short' or 'long'

    Retrieve median and 3 sigma errors of
    parameter of choice for a given star
    returns [median, low 1sigma, high 1sigma]
    """

    from PyMultiNest.pymultinest.analyse import Analyzer
    if shortlong == 'short':
        a = Analyzer(23, outputfiles_basename = outputpath + "/test-")
        params = ['C1','Trim_min','qrim','C2','Tmid_min','Tmid_max','qmid','Tatm_min','Tatm_max','qatm','Ol. 0.1','Ol. 2.0','Ol. 5.0','Py. 0.1','Py. 2.0','Py. 5.0','Fo. 0.1','Fo. 2.0','En. 0.1','En. 2.0','Si. 0.1','Si. 2.0','Si. 5.0']

    elif shortlong == 'long':
        a = Analyzer(20, outputfiles_basename = outputpath + "/test-")
        params = ['C2','Tmid_min','Tmid_max','qmid','Tatm_min','Tatm_max','qatm','Ol. 0.1','Ol. 2.0','Ol. 5.0','Py. 0.1','Py. 2.0','Py. 5.0','Fo. 0.1','Fo. 2.0','En. 0.1','En. 2.0','Si. 0.1','Si. 2.0','Si. 5.0']
    i = params.index(param)
    stats = a.get_stats()
    # print(stats['modes'][0]['maximum a posterior'][i])
    p = stats['modes'][0]['mean'][i]
    pl,ph = stats['marginals'][i]['1sigma']
    return p, p-pl, ph-p

def get_all1sigma(outputpath,n_params,region):
    if "parameters1sigma.csv" not in listdir(outputpath):

        if n_params == 23:
            shortlong = region
            params = ['C1','Trim_min','qrim','C2','Tmid_min','Tmid_max','qmid','Tatm_min','Tatm_max','qatm','Ol. 0.1','Ol. 2.0','Ol. 5.0','Py. 0.1','Py. 2.0','Py. 5.0','Fo. 0.1','Fo. 2.0','En. 0.1','En. 2.0','Si. 0.1','Si. 2.0','Si. 5.0']
        elif n_params == 20:
            shortlong = region
            params = ['C2','Tmid_min','Tmid_max','qmid','Tatm_min','Tatm_max','qatm','Ol. 0.1','Ol. 2.0','Ol. 5.0','Py. 0.1','Py. 2.0','Py. 5.0','Fo. 0.1','Fo. 2.0','En. 0.1','En. 2.0','Si. 0.1','Si. 2.0','Si. 5.0']
        else:
            shortlong = region
            params = n_params
        output = pd.DataFrame()
        for parameter in params:
            output = output.append([get_1sigma(star,parameter,shortlong)],ignore_index=True)
        output.columns=['Mean','SigmaMin','SigmaPlus']
        output.to_csv(outputpath +"/parameters1sigma.csv")
    else:
        output = pd.read_csv(outputpath + "/parameters1sigma.csv")
    return output

def dust_sigma(pandaFrame):
    """ pandaFrame['Mean'], pandaFrame['SigmaMin'], pandaFrame['SigmaPlus'] """
    kap = [0.1*3,2*3,5*3,0.1*2.8,2*2.8,5*2.8,0.1*3.2,2*3.2,0.1*2.8,2*2.8,0.1*2.2,2*2.2,5*2.2]
    dusties = np.multiply(kap,pandaFrame['Mean'])
    dust_parameters = dusties/dusties.sum()
    parameters_sigma_plus = np.multiply(pandaFrame['SigmaPlus'],kap)
    parameters_sigma_min = np.multiply(pandaFrame['SigmaMin'],kap)
    mass_sigma_plus = (dusties+parameters_sigma_plus)/(dusties.sum()+parameters_sigma_plus)-dust_parameters
    mass_sigma_min = dust_parameters-(dusties-parameters_sigma_min)/(dusties.sum()-parameters_sigma_min)
    return dust_parameters,mass_sigma_min, mass_sigma_plus

def dust_sigma_symmetric(pandaFrame):
    """ pandaFrame['Mean'], pandaFrame['Sigma'] """
    kap = [0.1*3,2*3,5*3,0.1*2.8,2*2.8,5*2.8,0.1*3.2,2*3.2,0.1*2.8,2*2.8,0.1*2.2,2*2.2,5*2.2]
    dusties = np.multiply(kap,pandaFrame['Mean'])
    dust_parameters = dusties/dusties.sum()
    parameters_sigma = np.multiply(pandaFrame['Sigma'],kap)
    mass_sigma_plus = parameters_sigma/dusties.sum()
    mass_sigma_min = dust_parameters-(dusties-parameters_sigma)/(dusties.sum()-parameters_sigma)
    return dust_parameters,mass_sigma_min,mass_sigma_plus
