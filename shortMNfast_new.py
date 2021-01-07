"""
--------------------------------------------------------
Python Script that activates MultiNest Fitting Procedure
--------------------------------------------------------
- Adjusted to wavelengths <17 micron
- Runs for precise results and uses priors obtained from
    short wavelength runs and long wavelength fast run
--------------------------------------------------------
RUNNING:
$ python3 shortMNfast.py [starname] [precision]

[starname]  = name of protoplanetary system to be fitted
[precision] = fast (quick results) | full (precise results)

"""

import os, glob, json, time, math, sys
import os
import numpy as np
import pymultinest
import scipy.stats
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import interpolate
from scipy.stats import chisquare
import pandas as pd
from timeit import default_timer as timer
from model import Spectral,Model


"""
Definition of paramater space
Parameter space can be defined
in .csv file with columns:
['name', 'value', 'min', 'max', 'comp', 'var']
"""
try:
    csv_file = str('Output/short/' + sys.argv[1] + '/'+ sys.argv[2] +'/priors.csv')
    paravalues = pd.read_csv(csv_file)

    C1_values       = paravalues[(paravalues['name']=='C1')].values[0].tolist()
    Trimmin_values  = paravalues[(paravalues['name']=='Trimmin')].values[0].tolist()
    qrim_values     = paravalues[(paravalues['name']=='qrim')].values[0].tolist()
    C2_values       = paravalues[(paravalues['name']=='C2')].values[0].tolist()
    Tmidmin_values  = paravalues[(paravalues['name']=='Tmidmin')].values[0].tolist()
    Tmidmax_values  = paravalues[(paravalues['name']=='Tmidmax')].values[0].tolist()
    qmid_values     = paravalues[(paravalues['name']=='qmid')].values[0].tolist()
    Tatmmin_values  = paravalues[(paravalues['name']=='Tatmmin')].values[0].tolist()
    Tatmmax_values  = paravalues[(paravalues['name']=='Tatmmax')].values[0].tolist()
    qatm_values     = paravalues[(paravalues['name']=='qatm')].values[0].tolist()
    k0_values       = paravalues[(paravalues['name']=='Ol. 0.1')].values[0].tolist()
    k1_values       = paravalues[(paravalues['name']=='Ol. 2.0')].values[0].tolist()
    k2_values       = paravalues[(paravalues['name']=='Ol. 5.0')].values[0].tolist()
    k3_values       = paravalues[(paravalues['name']=='Py. 0.1')].values[0].tolist()
    k4_values       = paravalues[(paravalues['name']=='Py. 2.0')].values[0].tolist()
    k5_values       = paravalues[(paravalues['name']=='Py. 5.0')].values[0].tolist()
    k6_values       = paravalues[(paravalues['name']=='Fo. 0.1')].values[0].tolist()
    k7_values       = paravalues[(paravalues['name']=='Fo. 2.0')].values[0].tolist()
    k8_values       = paravalues[(paravalues['name']=='En. 0.1')].values[0].tolist()
    k9_values       = paravalues[(paravalues['name']=='En. 2.0')].values[0].tolist()
    k10_values       = paravalues[(paravalues['name']=='Si. 0.1')].values[0].tolist()
    k11_values       = paravalues[(paravalues['name']=='Si. 2.0')].values[0].tolist()
    k12_values       = paravalues[(paravalues['name']=='Si. 5.0')].values[0].tolist()
    model_name = 'PPD'

except Exception as e:
    print(e, 'No matching Priors.csv')
    quit()


listvar = [C1_values,Trimmin_values,qrim_values,C2_values,Tmidmin_values,
            Tmidmax_values,qmid_values,Tatmmin_values,Tatmmax_values,
            qatm_values,k0_values,k1_values,k2_values,k3_values,k4_values,
            k5_values,k6_values,k7_values,k8_values,k9_values,k10_values,
            k11_values,k12_values]

# pandavar = pd.DataFrame(listvar,columns = ['name','value', 'min', 'max', 'comp', 'var'])



from math import log10, isnan, isinf

def d_create_uniform_prior_for(model, par):
    """
    |PyMN| Use for location variables (position)
    The uniform prior gives equal weight in non-logarithmic scale.
    """
    pname, pval, pmin, pmax, pcom, pindex = par
    print('  uniform prior for %s between %f and %f ' % (pname, pmin, pmax))
    # TODO: should we use min/max or bottom/top?
    low = float(pmin)
    spread = float(pmax - pmin)
    def uniform_transform(x): return x * spread + low
    return dict(model=model, index=pindex, com=pcom, name=pname,
        transform=uniform_transform, aftertransform=lambda x: x)

def d_create_jeffreys_prior_for(model, par):
    """
    |PyMN| Use for scale variables (order of magnitude)
    The Jeffreys prior gives equal weight to each order of magnitude between the
    minimum and maximum value. Flat in logarithmic scale
    """
    # pval, pdelta, pmin, pbottom, ptop, pmax = par.values   ## call XSPEC
    # TODO: should we use min/max or bottom/top?
    #print '  ', par.values
    pname, pval, pmin, pmax, pcom, pindex = par
    print('  jeffreys prior for %s between %e and %e ' % (pname, pmin, pmax)) ## call XSPEC
    low = log10(pmin) #minimo
    spread = log10(pmax) - log10(pmin) #intervallo
    def log_transform(x): return x * spread + low
    def log_after_transform(x): return 10**x
    return dict(model=model, index=pindex, com=pcom, name=pname,
        transform=log_transform, aftertransform=log_after_transform)  ## call XSPEC


def create_prior_function(transformations):
    """
    |PyMN|
    Creates a single prior transformation function from parameter transformations
    """

    def prior(cube, ndim, nparams):
        try:
            for i, t in enumerate(transformations):
                transform = t['transform']
                cube[i] = transform(cube[i])
        except Exception as e:
            print('ERROR: Exception in prior function. Faulty transformations specified!')
            print('ERROR:', e)
            print(i, transformations[i])
            import sys
            sys.exit(-126)
    return prior


def set_parameters(transformations, values):
    """
    |PyMN|MNPPD| Set current parameters.
    """
    # transfomations is a dict with [model, index, name, transform, aftertransform]
    pars = []

    for i, t in enumerate(transformations):  # enumerate gives the values ==> t is the name of the transformation
        v = t['aftertransform'](values[i])
        assert not isnan(v) and not isinf(v), 'ERROR: parameter %d (index %d, %s) to be set to %f' % (
            i, t['index'], t['name'], v)

        pars.append(v)
    # Definition of model object from model.py with parameters [pars].

    f.define_parameters(pars)
    f.compute()
    model = f.components['disk_model'] + s.stellar

    return model


def set_best_fit(analyzer, transformations):
    """
    |PyMN| Set to best fit
    """
    try:
        modes = analyzer.get_mode_stats()['modes']
        highestmode = sorted(modes, key=lambda x: x['local evidence'])[0]
        params = highestmode['maximum a posterior']
    except Exception as e:
        # modes were not described by MultiNest, last point instead
        pass
    params = analyzer.get_best_fit()['parameters']
    set_parameters(transformations=transformations, values=params)


def chisqg(ydata,ymod,sd=None):
    """
    |MNPPD| Address Chi square of current model
    [ymod] and spectrum [ymod] with errors [sd]
    """
    try:
        chisq=np.sum( ((ydata-ymod)/sd)**2 )
        chisq=np.sum( ((s.spitzer['flux']-ymod)/s.spitzer['sigma'])**2 )
        return chisq
    except Exception as e:
        print('Exception in chisquare Function: ', e)
        return 2e300


def d_nested_run(transformations, prior_function = None, sampling_efficiency = 0.8,
        n_live_points = 50, evidence_tolerance = 5.,
    outputfiles_basename = 'chains/', verbose=True, **kwargs):
    """ |PyMN|
    Run the Bayesian analysis with specified parameters+transformations.

    If prior is None, uniform priors are used on the passed parameters.
    If parameters is also None, all thawed parameters are used.

    :param transformations: Parameter transformation definitions
    :param prior_function: set only if you want to specify a custom, non-separable prior
    :param outputfiles_basename: prefix for output filenames.
    The remainder are multinest arguments (see PyMultiNest and MultiNest documentation!)

    The remainder are multinest arguments (see PyMultiNest and MultiNest documentation!)
    n_live_points: 400 are often enough

    For quick results, use sampling_efficiency = 0.8, n_live_points = 50,
    evidence_tolerance = 5.
    The real results must be estimated with sampling_efficiency = 0.3,
    n_live_points = 400, evidence_tolerance = 0.5
    and without using const_efficiency_mode, otherwise it is not reliable.
    """

    # for convenience
    if outputfiles_basename.endswith('/'):
        if not os.path.exists(outputfiles_basename):
            os.mkdir(outputfiles_basename)

    # associate the prior transformations to the prior_functions. It calls the function 'create_prior_funciton()'
    # it is described below.
    if prior_function is None:
        prior_function = create_prior_function(transformations)


    def log_likelihood(cube,ndim,nparams):
        try:
            model = set_parameters(transformations=transformations, values = cube)

            chi2 = chisqg(s.spitzer['flux'],model,s.spitzer['sigma'])
            l = -0.5*float(chi2) # CSTAT value from the most recent fit (GET ONLY)
            return l
        except Exception as e:
            print('Exception in log-Likelihood Function: ', e)
            return -1e300


    # run multinest

    n_params = len(transformations)
    # PyMultiNest.RUN is reported below.
    pymultinest.run(log_likelihood, prior_function, n_params,
        sampling_efficiency = sampling_efficiency, n_live_points = n_live_points,
        outputfiles_basename = outputfiles_basename,
        verbose=verbose, **kwargs) # You can use [**kwargs] to let your functions take an arbitrary number
                                   # of keyword arguments ("kwargs" means "keyword arguments")


    paramnames = [str(t['name']) for t in transformations]
    print('**---- Computation time: ' + str((timer()-start)/3600) + ' hours -----**')

    # json.dump [JSON (JavaScript Object Notation)] write a file.json, JAVA compatible. Keys in key/value
    # pairs of JSON e always of the type str. When a dictionary is converted into JSON, all the keys of the
    # dictionary are coerced to strings. As a result of this, if a dictionary is converted into JSON and
    # then back into a dictionary, the dictionary may not equal the original one.
    json.dump(paramnames, open('%sparams.json' % outputfiles_basename, 'w'), indent=4)

    # store as chain too, and try to load it for error computations
    # pyMultinest.ANALYZER is reported below
    analyzer = pymultinest.Analyzer(n_params = len(transformations),
        outputfiles_basename = outputfiles_basename)
    posterior = analyzer.get_equal_weighted_posterior()

    # set current parameters to best fit
    set_best_fit(analyzer, transformations)

    return analyzer


start = timer()

"""Specify spitzer IRS and stellar atmosphere model from Spectral object"""
s = Spectral(sys.argv[1])
s.cut_spectrum(begin=5.24)
if sys.argv[1] =='DF-Tau':     #Removal of wavelengths with outlying fluxes
    s.cut_spectrum(begin=5.54)
s.cut_spectrum(end=17)
s.setunity()
s.rebin_starfit()
s.read_opacity()
s.restore_index()
"""Specify empty model object"""
f = Model(s.spitzer['wavelength'],s.opacities)

# define prior
transformations = [
    # jeffreys prior for nH (but see below)
    d_create_uniform_prior_for(model_name, C1_values),
    d_create_uniform_prior_for(model_name, Trimmin_values),
    d_create_uniform_prior_for(model_name, qrim_values),
    d_create_uniform_prior_for(model_name, C2_values),
    d_create_uniform_prior_for(model_name, Tmidmin_values),
    d_create_uniform_prior_for(model_name, Tmidmax_values),
    d_create_uniform_prior_for(model_name, qmid_values),
    d_create_uniform_prior_for(model_name, Tatmmin_values),
    d_create_uniform_prior_for(model_name, Tatmmax_values),
    d_create_uniform_prior_for(model_name, qatm_values),
    d_create_uniform_prior_for(model_name, k0_values),
    d_create_uniform_prior_for(model_name, k1_values),
    d_create_uniform_prior_for(model_name, k2_values),
    d_create_uniform_prior_for(model_name, k3_values),
    d_create_uniform_prior_for(model_name, k4_values),
    d_create_uniform_prior_for(model_name, k5_values),
    d_create_uniform_prior_for(model_name, k6_values),
    d_create_uniform_prior_for(model_name, k7_values),
    d_create_uniform_prior_for(model_name, k8_values),
    d_create_uniform_prior_for(model_name, k9_values),
    d_create_uniform_prior_for(model_name, k10_values),
    d_create_uniform_prior_for(model_name, k11_values),
    d_create_uniform_prior_for(model_name, k12_values)


]

# where to store intermediate and final results? this is the prefix used
output_folder = 'Output/short/' + sys.argv[1] + '/'
try:
    os.mkdir(output_folder)
    print("New Directory created")
except:
    print("Using folder: short/" + sys.argv[1])

try:
    os.mkdir(output_folder + sys.argv[2])
    output_folder = output_folder + sys.argv[2]     # fast or full folder
    print('New folder created: ' + output_folder)
except:
    output_folder = output_folder + sys.argv[2] +''    # fast or full folder
outputfiles_basename = output_folder + '/test-'
# pandavar.to_csv(output_folder + '/priors.csv',index=False)

# Use fast when analyzing a spectrum for the first time to check whether priors are decent
# Use full for accurate results
if sys.argv[2] == 'fast':
    efficiency = 0.8
    livepoints = 50
    tolerance = 5.
elif sys.argv[2] == 'full':
    efficiency = 0.3
    livepoints = 400
    tolerance = 0.5

# send it off!
d_nested_run(transformations = transformations,
    outputfiles_basename = outputfiles_basename,
    verbose=True, # show a bit of progress
    resume=True, # MultiNest supports resuming a crashed/aborted run
    sampling_efficiency = efficiency,
    n_live_points =livepoints,
    evidence_tolerance = tolerance
    )


# get the results
analyzer = pymultinest.Analyzer(n_params = len(transformations),
        outputfiles_basename = outputfiles_basename)


# make marginal plots
# bxa.plot.marginal_plots(analyzer)
a = analyzer.get_stats()
print('Model evidence: ln(Z) = %.2f +- %.2f' % (a['global evidence'], a['global evidence error']))
# store stats

json.dump(a, open(outputfiles_basename + 'stats.json', 'w'), indent=4)
