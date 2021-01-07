import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.integrate import quad

class Spectral:
    """
    --Use: s = Spectral(star)--
    Creates an object containing
    s.spitzer : a IRS spectrum          from [.csv]
    s.stellar : stellar atmosphere flux from [.dat]

    Both attributes are manipulated by
    applying the methods described
    after the __init__() obstructor
    """
    def __init__(self,star):
        self.star = star
        spitzer_path = '/home/chris/Documents/MNPPD/IRS/' + self.star + '.csv'
        starfit_path = '/home/chris/Documents/MNPPD/StellarAtmosphere/' + self.star + '.dat'

        self.starfit = pd.read_csv(starfit_path,sep = '\t')
        self.spitzer = pd.read_csv(spitzer_path,usecols=['wavelength','flux','sigma'])
        self.short_spitzer = self.spitzer[(self.spitzer['wavelength']<17)]
        self.stellar = pd.Series()
        self.opacities = pd.DataFrame(columns=['Ol. 0.1','Ol. 2.0','Ol. 5.0','Py. 0.1','Py. 2.0','Py. 5.0','Fo. 0.1','Fo. 2.0','En. 0.1','En. 2.0','Si. 0.1','Si. 2.0','Si. 5.0'])

    # ===== Methods =====
    def cut_spectrum(self,begin=0,end=1e8):
        """
        removes all data lower than
        [begin] and/or higher than [end]
        """
        self.spitzer = self.spitzer[(self.spitzer['wavelength']>begin) &
                                    (self.spitzer['wavelength']<end)]
        if self.spitzer['wavelength'].max() <= 17:
            self.short_spitzer = self.spitzer[(self.spitzer['wavelength']<17)]
        return

    def cutout_spectrum(self,begin,end):
        """
        Removes part of spectrum from
        [begin] to [end]
        """
        self.spitzer = pd.concat([self.spitzer[(self.spitzer['wavelength']<begin)],
                                    self.spitzer[(self.spitzer['wavelength']>end)]])
        self.spitzer['wavelength']=self.spitzer['wavelength'].reset_index(drop=True)

        return

    def setunity(self):
        """
        Rescales spitzer spectrum to
        a maximum of 1. Also rescales
        spitzer error and stellar atmosphere
        flux with same factor F_max
        """
        norm = self.spitzer['flux'].max()

        self.spitzer['sigma'] = self.spitzer['sigma'] / norm
        self.starfit['flux'] = self.starfit['flux'] / norm
        self.spitzer['flux'] = self.spitzer['flux'] / norm
        return

    def restore_index(self):
        self.spitzer = self.spitzer.reset_index(drop=True)
        return

    def rebin_starfit(self):
        """
        Rebins stellar atmosphere model to same
        wavelengths as IRS spectrum
        """
        starfit_func = interpolate.interp1d(self.starfit['wav'],self.starfit['flux'])
        self.stellar = starfit_func(self.spitzer['wavelength'])

        return

    def read_opacity(self):
        """
        Reads in opacity curve from file
        and rebins it to spitzer wavelengths
        """
        kappa_folder='/home/chris/Documents/MNPPD/opacities/QVAL/' # list of kappa files.
        opacities = np.loadtxt('/home/chris/Documents/MNPPD/opacities/opacity_files.inp',skiprows=2,max_rows=13,dtype= 'str')[:,1]
        for i,col in enumerate(self.opacities.columns):
            print(i,col)
            kappa = pd.read_csv(kappa_folder + opacities[i],sep='\s+')
            f = interpolate.interp1d(kappa.iloc[:,0],kappa.iloc[:,1])
            self.opacities[col] = f(self.spitzer['wavelength'])
        # print(self.spitzer,type(self.stellar))
        return

    def __repr__(self):

        return"Norm({max})".format(max = 0)


class Function:
    """
    Parent class of Model:
    Functions needed in the model
    components are defined here.
    """
    def __init__(self,wavelengths):
        """load attributes"""
        self.h = 6.626e-34       # Planck Constant
        self.c = 3.0e+8          # Speed of light. unit = m/s
        self.k = 1.38e-23        # Boltzmann constant

        self.wavelengths = wavelengths

    def planck(self,wav,T):
        """
        Planck function of temperature [T]
        binned to wavelengths [wav]
        """

        wav = wav * 1e-6                       #convert micron to meter
        a = 2.0*self.h*self.c**2
        b = self.h*self.c/(wav*self.k*T)
        intensity = a/ ( (wav**5) * (np.exp(b) - 1.0) )     #Planck in wavelength units
        intensity = -intensity/self.c*wav**2                     #convert to frequency units (* lamba^2/c)
        return intensity

    def integrand(self,T,q,wav):
        """
        Definition of integrand describing
        continuum emission at a certain temperature
        For formula see: Juhasz et al. (2010)
        """
        return 2*np.pi*self.planck(wav,T)*T**((2-q)/q)    # function inside midplane integrand

    def midplane(self,q,wav,Tmax,Tmin):
        """
        Integrates [integrand] from [Tmax] to [Tmin]
        """
        return quad(self.integrand, Tmax, Tmin, args=(q,wav))[0]   # unit = Jy

    def dust(self,C_ij):
        """
        Sum of opacities and according dust
        scale factors (C_ij * k_ij)
        """
        Fdust = []                                          # where C_ij is abundance
        Ftot = np.zeros(len(self.wavelengths))
        C_ij = np.array(C_ij)
        for i,col in enumerate(self.opacities.columns):
            Fdust.append(C_ij[i] * self.opacities[col])
            Ftot = np.add(Fdust[i],Ftot)
        return Ftot


class Model(Function):
    """
    --Use: f = Model(wavelengths,opacieties)--

    Creates an object containing
    f.components : with columns ['disk_model','inner_rim','midplane','atmosphere']

    Model object is updated using:
    --     f.define_parameters(pars)        --
    --     f.compute()                      --
    """
    def __init__(self,wavelengths,opacities):
        Function.__init__(self,wavelengths)
        self.kappa_folder = '/home/christianl/Documents/Thesis/specfit_v1.2/QVAL/'
        self.components = pd.DataFrame(columns=['disk_model','inner_rim','midplane','atmosphere'])
        self.vec_midplane = np.vectorize(self.midplane)
        self.parameters_values = []
        self.opacities = opacities

    def define_parameters(self,paramaters_values):
        """Update model parameters"""
        self.parameters_values = paramaters_values

    def normalize_vec(self,q,Tmax,Tmin):              # Normalize vector
        """
        Scales integral to maximum of 1
        such that scale factors are
        defined between 0 and 1.
        """
        integral = self.vec_midplane(q,self.wavelengths,Tmax,Tmin)
        return integral/integral.max()

    def normalize_vec_rim(self,q,Tmax,Tmin,short_wavelengths):
        """
        Scales inner rim integral of
        17-35 micron regime such that
        inner rim integral has maximum
        1 at 5-17 micron regime.
        """
        integral = self.vec_midplane(q,self.wavelengths,Tmax,Tmin)
        short_integral = self.vec_midplane(q,short_wavelengths,Tmax,Tmin)
        return integral/short_integral.max()

    def compute(self):
        """
        Computes new model components
        for new set of parameters
        """
        C1,Trim_min,qrim,C2,Tmid_min,Tmid_max,qmid,Tatm_min,Tatm_max,qatm = self.parameters_values[:-13]
        D_ij = self.parameters_values[-13:]
        self.components['inner_rim'] = C1*self.normalize_vec(qrim,1500,Trim_min)
        self.components['midplane'] = C2*self.normalize_vec(qmid,Tmid_max,Tmid_min)
        B_atm = self.normalize_vec(qatm,Tatm_max,Tatm_min)
        self.components['atmosphere'] = np.multiply(B_atm,self.dust(D_ij))
        self.components['disk_model'] = self.components['inner_rim'] + self.components['midplane'] + self.components['atmosphere']
        return

    def longwave_rim(self,factor,short_wavelengths):
        """
        Rescales inner rim at long
        wavelengths to compensate for
        different maximum values at
        5-17 and 17-35 micron spectrum
        """
        C1,Trim_min,qrim = self.parameters_values[0:3]
        self.components['inner_rim'] = C1*self.normalize_vec_rim(qrim,1500,Trim_min,short_wavelengths)
        self.components['inner_rim'] *= factor
        # self.components['disk_model'] = self.components['inner_rim'] + self.components['midplane'] + self.components['atmosphere']
        return

    def compute_longwave(self):
        """
        Computes new model components
        for new set of parameters
        17-35 micron
        """
        C1,Trim_min,qrim,C2,Tmid_min,Tmid_max,qmid,Tatm_min,Tatm_max,qatm = self.parameters_values[:-13]
        D_ij = self.parameters_values[-13:]
        # self.components['inner_rim'] = C1*self.normalize_vec(qrim,1500,Trim_min)
        self.components['midplane'] = C2*self.normalize_vec(qmid,Tmid_max,Tmid_min)
        B_atm = self.normalize_vec(qatm,Tatm_max,Tatm_min)
        self.components['atmosphere'] = np.multiply(B_atm,self.dust(D_ij))
        self.components['disk_model'] = self.components['inner_rim'] + self.components['midplane'] + self.components['atmosphere']
        return
