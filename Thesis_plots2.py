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
# print(stars)
# quit()
# stars = ['DR-Tau','FT-Tau','PDS70','RW-Aur']
print(stars)

def FitDustPlot():
    for star in stars:
        # if star == 'PDS70':
        if 'fullkas' in listdir(path + star):
            os.system("python3 FunctionsFitting2.py " + star + " "
                        + path
                        + star + "/full/test-stats.dat ")
        if 'full' in listdir(path_long + star):
            os.system("python3 FunctionsFitting2.py " + star + " "
                        + path_long
                        + star + "/full/test-stats.dat long")

def Marginals():
    for star in stars:
        try:
            if "test-corner.png" in listdir(path + star + '/full'):
                print("already present")
            else:
                os.system("python3 PyMultiNest/multinest_marginals_fancy.py /home/chris/Documents/Thesis/jupbooks/daniele/JulyRun/short/"
                + star + "/full/test-")
        except:
            print(star)
def SamplePlot():
    images = []

    im1 = Image.open(path + stars[0] + "/full/fitfig.png")
    im2 = Image.open(path + stars[0] + "/full/dustspecies.png")
    blank_im = Image.new('RGB', (im1.size[0]+im2.size[0],im1.size[1]),color='white')
    blank_im.paste(im1,(0,0),mask=im1.split()[3])
    blank_im.paste(im2,(im1.size[0],0),mask=im2.split()[3])
    im1 = blank_im #ConvertImage(blank_im)
    print(im1.getbands())
    # images.append(Image.open(path + stars[0] + "/fast/dustspecies.png"))
    for star in stars:
        if 'full' in listdir(path + star):
            spectrum = (Image.open(path + star + "/full/fitfig.png"))
            dust = (Image.open(path + star + "/full/dustspecies.png"))
            blank_im = Image.new('RGB', (spectrum.size[0]+dust.size[0],spectrum.size[1]),color='white')
            blank_im.paste(spectrum,(0,0),mask=spectrum.split()[3])
            blank_im.paste(dust,(spectrum.size[0],0),mask=dust.split()[3])
            images.append(blank_im)
        # images.append(ConvertImage(Image.open(path + star + "/fast/fitfig.png")))
        # images.append(ConvertImage(Image.open(path + star + "/fast/dustspecies.png")))
    im1.save(path[:-1] + "-results_full.pdf", "PDF" ,resolution=100.0, save_all=True, append_images=images[1:])

def TotalPlot():
    images = []
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/Ubuntu-B.ttf", 50)
    im1 = Image.open(path + stars[0] + "/full/fitfig.png")
    im2 = Image.open(path + stars[0] + "/full/dustspecies.png")
    im3 = Image.open(path_long + stars[0] + "/full/fitfig.png")
    im4 = Image.open(path_long + stars[0] + "/full/dustspecies.png")
    blank_im = Image.new('RGB', (im1.size[0]+im2.size[0],im1.size[1]*2),color='white')
    blank_im.paste(im1,(0,0),mask=im1.split()[3])
    blank_im.paste(im2,(im1.size[0],0),mask=im2.split()[3])
    blank_im.paste(im3,(0,im1.size[1]),mask=im3.split()[3])
    blank_im.paste(im4,(im1.size[0],im1.size[1]),mask=im4.split()[3])
    draw = ImageDraw.Draw(blank_im)
    draw.text((im1.size[0]/1.1, 0), stars[0], fill ="red", font = font, align ="left")
    im1 = blank_im #ConvertImage(blank_im)
    # print(im1.getbands())
    # images.append(Image.open(path + stars[0] + "/fast/dustspecies.png"))
    for star in stars:
        try:
            spectrum = (Image.open(path + star + "/full/fitfig.png"))
            dust = (Image.open(path + star + "/full/dustspecies.png"))
            im3 = Image.open(path_long + star + "/full/fitfig.png")
            im4 = Image.open(path_long + star + "/full/dustspecies.png")
            blank_im = Image.new('RGB', (spectrum.size[0]+dust.size[0],spectrum.size[1]*2),color='white')
            blank_im.paste(spectrum,(0,0),mask=spectrum.split()[3])
            blank_im.paste(dust,(spectrum.size[0],0),mask=dust.split()[3])
            blank_im.paste(im3,(0,spectrum.size[1]),mask=im3.split()[3])
            blank_im.paste(im4,(spectrum.size[0],spectrum.size[1]),mask=im4.split()[3])
            draw = ImageDraw.Draw(blank_im)
            draw.text((spectrum.size[0]/1.1, 0), star, fill ="red", font = font, align ="left")
            images.append(blank_im)

        except Exception as e:
            print(star, e)
        # images.append(ConvertImage(Image.open(path + star + "/fast/fitfig.png")))
        # images.append(ConvertImage(Image.open(path + star + "/fast/dustspecies.png")))
    im1.save(path[:-1] + "-totalresults_total.pdf", "PDF" ,resolution=100.0, save_all=True, append_images=images[1:])
# path = path_long
def MarginalsSample():
    images = []
    stars.remove('XX-Cha')
    im1 = Image.open(path + stars[0] + "/full/test-corner.png")
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/Ubuntu-B.ttf", 200)
    draw = ImageDraw.Draw(im1)
    draw.text((im1.size[0]/2, 0), stars[0], fill ="red", font = font, align ="center")
    blank_im = Image.new('RGB',im1.size,color='white')
    blank_im.paste(im1,(0,0),mask=im1.split()[3])
    im1 = blank_im #ConvertImage(blank_im)
    print(im1.getbands())
    # images.append(Image.open(path + stars[0] + "/fast/dustspecies.png"))
    for star in stars:
        print(star)
        try:
        # if 'test-corner.png' in listdir(path + star + '/full'):
            corner = (Image.open(path + star + "/full/test-corner.png"))
            font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/Ubuntu-B.ttf", 200)
            draw = ImageDraw.Draw(corner)
            draw.text((im1.size[0]/2, 0), star, fill ="red", font = font, align ="center")
            blank_im = Image.new('RGB',corner.size,color='white')
            blank_im.paste(corner,(0,0),mask=corner.split()[3])
            images.append(blank_im)
        except:
            print("No test-corner.png found for: " + star)
        # images.append(ConvertImage(Image.open(path + star + "/fast/fitfig.png")))
        # images.append(ConvertImage(Image.open(path + star + "/fast/dustspecies.png")))
    im1.save(path[:-1] + "-corners_full.pdf", "PDF" ,resolution=100.0, save_all=True, append_images=images[1:])

def ConvertImage(im):
    rgb = Image.new('RGB', im.size, (255, 255, 255))  # white background
    rgb.paste(im, mask=im.split()[3])               # paste using alpha channel as mask
    # rgb.save(PDF_FILE, 'PDF', resoultion=100.0)
    return rgb

def import_folders():
    for star in stars:
        try:
            os.system("rsync -larv christianl@ssh.sron.nl:/home/christianl/Documents/Thesis/jupbooks/daniele/JulyRun/long/"
                        + star + "/full " + path + star)
        except:
            print(star + ": no folder found")

def copy_files():
    stars =['AA-Tau','BP-Tau','CX-Tau','CY-Tau','DF-Tau','DG-Tau','DL-Tau','DM-Tau','DN-Tau','DR-Tau'
            ,'FT-Tau','GW-Lup','HK-Tau-B','IM-Lup','LkCa15','PDS70','RNO90','RW-Aur','SY-Cha','SZ50'
            ,'SZ98','TW-Hya','V1094Sco','VW-Cha','WX-Cha','XX-Cha']
    for star in stars:
        os.system("cp /home/chris/Documents/Thesis/DataMine/"+star+"/starfit.png /home/chris/Documents/Thesis/Articles/latex/photoplots/" + star +"_phot.png")

def Likelihood_Crystal():
    stars = ['AA-Tau','BP-Tau','CY-Tau','DF-Tau','DL-Tau','DM-Tau','DN-Tau','DR-Tau','FT-Tau','HK-Tau-B','LkCa15','RW-Aur','WX-Cha','XX-Cha']

    for star in stars:
        try:
            # MN_output = pd.read_csv(path + star + "/full/test-stats.dat",sep='\s+',skiprows=2,nrows=23,index_col=False,usecols=[1,2])
            # MN_output.columns = ['Mean','Sigma']
            # MN_output = fn.get_all1sigma(star,['Ol. 0.1','Ol. 2.0','Ol. 5.0','Py. 0.1','Py. 2.0','Py. 5.0','Fo. 0.1','Fo. 2.0','En. 0.1','En. 2.0','Si. 0.1','Si. 2.0','Si. 5.0'],'short')
            MN_output = pd.read_csv(path + star + "/full/parameters3sigma.csv",skiprows=10,index_col=False)
            MN_output.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            parameters_values, parameters_sigma_min, parameters_sigma_plus = fn.dust_sigma(MN_output)
            print(type(parameters_values),type(parameters_sigma_min),type(parameters_sigma_plus))
            statistic = pd.read_csv(path + star + "/full/test-stats.dat",sep='\s+',nrows=1)
            loglike = np.abs(float(statistic.columns[5]))
            loglike_err = np.abs(float(statistic.columns[7]))
            spitzer_path = '/home/chris/Documents/Thesis/Diana/CSV/' + star + '.csv'
            spitzer = pd.read_csv(spitzer_path,usecols=['wavelength','flux','sigma'])
            spitzer = spitzer[(spitzer['wavelength']<17)]
            spitzer['flux'] = spitzer['flux']/spitzer['flux'].max()
            # spitzer['sigma'] = spitzer['sigma']/spitzer['flux'].max()
            # mean_error = spitzer['sigma']/spitzer['flux']
            model_data = pd.read_csv('/home/chris/Documents/Thesis/jupbooks/daniele/JulyRun/Model_spectra/' + star + '.csv',usecols=['wav','flux'])
            chi2 = chisqg(spitzer['flux'],model_data['flux'],spitzer['sigma'])
            # kap = [0.1*3,2*3,5*3,0.1*2.8,2*2.8,5*2.8,0.1*3.2,2*3.2,0.1*2.8,2*2.8,0.1*2.2,2*2.2,5*2.2]
            # dusties = np.multiply(kap,parameters_values)
            # dust_parameters = dusties/dusties.sum()
            # dust_sigma_min = parameters_sigma_min/dusties.sum()
            # dust_sigma_min = np.multiply(kap,dust_sigma_min)/dusties.sum()
            # dust_sigma_plus = parameters_sigma_plus/dusties.sum()
            # dust_sigma_plus = np.multiply(kap,dust_sigma_plus)/dusties.sum()
            # print('2')
            Crystal_frac = parameters_values[6:10].sum()/(parameters_values.sum())
            CF_sigma_min = parameters_sigma_min[6:10].sum()/(parameters_values.sum())
            CF_sigma_plus = parameters_sigma_plus[6:10].sum()/(parameters_values.sum())
            print('3')
            # crystal_sigma = crystal_frac*(1+dust_sigma[6:10])
            print(Crystal_frac,CF_sigma_min,CF_sigma_plus, star)
            plt.errorbar(Crystal_frac,chi2,xerr=[[CF_sigma_min],[CF_sigma_plus]],marker='o',color = 'black')
        except Exception as e:
            print(e)
    # plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$M_{cryst}$/$M_{tot}$')
    plt.ylabel('L')
    plt.title('Loglikelihood')      # $\chi ^2$
    plt.show()
"""
Correlation of 9.8 micron band strength with crystallinity
"""
def continuum(data):                        ## Make linear continuum under 9.8 peak
    micron_72 = data[(data['wavelength'] > 7.5) & (data['wavelength'] < 8)] #CSV : 7-7.5 BD: 7.5-8
    micron_135 = data[(data['wavelength'] > 13) & (data['wavelength'] < 13.5)]
    p1 = np.mean(micron_72)
    p2 = np.mean(micron_135)
    a = (p1['flux']-p2['flux'])/(p1['wavelength']-p2['wavelength'])
    b = p1['flux']-a*p1['wavelength']
    return a*data['wavelength']+b

def calc_amorph(data):                  ## Calculate F9.8/F11.3
    norm_data = data['flux']/continuum(data)
    micron_98 = np.mean(norm_data[(data['wavelength'] > 9.75) & (data['wavelength'] < 9.85)])
    micron_113 = np.mean(norm_data[(data['wavelength'] > 11.25) & (data['wavelength'] < 11.35)])
    return micron_98/micron_113, micron_98

def Crystal(star):
    try:
        MN_output = pd.read_csv(path + star + "/full/parameters3sigma.csv",skiprows=10,index_col=False)
        MN_output.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
    except:
        MN_output = fn.get_all1sigma(star,23,'short')[10:]
    parameters_values, parameters_sigma_min, parameters_sigma_plus = fn.dust_sigma(MN_output)
    Crystal_frac = parameters_values[6:10].sum()/(parameters_values.sum())
    CF_sigma_min = parameters_sigma_min[6:10].sum()/(parameters_values.sum())
    CF_sigma_plus = parameters_sigma_plus[6:10].sum()/(parameters_values.sum())
    return Crystal_frac,CF_sigma_min,CF_sigma_plus

def featureCrystal():
    for star in stars:
        if 'full' in listdir(path + star):
            spitzer_path = '/home/chris/Documents/Thesis/Diana/CSV/' + star + '.csv'
            spitzer = pd.read_csv(spitzer_path,usecols=['wavelength','flux','sigma'])
            spitzer = spitzer[(spitzer['wavelength']<17)]
            spitzer['flux'] = spitzer['flux']/spitzer['flux'].max()
            Crystal_frac,CF_sigma_min,CF_sigma_plus = Crystal(star)
            plt.errorbar(Crystal_frac,calc_amorph(spitzer)[1],xerr=[[CF_sigma_min],[CF_sigma_plus]],marker='o')

    plt.show()
    starsdiana = ['AA-Tau','BP-Tau','CY-Tau','DF-Tau','DL-Tau','DM-Tau','DN-Tau','DR-Tau','FT-Tau','HK-Tau-B','LkCa15','RW-Aur','WX-Cha','XX-Cha']

    for star in starsdiana:
        if 'full' in listdir(path + star):
            spitzer_path = '/home/chris/Documents/Thesis/Diana/CSV/' + star + '.csv'
            spitzer = pd.read_csv(spitzer_path,usecols=['wavelength','flux','sigma'])
            spitzer = spitzer[(spitzer['wavelength']<17)]
            spitzer['flux'] = spitzer['flux']/spitzer['flux'].max()
            Crystal_frac,CF_sigma_min,CF_sigma_plus = Crystal(star)
            plt.errorbar(Crystal_frac,calc_amorph(spitzer)[1],xerr=[[CF_sigma_min],[CF_sigma_plus]],marker='o')
    # plt.xscale('log')
    plt.yscale('log')
    plt.show()

def ForEnstatite():
    cumulative = pd.DataFrame(columns=['Fo','En','w'])
    starsdiana = ['AA-Tau','BP-Tau','CY-Tau','DF-Tau','DL-Tau','DM-Tau','DN-Tau','DR-Tau','FT-Tau','HK-Tau-B','LkCa15','RW-Aur','WX-Cha','XX-Cha']
    l_once,s_once = 0,0
    for star in stars:

        try:
            xuplims, uplims = False,False
            MN_output = pd.read_csv(path + star + "/full/parameters1sigma.csv",skiprows=10,index_col=False)
            MN_output.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            MN_outputlim = pd.read_csv(path_long + star + "/full/parameters3sigma.csv",skiprows=7,index_col=False)
            MN_outputlim.columns = ['kaas','Mean','SigmaMin','SigmaPlus']

            parameters_values, parameters_sigma_min, parameters_sigma_plus = fn.dust_sigma(MN_output)
            parameters_lim, min_lim,plus_lim = fn.dust_sigma(MN_outputlim)
            En_lim_min = min_lim[8:10].sum()/(parameters_values.sum())
            En_lim_plus = plus_lim[8:10].sum()/(parameters_values.sum())
            For_lim_min = min_lim[6:8].sum()/(parameters_values.sum())
            For_lim_plus = plus_lim[6:8].sum()/(parameters_values.sum())

            parameters_values, parameters_sigma_min, parameters_sigma_plus = fn.dust_sigma(MN_output)
            print(parameters_values.sum())
            For_frac = parameters_values[6:8].sum()/(parameters_values.sum())
            En_frac = parameters_values[8:10].sum()/(parameters_values.sum())
            For_sigma_min = parameters_sigma_min[6:8].sum()/(parameters_values.sum())
            For_sigma_plus = parameters_sigma_plus[6:8].sum()/(parameters_values.sum())
            En_sigma_min = parameters_sigma_min[8:10].sum()/(parameters_values.sum())
            En_sigma_plus = parameters_sigma_plus[8:10].sum()/(parameters_values.sum())
            # crystal_sigma = crystal_frac*(1+dust_sigma[6:10])
            # print(dust_parameters)
            w = For_sigma_min + For_sigma_plus / For_frac +En_sigma_min + En_sigma_plus / En_frac
            cumulative = cumulative.append({'Fo':For_frac,'En':En_frac,'w':w},ignore_index=True)
            print( star,For_frac,En_frac)
            mfc,marker,count = 'blue','o',0

            if En_frac - En_lim_min <0.001:
                # xuplims = [1]
                xuplims = True
                En_frac += En_lim_plus
                marker='|'
                print(star, "Enstatite limit")
                count+=1
                # plt.arrow(En_frac,For_frac,-0.01,0,color='red',hold=None,width=0.0001,head_width=0.01,head_length=0.005)
                # En_sigma_min = 10**(En_frac)
                # En_frac = En_frac + En_sigma_plus
            if For_frac - For_lim_min<0.001:
                # yuplims = [1]
                uplims = True
                For_frac += For_lim_plus
                marker='_'
                print(star, "Forsterite limit")
                count+=1
                if xuplims==True:
                    marker="x"
            plot = plt.errorbar(En_frac,For_frac,xerr=[[En_sigma_min],[En_sigma_plus]],yerr=[[For_sigma_min],[For_sigma_plus]],
                                        xuplims=xuplims,uplims=uplims,marker=marker,markersize=5,color = 'blue',elinewidth=0.3)
            if count==0 and s_once == 0:
                plot.set_label('5-17 $\mu m$')
                s_once+=1
            # else:
            #     plt.errorbar(En_frac,For_frac,xerr=[[En_sigma_min],[En_sigma_plus]],yerr=[[For_sigma_min],[For_sigma_plus]],
            #                             marker="o",color = 'blue',label = '5-17',elinewidth=0.3)

        except Exception as e:
            print(star + " does not have file",e)
        try:
            xuplims, uplims = False,False
            MN_output = pd.read_csv(path_long + star + "/full/parameters1sigma.csv",skiprows=7,index_col=False)
            MN_output.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            MN_outputlim = pd.read_csv(path_long + star + "/full/parameters3sigma.csv",skiprows=7,index_col=False)
            MN_outputlim.columns = ['kaas','Mean','SigmaMin','SigmaPlus']

            parameters_values, parameters_sigma_min, parameters_sigma_plus = fn.dust_sigma(MN_output)
            parameters_lim, min_lim,plus_lim = fn.dust_sigma(MN_outputlim)
            En_lim_min = min_lim[8:10].sum()/(parameters_values.sum())
            En_lim_plus = plus_lim[8:10].sum()/(parameters_values.sum())
            For_lim_min = min_lim[6:8].sum()/(parameters_values.sum())
            For_lim_plus = plus_lim[6:8].sum()/(parameters_values.sum())

            For_frac = parameters_values[6:8].sum()/(parameters_values.sum())
            En_frac = parameters_values[8:10].sum()/(parameters_values.sum())
            For_sigma_min = parameters_sigma_min[6:8].sum()/(parameters_values.sum())
            For_sigma_plus = parameters_sigma_plus[6:8].sum()/(parameters_values.sum())
            En_sigma_min = parameters_sigma_min[8:10].sum()/(parameters_values.sum())
            En_sigma_plus = parameters_sigma_plus[8:10].sum()/(parameters_values.sum())
            print( star,For_frac,En_frac)
            mfc,marker,count,label = 'red','o',0,'17-35 $\mu m$'
            if En_frac - En_lim_min <0.001:
                # xuplims = [1]
                xuplims = True
                En_frac += En_lim_plus
                marker='|'
                print(star, "Enstatite limit")
                count+=1
                # plt.arrow(En_frac,For_frac,-0.01,0,color='red',hold=None,width=0.0001,head_width=0.01,head_length=0.005)
                # En_sigma_min = 10**(En_frac)
                # En_frac = En_frac + En_sigma_plus
            if For_frac - For_lim_min<0.001:
                # yuplims = [1]
                uplims = True
                For_frac += For_lim_plus
                marker='_'
                print(star, "Forsterite limit")
                count+=1
                if xuplims==True:
                    marker="x"
            plot = plt.errorbar(En_frac,For_frac,xerr=[[En_sigma_min],[En_sigma_plus]],yerr=[[For_sigma_min],[For_sigma_plus]],
                                        marker=marker,markersize=5,xuplims=xuplims,uplims=uplims,color = 'red',elinewidth=0.3)
            # else:
                # plt.errorbar(En_frac,For_frac,xerr=[[En_sigma_min],[En_sigma_plus]],yerr=[[For_sigma_min],[For_sigma_plus]],
                                        # marker="^",color = 'red',label = '17-35',elinewidth=0.3)

            if count==0 and l_once == 0:
                plot.set_label('17-35 $\mu m$')
                l_once+=1

        except Exception as e:
            print(star + " does not have file",e)
    print(cumulative.corr())
    # print(np.cov(np.array(cumulative['SiO2'],cumulative['En'])))
    print('correlation coeff: ' ,corr(cumulative['Fo'],cumulative['En'],cumulative['w']))
    plt.plot(plt.xlim(),plt.xlim(),linestyle=":",color='black',label='M$_{en}$ = M$_{fo}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('M$_{En}$ / M$_{Tot}$',size=15)
    plt.ylabel('M$_{Fo}$ / M$_{Tot}$',size=15)
    plt.legend()
    plt.show()

def m(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)

def cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)

def corr(x, y, w):
    """Weighted Correlation"""
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))

def Forsterite():
    cumulative = pd.DataFrame(columns=['Fo','En','w'])
    starsdiana = ['AA-Tau','BP-Tau','CY-Tau','DF-Tau','DL-Tau','DM-Tau','DN-Tau','DR-Tau','FT-Tau','HK-Tau-B','LkCa15','RW-Aur','WX-Cha','XX-Cha']
    l_once,s_once = 0,0
    Forsterite_short = []
    Forsterite_long = []
    for star in stars:
        try:
            MN_output = pd.read_csv(path + star + "/full/parameters3sigma.csv",skiprows=10,index_col=False)
            MN_output.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            # MN_output = fn.get_all1sigma(star,['Ol. 0.1','Ol. 2.0','Ol. 5.0','Py. 0.1','Py. 2.0','Py. 5.0','Fo. 0.1','Fo. 2.0','En. 0.1','En. 2.0','Si. 0.1','Si. 2.0','Si. 5.0'],'short')
            parameters_values, parameters_sigma_min, parameters_sigma_plus = fn.dust_sigma(MN_output)
            print(parameters_values.sum())
            For_frac = parameters_values[6:8].sum()/(parameters_values.sum())
            En_frac = parameters_values[8:10].sum()/(parameters_values.sum())
            For_sigma_min = parameters_sigma_min[6:8].sum()/(parameters_values.sum())
            For_sigma_plus = parameters_sigma_plus[6:8].sum()/(parameters_values.sum())
            En_sigma_min = parameters_sigma_min[8:10].sum()/(parameters_values.sum())
            En_sigma_plus = parameters_sigma_plus[8:10].sum()/(parameters_values.sum())
            SiO2_frac = parameters_values[10:13].sum()/(parameters_values.sum())
            # crystal_sigma = crystal_frac*(1+dust_sigma[6:10])
            # print(dust_parameters)
            w = For_sigma_min + For_sigma_plus / For_frac +En_sigma_min + En_sigma_plus / En_frac
            cumulative = cumulative.append({'Fo':For_frac,'En':En_frac,'w':w},ignore_index=True)
            print( star,For_frac,En_frac)
            mfc,marker,count = 'blue','o',0

            if En_frac - En_sigma_min <0.001:
                # xuplims = [1]
                mfc = 'none'
                En_sigma_min = 0
                marker=8
                count+=1
                print(star, "Enstatite limit")
                # plt.arrow(En_frac,For_frac,-0.01,0,color='red',hold=None,width=0.0001,head_width=0.01,head_length=0.005)
                # En_sigma_min = 10**(En_frac)
                # En_frac = En_frac + En_sigma_plus
            if For_frac - For_sigma_min<0.001:
                # yuplims = [1]
                mfc = 'none'
                marker=11
                For_sigma_min = 0
                count+=1
                print(star, "Forsterite limit")
            if count==2:
                marker="D"
            plot = plt.errorbar(En_frac,For_frac,xerr=[[En_sigma_min],[En_sigma_plus]],yerr=[[For_sigma_min],[For_sigma_plus]],
                                        marker=marker,color = 'blue',mfc=mfc,elinewidth=0.3)
            if count==0 and s_once == 0:
                plot.set_label('5-17 $\mu m$')
                s_once+=1
            # else:
            #     plt.errorbar(En_frac,For_frac,xerr=[[En_sigma_min],[En_sigma_plus]],yerr=[[For_sigma_min],[For_sigma_plus]],
            #                             marker="o",color = 'blue',label = '5-17',elinewidth=0.3)
            Forsterite_short.append(For_frac+En_frac + SiO2_frac)
        except Exception as e:
            print(star + " does not have file",e)
        try:
            MN_output = pd.read_csv(path_long + star + "/full/parameters3sigma.csv",skiprows=7,index_col=False)
            MN_output.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            parameters_values, parameters_sigma_min, parameters_sigma_plus = fn.dust_sigma(MN_output)
            For_frac = parameters_values[6:8].sum()/(parameters_values.sum())
            En_frac = parameters_values[8:10].sum()/(parameters_values.sum())
            For_sigma_min = parameters_sigma_min[6:8].sum()/(parameters_values.sum())
            For_sigma_plus = parameters_sigma_plus[6:8].sum()/(parameters_values.sum())
            En_sigma_min = parameters_sigma_min[8:10].sum()/(parameters_values.sum())
            En_sigma_plus = parameters_sigma_plus[8:10].sum()/(parameters_values.sum())
            SiO2_frac = parameters_values[10:13].sum()/(parameters_values.sum())
            print( star,For_frac,En_frac)
            mfc,marker,count,label = 'red','o',0,'17-35 $\mu m$'
            if En_frac - En_sigma_min <0.001:
                # xuplims = [1]
                mfc = 'none'
                En_sigma_min = 0
                marker=8
                count+=1
                label=None
                # plt.arrow(En_frac,For_frac,-0.01,0,color='red',hold=None,width=0.0001,head_width=0.01,head_length=0.005)
                # En_sigma_min = 10**(En_frac)
                # En_frac = En_frac + En_sigma_plus
            if For_frac - For_sigma_min<0.001:
                # yuplims = [1]
                mfc = 'none'
                marker=11
                For_sigma_min = 0
                count+=1
                label=None
            if count==2:
                marker="D"

            plot = plt.errorbar(En_frac,For_frac,xerr=[[En_sigma_min],[En_sigma_plus]],yerr=[[For_sigma_min],[For_sigma_plus]],
                                        marker=marker,color = 'red',mfc=mfc,elinewidth=0.3)
            # else:
                # plt.errorbar(En_frac,For_frac,xerr=[[En_sigma_min],[En_sigma_plus]],yerr=[[For_sigma_min],[For_sigma_plus]],
                                        # marker="^",color = 'red',label = '17-35',elinewidth=0.3)

            if count==0 and l_once == 0:
                plot.set_label('17-35 $\mu m$')
                l_once+=1
            Forsterite_long.append(For_frac+En_frac+SiO2_frac)
        except Exception as e:
            print(star + " does not have file",e)
    print(cumulative.corr())
    # print(np.cov(np.array(cumulative['SiO2'],cumulative['En'])))
    print('correlation coeff: ' ,corr(cumulative['Fo'],cumulative['En'],cumulative['w']))
    plt.plot(plt.xlim(),plt.xlim(),linestyle=":",color='black',label='M$_{en}$ = M$_{fo}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('M$_{En}$ / M$_{Sil}$')
    plt.ylabel('M$_{Fo}$ / M$_{Sil}$')
    plt.legend()
    plt.show()
    plt.scatter(Forsterite_short,Forsterite_long)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

def EnstatiteSiO2():
    cumulative = pd.DataFrame(columns=['SiO2','En','w'])
    # starsdiana = ['AA-Tau','BP-Tau','CY-Tau','DF-Tau','DL-Tau','DM-Tau','DN-Tau','DR-Tau','FT-Tau','HK-Tau-B','LkCa15','RW-Aur','WX-Cha','XX-Cha']
    l_once,s_once=0,0
    for star in stars:
        try:
            xuplims, uplims = False,False
            MN_output = pd.read_csv(path + star + "/full/parameters1sigma.csv",skiprows=10,index_col=False)
            MN_output.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            MN_outputlim = pd.read_csv(path + star + "/full/parameters3sigma.csv",skiprows=10,index_col=False)
            MN_outputlim.columns = ['kaas','Mean','SigmaMin','SigmaPlus']

            parameters_values, parameters_sigma_min, parameters_sigma_plus = fn.dust_sigma(MN_output)
            parameters_lim, min_lim,plus_lim = fn.dust_sigma(MN_outputlim)
            En_lim_min = min_lim[8:10].sum()/(parameters_values.sum())
            En_lim_plus = plus_lim[8:10].sum()/(parameters_values.sum())
            SiO2_lim_min = min_lim[10:13].sum()/(parameters_values.sum())
            SiO2_lim_plus = plus_lim[10:13].sum()/(parameters_values.sum())

            parameters_values, parameters_sigma_min, parameters_sigma_plus = fn.dust_sigma(MN_output)
            print(parameters_values.sum())
            SiO2_frac = parameters_values[10:13].sum()/(parameters_values.sum())
            En_frac = parameters_values[8:10].sum()/(parameters_values.sum())
            SiO2_sigma_min = parameters_sigma_min[10:13].sum()/(parameters_values.sum())
            SiO2_sigma_plus = parameters_sigma_plus[10:13].sum()/(parameters_values.sum())
            En_sigma_min = parameters_sigma_min[8:10].sum()/(parameters_values.sum())
            En_sigma_plus = parameters_sigma_plus[8:10].sum()/(parameters_values.sum())
            # crystal_sigma = crystal_frac*(1+dust_sigma[6:10])
            # print(dust_parameters)
            w = SiO2_sigma_min + SiO2_sigma_plus / SiO2_frac +En_sigma_min + En_sigma_plus / En_frac
            print( star,SiO2_frac,En_frac)
            cumulative = cumulative.append({'SiO2':SiO2_frac,'En':En_frac,'w':w},ignore_index=True)
            mfc,marker,count,label = 'blue','o',0,'17-35 $\mu m$'
            if En_frac - En_lim_min <0.001:
                # xuplims = [1]
                xuplims = True
                En_frac += En_lim_plus
                marker='|'
                print(star, "Enstatite limit")
                count+=1
                # plt.arrow(En_frac,For_frac,-0.01,0,color='red',hold=None,width=0.0001,head_width=0.01,head_length=0.005)
                # En_sigma_min = 10**(En_frac)
                # En_frac = En_frac + En_sigma_plus
            if SiO2_frac - SiO2_lim_min<0.001:
                # yuplims = [1]
                uplims = True
                SiO2_frac += SiO2_lim_plus
                marker='_'
                print(star, "SiO2 limit")
                count+=1
                if xuplims==True:
                    marker="x"

            plot = plt.errorbar(En_frac,SiO2_frac,xerr=[[En_sigma_min],[En_sigma_plus]],yerr=[[SiO2_sigma_min],[SiO2_sigma_plus]],
                                        marker=marker,markersize=5,xuplims=xuplims,uplims=uplims,color = 'blue',elinewidth=0.3)

            if count==0 and s_once == 0:
                plot.set_label('5-17 $\mu m$')
                s_once+=1
        except Exception as e:
            print(star + " does not have file",e)
        try:
            xuplims, uplims = False,False
            MN_output = pd.read_csv(path_long + star + "/full/parameters1sigma.csv",skiprows=7,index_col=False)
            MN_output.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            MN_outputlim = pd.read_csv(path_long + star + "/full/parameters3sigma.csv",skiprows=7,index_col=False)
            MN_outputlim.columns = ['kaas','Mean','SigmaMin','SigmaPlus']

            parameters_values, parameters_sigma_min, parameters_sigma_plus = fn.dust_sigma(MN_output)
            parameters_lim, min_lim,plus_lim = fn.dust_sigma(MN_outputlim)
            En_lim_min = min_lim[8:10].sum()/(parameters_values.sum())
            En_lim_plus = plus_lim[8:10].sum()/(parameters_values.sum())
            SiO2_lim_min = min_lim[10:13].sum()/(parameters_values.sum())
            SiO2_lim_plus = plus_lim[10:13].sum()/(parameters_values.sum())

            parameters_values, parameters_sigma_min, parameters_sigma_plus = fn.dust_sigma(MN_output)
            SiO2_frac = parameters_values[10:13].sum()/(parameters_values.sum())
            En_frac = parameters_values[8:10].sum()/(parameters_values.sum())
            SiO2_sigma_min = parameters_sigma_min[10:13].sum()/(parameters_values.sum())
            SiO2_sigma_plus = parameters_sigma_plus[10:13].sum()/(parameters_values.sum())
            En_sigma_min = parameters_sigma_min[8:10].sum()/(parameters_values.sum())
            En_sigma_plus = parameters_sigma_plus[8:10].sum()/(parameters_values.sum())
            print( star,SiO2_frac,En_frac)
            mfc,marker,count,label = 'red','o',0,'17-35 $\mu m$'
            if En_frac - En_lim_min <0.001:
                # xuplims = [1]
                xuplims = True
                En_frac += En_lim_plus
                marker='|'
                print(star, "Enstatite limit")
                count+=1
                # plt.arrow(En_frac,For_frac,-0.01,0,color='red',hold=None,width=0.0001,head_width=0.01,head_length=0.005)
                # En_sigma_min = 10**(En_frac)
                # En_frac = En_frac + En_sigma_plus
            if SiO2_frac - SiO2_lim_min<0.001:
                # yuplims = [1]
                uplims = True
                SiO2_frac += SiO2_lim_plus
                marker='_'
                print(star, "SiO2 limit")
                count+=1
                if xuplims==True:
                    marker="x"

            plot = plt.errorbar(En_frac,SiO2_frac,xerr=[[En_sigma_min],[En_sigma_plus]],yerr=[[SiO2_sigma_min],[SiO2_sigma_plus]],
                                        marker=marker,markersize=5,xuplims=xuplims,uplims=uplims,color = 'red',elinewidth=0.3)

            if count==0 and l_once == 0:
                plot.set_label('17-35 $\mu m$')
                l_once+=1
        except Exception as e:
            print(star + " does not have file",e)
    print(cumulative.corr())
    # print(np.cov(np.array(cumulative['SiO2'],cumulative['En'])))
    print('correlation coeff: ' ,corr(cumulative['SiO2'],cumulative['En'],cumulative['w']))
    plt.plot(plt.xlim(),plt.xlim(),linestyle=":",color='black',label='M$_{en}$ = M$_{SiO_2}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('M$_{En}$ / M$_{Tot}$',size=15)
    plt.ylabel('M$_{SiO2}$ / M$_{Tot}$',size=15)
    plt.legend()
    plt.show()

def ForsteriteSiO2():
    l_once,s_once=0,0
    for star in stars:
        try:
            xuplims,uplims = False,False
            MN_output = pd.read_csv(path + star + "/full/parameters1sigma.csv",skiprows=10,index_col=False)
            MN_output.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            MN_outputlim = pd.read_csv(path + star + "/full/parameters3sigma.csv",skiprows=10,index_col=False)
            MN_outputlim.columns = ['kaas','Mean','SigmaMin','SigmaPlus']

            parameters_values, parameters_sigma_min, parameters_sigma_plus = fn.dust_sigma(MN_output)
            parameters_lim, min_lim,plus_lim = fn.dust_sigma(MN_outputlim)
            En_lim_min = min_lim[6:8].sum()/(parameters_values.sum())
            En_lim_plus = plus_lim[6:8].sum()/(parameters_values.sum())
            SiO2_lim_min = min_lim[10:13].sum()/(parameters_values.sum())
            SiO2_lim_plus = plus_lim[10:13].sum()/(parameters_values.sum())

            parameters_values, parameters_sigma_min, parameters_sigma_plus = fn.dust_sigma(MN_output)
            print(parameters_values.sum())
            SiO2_frac = parameters_values[10:13].sum()/(parameters_values.sum())
            En_frac = parameters_values[6:8].sum()/(parameters_values.sum())
            SiO2_sigma_min = parameters_sigma_min[10:13].sum()/(parameters_values.sum())
            SiO2_sigma_plus = parameters_sigma_plus[10:13].sum()/(parameters_values.sum())
            En_sigma_min = parameters_sigma_min[6:8].sum()/(parameters_values.sum())
            En_sigma_plus = parameters_sigma_plus[6:8].sum()/(parameters_values.sum())
            # crystal_sigma = crystal_frac*(1+dust_sigma[6:10])
            # print(dust_parameters)
            print( star,SiO2_frac,En_frac)
            mfc,marker,count,label = 'blue','o',0,'17-35 $\mu m$'
            if En_frac - En_lim_min <0.001:
                # xuplims = [1]
                xuplims = True
                En_frac += En_lim_plus
                marker='|'
                print(star, "Forsterite limit")
                count+=1
                # plt.arrow(En_frac,For_frac,-0.01,0,color='red',hold=None,width=0.0001,head_width=0.01,head_length=0.005)
                # En_sigma_min = 10**(En_frac)
                # En_frac = En_frac + En_sigma_plus
            if SiO2_frac - SiO2_lim_min<0.001:
                # yuplims = [1]
                uplims = True
                SiO2_frac += SiO2_lim_plus
                marker='_'
                print(star, "SiO2 limit")
                count+=1
                if xuplims==True:
                    marker="x"

            plot = plt.errorbar(En_frac,SiO2_frac,xerr=[[En_sigma_min],[En_sigma_plus]],yerr=[[SiO2_sigma_min],[SiO2_sigma_plus]],
                                        marker=marker,markersize=5,xuplims=xuplims,uplims=uplims,color = 'blue',elinewidth=0.3)

            if count==0 and s_once == 0:
                plot.set_label('5-17 $\mu m$')
                s_once+=1
        except Exception as e:
            print(star + " does not have file",e)
        try:
            xuplims,uplims = False,False
            MN_output = pd.read_csv(path_long + star + "/full/parameters1sigma.csv",skiprows=7,index_col=False)
            MN_output.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            MN_outputlim = pd.read_csv(path_long + star + "/full/parameters3sigma.csv",skiprows=7,index_col=False)
            MN_outputlim.columns = ['kaas','Mean','SigmaMin','SigmaPlus']

            parameters_values, parameters_sigma_min, parameters_sigma_plus = fn.dust_sigma(MN_output)
            parameters_lim, min_lim,plus_lim = fn.dust_sigma(MN_outputlim)
            En_lim_min = min_lim[6:8].sum()/(parameters_values.sum())
            En_lim_plus = plus_lim[6:8].sum()/(parameters_values.sum())
            SiO2_lim_min = min_lim[10:13].sum()/(parameters_values.sum())
            SiO2_lim_plus = plus_lim[10:13].sum()/(parameters_values.sum())

            parameters_values, parameters_sigma_min, parameters_sigma_plus = fn.dust_sigma(MN_output)
            print(parameters_values.sum())
            SiO2_frac = parameters_values[10:13].sum()/(parameters_values.sum())
            En_frac = parameters_values[6:8].sum()/(parameters_values.sum())
            SiO2_sigma_min = parameters_sigma_min[10:13].sum()/(parameters_values.sum())
            SiO2_sigma_plus = parameters_sigma_plus[10:13].sum()/(parameters_values.sum())
            En_sigma_min = parameters_sigma_min[6:8].sum()/(parameters_values.sum())
            En_sigma_plus = parameters_sigma_plus[6:8].sum()/(parameters_values.sum())
            # crystal_sigma = crystal_frac*(1+dust_sigma[6:10])
            # print(dust_parameters)
            print( star,SiO2_frac,En_frac)
            mfc,marker,count,label = 'blue','o',0,'17-35 $\mu m$'
            if En_frac - En_lim_min <0.001:
                # xuplims = [1]
                # xuplims = [1]
                xuplims = True
                En_frac += En_lim_plus
                marker='|'
                print(star, "Forsterite limit")
                count+=1
                # plt.arrow(En_frac,For_frac,-0.01,0,color='red',hold=None,width=0.0001,head_width=0.01,head_length=0.005)
                # En_sigma_min = 10**(En_frac)
                # En_frac = En_frac + En_sigma_plus
            if SiO2_frac - SiO2_lim_min<0.001:
                # yuplims = [1]
                uplims = True
                SiO2_frac += SiO2_lim_plus
                marker='_'
                print(star, "SiO2 limit")
                count+=1
                if xuplims==True:
                    marker="x"

            plot = plt.errorbar(En_frac,SiO2_frac,xerr=[[En_sigma_min],[En_sigma_plus]],yerr=[[SiO2_sigma_min],[SiO2_sigma_plus]],
                                        marker=marker,markersize=5,xuplims=xuplims,uplims=uplims,color = 'red',elinewidth=0.3)

            if count==0 and l_once == 0:
                plot.set_label('17-35 $\mu m$')
                l_once+=1
        except Exception as e:
            print(star + " does not have file",e)
    plt.plot(plt.xlim(),plt.xlim(),linestyle=":",color='black',label='M$_{SiO_2}$ = M$_{fo}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('M$_{Fo}$ / M$_{Tot}$',size=15)
    plt.ylabel('M$_{SiO2}$ / M$_{Tot}$',size=15)
    plt.legend()
    plt.show()

def FoSiO2_En():
    cumulative = pd.DataFrame(columns=['FoSiO2','En','w'])
    stars = ['AA-Tau','BP-Tau','CY-Tau','DF-Tau','DL-Tau','DM-Tau','DN-Tau','DR-Tau','FT-Tau','HK-Tau-B','LkCa15','RW-Aur','WX-Cha','XX-Cha']

    for star in stars:
        try:
            MN_output = pd.read_csv(path + star + "/full/parameters3sigma.csv",skiprows=10,index_col=False)
            MN_output.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            # MN_output = fn.get_all1sigma(star,['Ol. 0.1','Ol. 2.0','Ol. 5.0','Py. 0.1','Py. 2.0','Py. 5.0','Fo. 0.1','Fo. 2.0','En. 0.1','En. 2.0','Si. 0.1','Si. 2.0','Si. 5.0'],'short')
            parameters_values, parameters_sigma_min, parameters_sigma_plus = fn.dust_sigma(MN_output)
            print(parameters_values.sum())
            FoSiO2_frac = (parameters_values[6:8].sum()+parameters_values[10:13].sum())/(parameters_values.sum())
            En_frac = parameters_values[8:10].sum()/(parameters_values.sum())
            FoSiO2_sigma_min = (parameters_sigma_min[6:8].sum()+parameters_sigma_min[10:13].sum())/(parameters_values.sum())
            FoSiO2_sigma_plus = (parameters_sigma_plus[6:8].sum()+parameters_sigma_plus[10:13].sum())/(parameters_values.sum())
            En_sigma_min = parameters_sigma_min[8:10].sum()/(parameters_values.sum())
            En_sigma_plus = parameters_sigma_plus[8:10].sum()/(parameters_values.sum())
            # crystal_sigma = crystal_frac*(1+dust_sigma[6:10])
            # print(dust_parameters)
            w = FoSiO2_sigma_min + FoSiO2_sigma_plus / FoSiO2_frac +En_sigma_min + En_sigma_plus / En_frac
            cumulative = cumulative.append({'FoSiO2':FoSiO2_frac,'En':En_frac,'w':w},ignore_index=True)
            print( star,FoSiO2_frac,En_frac)
            if star in stars:
                plt.errorbar(En_frac,FoSiO2_frac,xerr=[[En_sigma_min],[En_sigma_plus]],yerr=[[FoSiO2_sigma_min],[FoSiO2_sigma_plus]],
                                marker='o',color = 'red',label = '5-17',elinewidth=0.3)
            else:
                plt.errorbar(En_frac,FoSiO2_frac,xerr=[[En_sigma_min],[En_sigma_plus]],yerr=[[FoSiO2_sigma_min],[FoSiO2_sigma_plus]],
                                marker='o',color = 'blue',label = '5-17',elinewidth=0.3)
        except Exception as e:
            print(star + " does not have file",e)
        try:
            MN_output = pd.read_csv(path_long + star + "/full/parameters3sigma.csv",skiprows=7,index_col=False)
            MN_output.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            parameters_values, parameters_sigma_min, parameters_sigma_plus = fn.dust_sigma(MN_output)
            SiO2_frac = (parameters_values[6:8].sum()+parameters_values[10:13].sum())/(parameters_values.sum())
            En_frac = parameters_values[8:10].sum()/(parameters_values.sum())
            SiO2_sigma_min = (parameters_sigma_min[6:8].sum()+parameters_sigma_min[10:13].sum())/(parameters_values.sum())
            SiO2_sigma_plus = (parameters_sigma_plus[6:8].sum()+parameters_sigma_plus[10:13].sum())/(parameters_values.sum())
            En_sigma_min = parameters_sigma_min[8:10].sum()/(parameters_values.sum())
            En_sigma_plus = parameters_sigma_plus[8:10].sum()/(parameters_values.sum())
            print( star,SiO2_frac,En_frac)
            # plt.errorbar(En_frac,SiO2_frac,xerr=[[En_sigma_min],[En_sigma_plus]],yerr=[[SiO2_sigma_min],[SiO2_sigma_plus]],
                            # marker='o',color = 'red',label = '17-35',elinewidth=0.3)
        except Exception as e:
            print(star + " does not have file",e)
    print(cumulative.corr())
    # print(np.cov(np.array(cumulative['SiO2'],cumulative['En'])))
    print('correlation coeff: ' ,corr(cumulative['FoSiO2'],cumulative['En'],cumulative['w']))
    plt.plot(plt.xlim(),plt.xlim(),linestyle=":",color='black',label='M$_{en}$ = M$_{fo}$')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('M$_{En}$ / M$_{Sil}$')
    plt.ylabel('M$_{(Fo + SiO2)}$ / M$_{Sil}$')
    plt.legend(['M$_{En}$ = M$_{(Fo + SiO2)}$','5-17 $\mu m$','17-35 $\mu m$'])
    plt.show()

def OliPyro():
    l_once,s_once=0,0
    for star in stars:
        xuplims,uplims=False,False
        try:
            MN_output = pd.read_csv(path + star + "/full/parameters1sigma.csv",skiprows=10,index_col=False)
            MN_output.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            MN_outputlim = pd.read_csv(path + star + "/full/parameters3sigma.csv",skiprows=10,index_col=False)
            MN_outputlim.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            # MN_output = fn.get_all1sigma(star,['Ol. 0.1','Ol. 2.0','Ol. 5.0','Py. 0.1','Py. 2.0','Py. 5.0','Fo. 0.1','Fo. 2.0','En. 0.1','En. 2.0','Si. 0.1','Si. 2.0','Si. 5.0'],'short')
            parameters_values, parameters_sigma_min, parameters_sigma_plus = fn.dust_sigma(MN_output)
            parameters_lim, min_lim,plus_lim = fn.dust_sigma(MN_outputlim)
            En_lim_min = min_lim[3:6].sum()/(parameters_values.sum())
            En_lim_plus = plus_lim[3:6].sum()/(parameters_values.sum())
            SiO2_lim_min = min_lim[0:3].sum()/(parameters_values.sum())
            SiO2_lim_plus = plus_lim[0:3].sum()/(parameters_values.sum())

            print(parameters_values.sum())

            SiO2_frac = parameters_values[0:3].sum()/(parameters_values.sum())
            En_frac = parameters_values[3:6].sum()/(parameters_values.sum())
            SiO2_sigma_min = parameters_sigma_min[0:3].sum()/(parameters_values.sum())
            SiO2_sigma_plus = parameters_sigma_plus[0:3].sum()/(parameters_values.sum())
            En_sigma_min = parameters_sigma_min[3:6].sum()/(parameters_values.sum())
            En_sigma_plus = parameters_sigma_plus[3:6].sum()/(parameters_values.sum())
            # crystal_sigma = crystal_frac*(1+dust_sigma[6:10])
            # print(dust_parameters)
            print( star,SiO2_frac,En_frac)
            mfc,marker,count,label = 'blue','o',0,'17-35 $\mu m$'
            if En_frac - En_lim_min <0.001:
                # xuplims = [1]
                xuplims = True
                En_frac += En_lim_plus
                marker='|'
                print(star, "Forsterite limit")
                count+=1
                # plt.arrow(En_frac,For_frac,-0.01,0,color='red',hold=None,width=0.0001,head_width=0.01,head_length=0.005)
                # En_sigma_min = 10**(En_frac)
                # En_frac = En_frac + En_sigma_plus
            if SiO2_frac - SiO2_lim_min<0.001:
                uplims = True
                SiO2_frac += SiO2_lim_plus
                marker='_'
                print(star, "SiO2 limit")
                count+=1
                if xuplims==True:
                    marker="x"

            plot = plt.errorbar(En_frac,SiO2_frac,xerr=[[En_sigma_min],[En_sigma_plus]],yerr=[[SiO2_sigma_min],[SiO2_sigma_plus]],
                                        marker=marker,markersize=5,xuplims=xuplims,uplims=uplims,color = 'blue',elinewidth=0.3)

            if count==0 and s_once == 0:
                plot.set_label('5-17 $\mu m$')
                s_once+=1
        except Exception as e:
            print(star + " does not have file",e)
        xuplims,uplims=False,False
        try:
            MN_output = pd.read_csv(path_long + star + "/full/parameters1sigma.csv",skiprows=7,index_col=False)
            MN_output.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            MN_outputlim = pd.read_csv(path_long + star + "/full/parameters3sigma.csv",skiprows=7,index_col=False)
            MN_outputlim.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            # MN_output = fn.get_all1sigma(star,['Ol. 0.1','Ol. 2.0','Ol. 5.0','Py. 0.1','Py. 2.0','Py. 5.0','Fo. 0.1','Fo. 2.0','En. 0.1','En. 2.0','Si. 0.1','Si. 2.0','Si. 5.0'],'short')
            parameters_values, parameters_sigma_min, parameters_sigma_plus = fn.dust_sigma(MN_output)
            parameters_lim, min_lim,plus_lim = fn.dust_sigma(MN_outputlim)
            En_lim_min = min_lim[3:6].sum()/(parameters_values.sum())
            En_lim_plus = plus_lim[3:6].sum()/(parameters_values.sum())
            SiO2_lim_min = min_lim[0:3].sum()/(parameters_values.sum())
            SiO2_lim_plus = plus_lim[0:3].sum()/(parameters_values.sum())

            print(parameters_values.sum())

            SiO2_frac = parameters_values[0:3].sum()/(parameters_values.sum())
            En_frac = parameters_values[3:6].sum()/(parameters_values.sum())
            SiO2_sigma_min = parameters_sigma_min[0:3].sum()/(parameters_values.sum())
            SiO2_sigma_plus = parameters_sigma_plus[0:3].sum()/(parameters_values.sum())
            En_sigma_min = parameters_sigma_min[3:6].sum()/(parameters_values.sum())
            En_sigma_plus = parameters_sigma_plus[3:6].sum()/(parameters_values.sum())
            # crystal_sigma = crystal_frac*(1+dust_sigma[6:10])
            # print(dust_parameters)
            print( star,SiO2_frac,En_frac)
            mfc,marker,count,label = 'blue','o',0,'17-35 $\mu m$'
            if En_frac - En_lim_min <0.001:
                # xuplims = [1]
                xuplims = True
                En_frac += En_lim_plus
                marker='|'
                print(star, "Forsterite limit")
                count+=1
                # plt.arrow(En_frac,For_frac,-0.01,0,color='red',hold=None,width=0.0001,head_width=0.01,head_length=0.005)
                # En_sigma_min = 10**(En_frac)
                # En_frac = En_frac + En_sigma_plus
            if SiO2_frac - SiO2_lim_min<0.001:
                uplims = True
                SiO2_frac += SiO2_lim_plus
                marker='_'
                print(star, "SiO2 limit")
                count+=1
                if xuplims==True:
                    marker="x"

            plot = plt.errorbar(En_frac,SiO2_frac,xerr=[[En_sigma_min],[En_sigma_plus]],yerr=[[SiO2_sigma_min],[SiO2_sigma_plus]],
                                        marker=marker,markersize=5,xuplims=xuplims,uplims=uplims,color = 'red',elinewidth=0.3)

            if count==0 and l_once == 0:
                plot.set_label('17-35 $\mu m$')
                l_once+=1
        except Exception as e:
            print(star + " does not have file",e)

    # print(np.cov(np.array(cumulative['SiO2'],cumulative['En'])))
    plt.plot(plt.xlim(),plt.xlim(),linestyle=":",color='black',label='M$_{Ol}$ = M$_{Py}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('M$_{Py}$ / M$_{Tot}$',size=15)
    plt.ylabel('M$_{Ol}$ / M$_{Tot}$',size=15)
    plt.legend()
    plt.show()

def get_3sigma(star,param,shortlong):
    """
    star        : name of star (str)
    param       : name of parameter (str)
    shortlong   : 'short' or 'long'

    Retrieve median and 3 sigma errors of
    parameter of choice for a given star
    returns [median, low 3sigma, high 3sigma]
    """

    from PyMultiNest.pymultinest.analyse import Analyzer
    if shortlong == 'short':
        a = Analyzer(23, outputfiles_basename = "/home/chris/Documents/Thesis/jupbooks/daniele/JulyRun/short/" + star +"/full/test-")
        params = ['C1','Trim_min','qrim','C2','Tmid_min','Tmid_max','qmid','Tatm_min','Tatm_max','qatm','Ol. 0.1','Ol. 2.0','Ol. 5.0','Py. 0.1','Py. 2.0','Py. 5.0','Fo. 0.1','Fo. 2.0','En. 0.1','En. 2.0','Si. 0.1','Si. 2.0','Si. 5.0']

    elif shortlong == 'long':
        a = Analyzer(20, outputfiles_basename = "/home/chris/Documents/Thesis/jupbooks/daniele/JulyRun/long/" + star +"/full/test-")
        params = ['C2','Tmid_min','Tmid_max','qmid','Tatm_min','Tatm_max','qatm','Ol. 0.1','Ol. 2.0','Ol. 5.0','Py. 0.1','Py. 2.0','Py. 5.0','Fo. 0.1','Fo. 2.0','En. 0.1','En. 2.0','Si. 0.1','Si. 2.0','Si. 5.0']
    i = params.index(param)
    stats = a.get_stats()
    p = stats['marginals'][i]['median']
    pl,ph = stats['marginals'][i]['3sigma']
    return p, p-pl, ph-p

def save_all_param():
    for star in stars:
        try:
            fn.get_all1sigma(star,23,'short')
            # fn.get_all1sigma(star,20,'long')
        except Exception as e:
            print(e)
    return

# path = path_long
def Grainsize_Flaring():
    # stars = ['AA-Tau','BP-Tau','CY-Tau','DF-Tau','DL-Tau','DM-Tau','DN-Tau','DR-Tau','FT-Tau','HK-Tau-B','LkCa15','RW-Aur','WX-Cha','XX-Cha']
    for star in stars:
        try:
            MN_output_short = pd.read_csv(path_long + star + "/full/parameters3sigma.csv",skiprows=7,index_col=False)
            MN_output_short.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            dust_parameters,parameters_sigma_min,parameters_sigma_plus = fn.dust_sigma(MN_output_short)
            amorph_params = dust_parameters[[0,1,3,4,10,11]]
            amorph_params = amorph_params/amorph_params.sum()
            amorph_sigma_min = parameters_sigma_min[[0,1,3,4,10,11]]/amorph_params.sum()
            amorph_sigma_plus = parameters_sigma_plus[[0,1,3,4,10,11]]/amorph_params.sum()

            crystal_params = dust_parameters[6:10]
            crystal_params = crystal_params / crystal_params.sum()
            crystal_sigma_min = parameters_sigma_min[6:10]/ crystal_params.sum()
            crystal_sigma_plus = parameters_sigma_plus[6:10]/ crystal_params.sum()
            # print(amorph_params[10],crystal_params[:2])


            small_amorph = 0.1 *(amorph_params[0] + amorph_params[3] + amorph_params[10] )

            middle_amorph = 2 * (amorph_params[1] + amorph_params[4] + amorph_params[11])
            # high_amorph = 5 * (amorph_params[2] + amorph_params[5] + amorph_params[12])
            avg_amorph = small_amorph + middle_amorph #+ high_amorph
            # ratio_amorph = middle_amorph/small_amorph
            ratio_amorph = (dust_parameters[1]+dust_parameters[4]+dust_parameters[11])/(dust_parameters[0]+dust_parameters[3]+dust_parameters[10])
            amorph_min = 0.1*(amorph_sigma_min[0]+ amorph_sigma_min[3]+ amorph_sigma_min[10]) +2*(amorph_sigma_min[1] + amorph_sigma_min[4]+ amorph_sigma_min[11])#+5*(amorph_sigma_min[2]+ amorph_sigma_min[5])
            amorph_plus = 0.1*(amorph_sigma_plus[0]+ amorph_sigma_plus[3]+ amorph_sigma_plus[10]) +2*(amorph_sigma_plus[1] + amorph_sigma_plus[4]+ amorph_sigma_plus[11])# +5*(amorph_sigma_plus[2]+ amorph_sigma_plus[5]+ amorph_sigma_plus[12])
            print('test')
            small_crystal = 0.1 *  (crystal_params[6] + crystal_params[8])
            middle_crystal = 2 * (crystal_params[7] + crystal_params[9])
            avg_crystal = small_crystal + middle_crystal
            # ratio_crystal = middle_crystal/small_crystal
            ratio_crystal = (dust_parameters[7]+dust_parameters[9])/(dust_parameters[6]+dust_parameters[8])
            crystal_min = 0.1*(crystal_sigma_min[6]+ crystal_sigma_min[8]) +2*(crystal_sigma_min[7] + crystal_sigma_min[9])
            crystal_plus = 0.1*(crystal_sigma_plus[6]+ crystal_sigma_plus[8]) +2*(crystal_sigma_plus[7] + crystal_sigma_plus[9])
            # print(star,amorph_plus,amorph_min)
            print(ratio_amorph,ratio_crystal)

            # small_mass = 0.1*(dust_parameters[0]+ dust_parameters[3]+ dust_parameters[6]+ dust_parameters[8]+ dust_parameters[10])
            # middle_mass = 2*(dust_parameters[1]+ dust_parameters[4]+ dust_parameters[7]+ dust_parameters[9]+ dust_parameters[11])
            # high_mass = 5*(dust_parameters[2]+ dust_parameters[5]+ dust_parameters[12])
            # sigma_min = 0.1*(parameters_sigma_min[0]+ parameters_sigma_min[3]+ parameters_sigma_min[6]+ parameters_sigma_min[8]+ parameters_sigma_min[10]) +2*(parameters_sigma_min[1] + parameters_sigma_min[4]+ parameters_sigma_min[7]+parameters_sigma_min[9]+ parameters_sigma_min[11])
            #     #+ 5*(dust_parameters[2]+ dust_parameters[5]+ dust_parameters[12])
            # sigma_plus = 0.1*(parameters_sigma_plus[0]+ parameters_sigma_plus[3]+ parameters_sigma_plus[6]+ parameters_sigma_plus[8]+ parameters_sigma_plus[10])+ 2*(parameters_sigma_plus[1]+ parameters_sigma_plus[4]+ parameters_sigma_plus[7]+ parameters_sigma_plus[9]+ parameters_sigma_plus[11])
            #     #+ 5*(dust_parameters[2]+ dust_parameters[5]+ dust_parameters[12])
            # mass_avg = small_mass + middle_mass + high_mass

            # spitzer_path = '/home/chris/Documents/Thesis/Diana/CSV/' + star + '.csv'
            # spitzer = pd.read_csv(spitzer_path,usecols=['wavelength','flux','sigma'])
            # F8 = spitzer[(spitzer['wavelength'] > 7.8) & (spitzer['wavelength'] < 8.2)]
            # F24 = spitzer[(spitzer['wavelength'] > 30) & (spitzer['wavelength'] < 31)]
            # # # crystal_sigma = crystal_frac*(1+dust_sigma[6:10])
            # print(star , np.mean(F24['flux'])/np.mean(F8['flux']))

            plt.errorbar(ratio_amorph,ratio_crystal,yerr=None,marker='o',color='black')
            # plt.errorbar(avg_amorph,avg_crystal,xerr=[[amorph_min],[amorph_plus]],yerr=[[crystal_min],[crystal_plus]],marker='o',color = 'black',label = '5-17')
        except Exception as e:
            print(star , e)
            print('plakje')
    # plt.xlabel('F$_{24}$/F$_8$')
    plt.plot([0.5,20],[0.5,20])
    plt.xlabel('mass averaged amorph grainsize ($\mu m$)')
    plt.ylabel('mass averaged crystal grainsize ($\mu m$)')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

# MN_output_short = pd.read_csv(path + star + "/full/test-stats.dat",sep='\s+',skiprows=2,nrows=23,index_col=False,usecols=[1,2])
# MN_output_short.columns = ['Mean','Sigma']
# MN_output_long = pd.read_csv(path_long + star + "/full/test-stats.dat",sep='\s+',skiprows=2,nrows=20,index_col=False,usecols=[1,2])
# MN_output_long.columns = ['Mean','Sigma']

def Total_abundance():
    total_dust_short,total_sigma_short = [],[]
    total_dust_long,total_sigma_long = [],[]
    fig,axes = plt.subplots(2,1,figsize=(8,8))
    for star in stars:
        try:
            # MN_output_short = pd.read_csv(path + star + "/full/parameters3sigma.csv",skiprows=10,index_col=False)
            # MN_output_short.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            MN_output_short = pd.read_csv(path + star + "/full/test-stats.dat",sep='\s+',skiprows=13,nrows=13,index_col=False,usecols=[1,2])
            MN_output_short.columns = ['Mean','Sigma']

            dust_parameters,parameters_sigma_min,parameters_sigma_plus = fn.dust_sigma_symmetric(MN_output_short)
            total_dust_short.append(dust_parameters)
            total_sigma_short.append(parameters_sigma_plus)
            # MN_output_long = pd.read_csv(path_long + star + "/full/parameters3sigma.csv",skiprows=7,index_col=False)
            # MN_output_long.columns = ['kaas','Mean','SigmaMin','SigmaPlus']
            MN_output_long = pd.read_csv(path_long + star + "/full/test-stats.dat",sep='\s+',skiprows=10,nrows=13,index_col=False,usecols=[1,2])
            MN_output_long.columns = ['Mean','Sigma']
            dust_parameters,parameters_sigma_min,parameters_sigma_plus = fn.dust_sigma_symmetric(MN_output_long)
            total_dust_long.append(dust_parameters)
            total_sigma_long.append(parameters_sigma_plus)
        except Exception as e:
            print(e)
    total_dust_short = np.array(total_dust_short)
    total_short = []
    total_short_sigmas = []
    # print(np.transpose(total_sigma_short)[3])
    # print(np.transpose(total_dust_short)[3])
    for i,item in enumerate(np.transpose(total_dust_short)):
        sigmas = np.transpose(total_sigma_short)[i]
        # print(sigmas,item)
        # quit()
        weighted_mean = item/(sigmas**2)
        short_sigmas= (1/(sigmas**2))
        # weighted_mean = weighted_mean.sum()/short_sigmas.sum()
        total_short.append(weighted_mean.sum()/short_sigmas.sum())
        total_short_sigmas.append(np.sqrt(1/(short_sigmas.sum())))
        print(total_short)
    # total_short_sigmas = total_short_sigmas/np.sum(total_short)
    # total_short = total_short/np.sum(total_short)


    total_dust_long = np.array(total_dust_long)
    total_long = []
    for item in np.transpose(total_dust_long):
        # print(item.sum())
        total_long.append(item.sum())
    total_long = total_long/np.sum(total_long)
    barlist=['Ol. 0.1','Ol. 2.0','Ol. 5.0','Py. 0.1','Py. 2.0','Py. 5.0','Fo. 0.1','Fo. 2.0','En. 0.1','En. 2.0','Si. 0.1','Si. 2.0','Si. 5.0']
    axes[0].bar(barlist,total_short,yerr=total_short_sigmas,label='5-17 $\mu m$')
    axes[1].bar(barlist,total_long,label='17-35 $\mu m$',color='red')
    axes[0].set_ylabel('Mass fraction',size=15)
    axes[1].set_ylabel('Mass fraction',size=15)
    axes[0].set_ylim(0,0.4)
    axes[1].set_ylim(0,0.4)
    axes[0].tick_params(labelsize=15)
    axes[1].tick_params(labelsize=15)
    plt.suptitle('Sample averaged mass fractions',size=15)
    axes[0].legend()
    axes[1].legend()
    print(total_long[0:3].sum(),total_long[3:6].sum())
    plt.show()
    return

def chisqg(ydata,ymod,sd=None):
      # Chi-square statistic
      if False:
           chisq=np.sum((ydata-ymod)**2)
      else:
           chisq=np.sum( ((ydata-ymod)/sd)**2 )

      return chisq

def ForEnstatite2():
    for star in stars:
        try:
            MN_output = pd.read_csv(path + star + "/full/test-stats.dat",sep='\s+',skiprows=2,nrows=23,index_col=False,usecols=[1,2])
            MN_output.columns = ['Mean','Sigma']
            parameters_values = MN_output['Mean']
            parameters_sigma = MN_output['Sigma']
            kap = [0.1*3,2*3,5*3,0.1*2.8,2*2.8,5*2.8,0.1*3.2,2*3.2,0.1*2.8,2*2.8,0.1*2.2,2*2.2,5*2.2]
            dusties = np.multiply(kap,parameters_values[10:23])
            parameters_sigma_min = dusties/dusties.sum()
            dust_sigma = parameters_sigma[10:23]/dusties.sum()
            dust_sigma = np.multiply(kap,dust_sigma)
            For_frac = dust_parameters[6:8].sum()/(dust_parameters.sum())
            En_frac = dust_parameters[8:10].sum()/(dust_parameters.sum())

            # crystal_sigma = crystal_frac*(1+dust_sigma[6:10])
            # print(dust_parameters)
            print( star,For_frac,En_frac)

        except Exception as e:
            print(star + " does not have file",e)
        try:
            MN_output_long = pd.read_csv(path_long + star + "/full/test-stats.dat",sep='\s+',skiprows=2,nrows=20,index_col=False,usecols=[1,2])
            MN_output_long.columns = ['Mean','Sigma']
            parameters_values_long = MN_output_long['Mean']
            parameters_sigma_long = MN_output_long['Sigma']
            # kap = [0.1*3,2*3,5*3,0.1*2.8,2*2.8,5*2.8,0.1*3.2,2*3.2,0.1*2.8,2*2.8,0.1*2.2,2*2.2,5*2.2]
            dusties_long = np.multiply(kap,parameters_values_long[7:20])
            dust_parameters_long = dusties_long/dusties_long.sum()
            dust_sigma_long = parameters_sigma_long[7:20]/dusties_long.sum()
            dust_sigma_long = np.multiply(kap,dust_sigma_long)
            For_frac_long = dust_parameters_long[6:8].sum()/(dust_parameters_long.sum())
            En_frac_long = dust_parameters_long[8:10].sum()/(dust_parameters_long.sum())

            # crystal_sigma = crystal_frac*(1+dust_sigma[6:10])
            # print(dust_parameters)
            print(star,For_frac_long,En_frac_long)
            # plt.errorbar(En_frac_long,For_frac_long,marker='o',color = 'red',label='17-35')
        except Exception as e:
            print(star + " does not have file",e)
        plt.errorbar(En_frac/For_frac,En_frac_long/For_frac_long,marker='o',color = 'red',label='17-35')

    plt.plot(plt.xlim(),plt.xlim(),linestyle=":",color='black',label='M$_{en}$ = M$_{fo}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('M$_{En}$ / M$_{Fo}$ 5-17')
    plt.ylabel('M$_{En}$ / M$_{Fo}$ 17-37')
    plt.legend(['no change','data'])
    plt.show()

def correlation():
    """
    Creates chain of .txt files with two columns [x,y]
    Each .txt file represents one star from the sample
    x = (sum of) posterior points of parameter(set) 1
    y = (sum of) posterior points of parameter(set) 2

    Currently set for short wavelengths, can change in
    line a = Analyzer(23,........)
    """
    from PyMultiNest.pymultinest.analyse import Analyzer
    kap = [0.1*3,2*3,5*3,0.1*2.8,2*2.8,5*2.8,0.1*3.2,2*3.2,0.1*2.8,2*2.8,0.1*2.2,2*2.2,5*2.2]
    text = ""
    # stars = ['AA-Tau','BP-Tau','CY-Tau','DF-Tau','DL-Tau','DM-Tau','DN-Tau','DR-Tau','FT-Tau','HK-Tau-B','LkCa15','RW-Aur','WX-Cha','XX-Cha']

    for i,star in enumerate(stars):
        x = []
        y= []

        if "full" in listdir(path+star):
            # short wavelength
            a = Analyzer(23, outputfiles_basename = "/home/chris/Documents/Thesis/jupbooks/daniele/JulyRun/short/" + star +"/full/test-")
            # long wavelength
            # a = Analyzer(23, outputfiles_basename = "/home/chris/Documents/Thesis/jupbooks/daniele/JulyRun/long/" + star +"/full/test-")

            for params in a.get_equal_weighted_posterior():
                Dust_values = np.multiply(kap,params[-14:-1])
                Dust_values = Dust_values/Dust_values.sum()
                Enstatite = Dust_values[3:6].sum()              # total mass fraction of enstatite grains
                Forsterite = Dust_values[0:3].sum()             # total mass fraction of forsterite grains
                Silica = Dust_values[-3:].sum()                 # total mass fraction of Silica grains
                x.append(Enstatite)
                y.append(Forsterite)
            x = np.array(x)
            y =np.array(y)
            chain = np.transpose([x, y])
            np.savetxt('/home/chris/Documents/Thesis/jupbooks/correlation/syscorr/example/sysline/chain' + str(i), chain)
            text = text + 'in/datapoint' + str(i) + '.txt '
    print(text)

def runit():
    os.chdir("/home/chris/Documents/Thesis/jupbooks/PyMultiNest/pymultinest")
    output_basename = sys.argv[1]
    files = sys.argv[2:]
    resume = os.environ.get('RESUME', '0') == '1'
    example_models = dict(
    	# y is drawn independently of x, from a gaussian. 2 free parameters (stdev, mean)
    	independent_normal = PolyModel(1, rv_type = scipy.stats.norm),

    	# y is drawn independently of x, from a uniform dist. 2 free parameters (width, low)
    	independent_uniform = PolyModel(1, rv_type = scipy.stats.uniform),

    	# y is drawn as a function of x, with gaussian systematic scatter
    	# y ~ N(b*x + a, s)      3 free parameters (s, a, b)
    	line = PolyModel(2),

    	# y is drawn as a function of x, with gaussian systematic scatter
    	# y ~ N(c*x**2 + b*x + a, s)      4 free parameters (s, a, b, c)
    	square = PolyModel(3),

    	# two gaussians before and after 0. 4 parameters (means and stdevs)
    	tribinned = BinnedModel(bins = [(-0.5, 0.3), (0.3, 0.8), (0.8, 1.5)], rv_type = scipy.stats.norm)
    )
    modelnames = os.environ.get('MODELS', '').split()
    if len(modelnames) == 0:
    	modelnames = example_models.keys()

    models = []
    for m in modelnames:
    	print('using model:', m)
    	model = example_models[m]
    	models.append((m, model))
    print('loading data...')
    chains = [(f, numpy.loadtxt(f)) for f in files]
    print('loading data done.')

    syscorr.calc_models(models=models,
    	chains=chains,
    	output_basename=output_basename,
        resume=resume)

# Marginals()
# MarginalsSample()
# copy_files()
# FitDustPlot()
# SamplePlot()
# import_folders()
# TotalPlot()
# Likelihood_Crystal()
# ForEnstatite()
# Forsterite()
# Grainsize_Flaring()
Total_abundance()
# ForEnstatite()
# save_all_param()
# EnstatiteSiO2()
# ForsteriteSiO2()
# FoSiO2_En()
# runit()
# correlation()
# featureCrystal()
# OliPyro()
# FitDustPlot()
