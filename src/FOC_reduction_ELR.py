#!/usr/bin/python
#-*- coding:utf-8 -*-

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from aplpy import FITSFigure
from scipy import ndimage
from scipy import signal
from skimage.feature import register_translation
from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel
import os as os

import lib.fits as proj_fits        #Functions to handle fits files
import lib.reduction as proj_red    #Functions used in reduction pipeline
import lib.plots as proj_plots      #Functions for plotting data

def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    if not operation.lower() in ['sum', 'median', 'std']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        if operation.lower() == "sum":
            ndarray = ndarray.sum(-1*(i+1))
        if operation.lower() == "median":
            ndarray = ndarray.mean(-1*(i+1))
        if operation.lower() == "std":
            ndarray = ndarray.std(-1*(i+1))
    return ndarray

#load files
def load_files(infiles):
    data_array = []
    for name in infiles['filenames']:
        with fits.open(infiles['data_folder']+name) as f:
            data_array.append(f[0].data)
    data_array = np.array(data_array)
    return data_array

def rebin_array(data,config):
    data_m = []
    data_s = []
    for ii in range(len(data)):
        new_shape = (data[ii].shape[1]//config['downsample_factor'],\
                data[ii].shape[0]//config['downsample_factor'])
        data_m.append(bin_ndarray(data[ii], new_shape, operation='median'))
        data_s.append(bin_ndarray(data[ii], new_shape, operation='std'))
    return data_m, data_s

def cross_corr(data):
    shifts = []
    data_array_c = []
    for ii in range(len(data)):
        shift, error, diffphase = register_translation(data[0], data_array[ii])
        print('   -',infiles['filenames'][ii],shift)
        shifts.append(shift)
        ima_shifted = np.roll(data[ii],round(shifts[ii][0]),axis=0)
        ima_shifted = np.roll(ima_shifted,round(shifts[ii][1]),axis=1)
        data_array_c.append(ima_shifted)
    return data_array_c

plt.close('all')


### User input ####
#Create dictionaries
infiles = {'data_folder':'../data/NGC1068_x274020/',\
            'filenames':['x274020at.c0f.fits','x274020bt.c0f.fits','x274020ct.c0f.fits',\
                        'x274020dt.c0f.fits','x274020et.c0f.fits','x274020ft.c0f.fits','x274020gt.c0f.fits',\
                        'x274020ht.c0f.fits','x274020it.c0f.fits']}

config = {'conserve_flux':True,'upsample_factor':1,\
            'downsample_factor':8}

fig = {'cmap':'magma','vmin':-20,'vmax':200,'font_size':40.0,'label_size':20,'lw':3,\
        'SNRp_cut':3,'SNRi_cut':4,'scalevec':0.05,'step_vec':1,\
        'vec_legend':10,'figname':'NGC1068_FOC_ELR'}
#####################

##### SCRIPT #####
#load files
print('--- Load files')
print('  -',infiles['filenames'])
data_array = load_files(infiles)
#crosscorrelation
print('--- Cross-correlate HWP PAs')
data_array_c = cross_corr(data_array)
#resample image
print('--- Downsample image')
data_array_cr_m, data_array_cr_s = rebin_array(data_array_c,config)
#smoothing
print('--- Smoothing')
kernel = Gaussian2DKernel(x_stddev=1)
data_array_crs_m = []
data_array_crs_s = []
for ii in range(len(data_array_cr_m)):
    data_array_crs_m.append(convolve(data_array_cr_m[ii], kernel))
    data_array_crs_s.append(convolve(data_array_cr_m[ii], kernel))

#combine HWP PAs
pol0   = (data_array_crs_m[0]+data_array_crs_m[1]+data_array_crs_m[2])
pol60  = (data_array_crs_m[3]+data_array_crs_m[4]+data_array_crs_m[5]+data_array_crs_m[6])
pol120 = (data_array_crs_m[7]+data_array_crs_m[8])

#Stokes parameters
I_stokes = (2./3.)*(pol0 + pol60 + pol120)
Q_stokes = (2./3.)*(2*pol0 - pol60 - pol120)
U_stokes = (2./np.sqrt(3.))*(pol60 - pol120)

#rotate Stokes
#PA = 46.81 #deg
#I_stokes = ndimage.rotate(I_stokes,-PA,reshape=True)
#Q_stokes = ndimage.rotate(Q_stokes,-PA,reshape=True)
#U_stokes = ndimage.rotate(U_stokes,-PA,reshape=True)
#Q_stokes = Q_stokes*np.cos(np.deg2rad(PA)) - U_stokes*np.sin(np.deg2rad(PA))
#U_stokes = Q_stokes*np.sin(np.deg2rad(PA)) + U_stokes*np.cos(np.deg2rad(PA))


#Polarimetry
PI  = np.sqrt(Q_stokes*Q_stokes + U_stokes*U_stokes)
P   = PI/I_stokes*100
PA  = 0.5*arctan2(U_stokes,Q_stokes)*180./np.pi+90
s_P  = np.sqrt(2.)*(I_stokes)**(-0.5)
s_PA = s_P/(P/100.)*180./np.pi


### STEP 6. image to FITS with updated WCS
img = fits.open(infiles['data_folder']+infiles['filenames'][0])
new_wcs = WCS(naxis=2)
new_wcs.wcs.crpix = [I_stokes.shape[0]/2, I_stokes.shape[1]/2]
new_wcs.wcs.crval = [img[0].header['CRVAL1'], img[0].header['CRVAL2']]
new_wcs.wcs.cunit = ["deg", "deg"]
new_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
new_wcs.wcs.cdelt = [img[0].header['CD1_1']*config['downsample_factor'],\
                     img[0].header['CD1_2']*config['downsample_factor']]

#hdu_ori = WCS(img[0])
stkI=fits.PrimaryHDU(data=I_stokes,header=new_wcs.to_header())
pol=fits.PrimaryHDU(data=P,header=new_wcs.to_header())
pang=fits.PrimaryHDU(data=PA,header=new_wcs.to_header())
pol_err=fits.PrimaryHDU(data=s_P,header=new_wcs.to_header())
pang_err=fits.PrimaryHDU(data=s_PA,header=new_wcs.to_header())


### STEP 7. polarization map
#quality cuts
pxscale = stkI.header['CDELT1']

#apply quality cuts
SNRp = pol.data/pol_err.data
pol.data[SNRp < fig['SNRp_cut']] = np.nan

SNRi = stkI.data/np.std(stkI.data[0:10,0:10])
SNRi = fits.PrimaryHDU(SNRi,header=stkI.header)
pol.data[SNRi.data < fig['SNRi_cut']] = np.nan

levelsI = 2**(np.arange(2,20,0.5))

fig1 = plt.figure(figsize=(11,10))
gc = FITSFigure(stkI,figure=fig1)
gc.show_colorscale(cmap=fig['cmap'],vmin=fig['vmin'],vmax=fig['vmax'])
gc.show_contour(SNRi,levels=levelsI,colors='grey',linewidths=0.5)
gc.show_vectors(pol,pang,scale=fig['scalevec'],\
                step=fig['step_vec'],color='black',linewidth=2.0)
gc.show_vectors(pol,pang,scale=fig['scalevec'],\
                step=fig['step_vec'],color='white',linewidth=1.0)
#legend vector
vecscale = fig['scalevec'] * new_wcs.wcs.cdelt[0]
gc.add_scalebar(fig['vec_legend']*vecscale,'P ='+np.str(fig['vec_legend'])+'%',\
                corner='bottom right',frame=True,color='black',facecolor='blue')
figname = figname = fig['figname']+'_test_polmap.pdf'
fig1.savefig(figname,dpi=300)
os.system('open '+figname)

sys.exit()














fig2,((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(40,20),dpi=200)
CF = ax1.imshow(I_stokes,origin='lower')
cbar = plt.colorbar(CF,ax=ax1)
cbar.ax.tick_params(labelsize=20)
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.text(2,2,'I',color='white',fontsize=10)

CF = ax2.imshow(Q_stokes,origin='lower')
cbar = plt.colorbar(CF,ax=ax2)
cbar.ax.tick_params(labelsize=20)
ax2.tick_params(axis='both', which='major', labelsize=20)
ax2.text(2,2,'Q',color='white',fontsize=10)

CF = ax3.imshow(U_stokes,origin='lower')
cbar = plt.colorbar(CF,ax=ax3)
cbar.ax.tick_params(labelsize=20)
ax3.tick_params(axis='both', which='major', labelsize=20)
ax3.text(2,2,'U',color='white',fontsize=10)

v = np.linspace(0,40,50)
CF = ax4.imshow(P,origin='lower',vmin=0,vmax=40)
cbar = plt.colorbar(CF,ax=ax4)
cbar.ax.tick_params(labelsize=20)
ax4.tick_params(axis='both', which='major', labelsize=20)
ax4.text(2,2,'P',color='white',fontsize=10)

CF = ax5.imshow(PA,origin='lower',vmin=0,vmax=180)
cbar = plt.colorbar(CF,ax=ax5)
cbar.ax.tick_params(labelsize=20)
ax5.tick_params(axis='both', which='major', labelsize=20)
ax5.text(2,2,'PA',color='white',fontsize=10)

CF = ax6.imshow(PI,origin='lower')
cbar = plt.colorbar(CF,ax=ax6)
cbar.ax.tick_params(labelsize=20)
ax6.tick_params(axis='both', which='major', labelsize=20)
ax6.text(2,2,'PI',color='white',fontsize=10)

figname = fig['figname']+'_test_Stokes.pdf'
fig2.savefig(figname,dpi=300)
#os.system('open '+figname)



### Step 4. Binning and smoothing
#Images can be binned and smoothed to improve SNR. This step can also be done
#using the PolX images.


### Step 5. Roate images to have North up
#Images needs to be reprojected to have North up.
#this procedure implies to rotate the Stokes QU using a rotation matrix


### STEP 6. image to FITS with updated WCS
img = fits.open(infiles['data_folder']+infiles['filenames'][0])
new_wcs = WCS(naxis=2)
new_wcs.wcs.crpix = [I_stokes.shape[0]/2, I_stokes.shape[1]/2]
new_wcs.wcs.crval = [img[0].header['CRVAL1'], img[0].header['CRVAL2']]
new_wcs.wcs.cunit = ["deg", "deg"]
new_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
new_wcs.wcs.cdelt = [img[0].header['CD1_1']*config['downsample_factor'],\
                     img[0].header['CD1_2']*config['downsample_factor']]

#hdu_ori = WCS(img[0])
stkI=fits.PrimaryHDU(data=I_stokes,header=new_wcs.to_header())
pol=fits.PrimaryHDU(data=P,header=new_wcs.to_header())
pang=fits.PrimaryHDU(data=PA,header=new_wcs.to_header())
pol_err=fits.PrimaryHDU(data=s_P,header=new_wcs.to_header())
pang_err=fits.PrimaryHDU(data=s_PA,header=new_wcs.to_header())


### STEP 7. polarization map
#quality cuts
pxscale = stkI.header['CDELT1']

#apply quality cuts
SNRp = pol.data/pol_err.data
pol.data[SNRp < fig['SNRp_cut']] = np.nan

SNRi = stkI.data/np.std(stkI.data[0:10,0:10])
SNRi = fits.PrimaryHDU(SNRi,header=stkI.header)
pol.data[SNRi.data < fig['SNRi_cut']] = np.nan
print(np.max(SNRi))
fig1 = plt.figure(figsize=(11,10))
gc = FITSFigure(stkI,figure=fig1)
gc.show_colorscale(fig['cmap'])
#gc.show_contour(np.log10(SNRi.data),levels=np.linspace(np.log10(fig['SNRi_cut']),np.nanmax(np.log10(SNRi.data,20))),\
#                filled=True,cmap='magma')
gc.show_vectors(pol,pang,scale=scalevec,step=step_vec,color='white',linewidth=1.0)

figname = figname = fig['figname']+'_test_polmap.pdf'
fig1.savefig(figname,dpi=300)
os.system('open '+figname)
