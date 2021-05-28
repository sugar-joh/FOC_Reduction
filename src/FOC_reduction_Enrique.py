#!/usr/bin/python
#-*- coding:utf-8 -*-

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from aplpy import FITSFigure
import scipy.ndimage
import os as os

import lib.fits as proj_fits        #Functions to handle fits files
import lib.reduction as proj_red    #Functions used in reduction pipeline
import lib.plots as proj_plots      #Functions for plotting data

plt.close('all')

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
    if not operation.lower() in ['sum', 'mean', 'average', 'avg']:
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
        elif operation.lower() in ["mean", "average", "avg"]:
            ndarray = ndarray.mean(-1*(i+1))
    return ndarray

def plots(ax,data,vmax,vmin):
    ax.imshow(data,vmax=vmax,vmin=vmin,origin='lower')



### User input

#infiles = ['x274020at.c0f.fits','x274020bt.c0f.fits','x274020ct.c0f.fits','x274020dt.c0f.fits',
#             'x274020et.c0f.fits','x274020ft.c0f.fits','x274020gt.c0f.fits','x274020ht.c0f.fits',
#             'x274020it.c0f.fits']

globals()['data_folder'] = "../data/3C405_x136060/"
infiles = ['x1360601t_c0f.fits','x1360602t_c0f.fits','x1360603t_c0f.fits']
#infiles = ['x1360601t_c1f.fits','x1360602t_c1f.fits','x1360603t_c1f.fits']
globals()['plots_folder'] = "../plots/3C405_x136060/"

#Centroids
#The centroids should be estimated by cross-correlating the images.
#Here I used the position of the central source for each file as the reference pixel position.
#ycen_0 = [304,306,303,296,295,295,294,305,304]
#xcen_0 = [273,274,273,276,274,274,274,272,271]

data_array = []
for name in infiles:
    with fits.open(data_folder+name) as f:
        data_array.append(f[0].data)
data_array = np.array(data_array)
shape = data_array.shape
data_array, error_array = proj_red.crop_array(data_array, step=5, null_val=0., inside=True)
data_array, error_array, shifts, errors = proj_red.align_data(data_array, error_array)
center = np.array([(np.array(shape[1:])/2).astype(int),]*len(infiles))-shifts
xcen_0 = center[:,0].astype(int)
ycen_0 = center[:,1].astype(int)

#size, in pixels, of the FOV centered in x_cen_0,y_cen_0
Dx = 500
Dy = 500

#set the new image size (Dxy x Dxy pixels binning)
Dxy = 10
new_shape   = (Dx//Dxy,Dy//Dxy)

#figures
#test alignment
vmin = 0
vmax = 6
font_size=40.0
label_size=20.
lw = 3.

#pol. map
SNRp_cut = 3 #P measumentes with SNR>3
SNRi_cut = 5 #I measuremntes with SNR>30, which implies an uncerrtainty in P of 4.7%.
scalevec = 0.05 #length of vectors in pol. map
step_vec = 1    #plot all vectors in the array. if step_vec = 2, then every other vector will be plotted
vec_legend = 10 #% pol for legend
#figname = 'NGC1068_FOC.png'
figname = '3C405_FOC_Enrique'


### SCRIPT ###
### Step 1. Check input images before data reduction
#this step is very simplistic.
#Here I used the position of the central source for each file as the
#reference pixel position.
#The centroids should be estimated by cross-correlating the images,
#and compare with the simplistic approach of using the peak pixel of the
#object as the reference pixel.


fig,axs = plt.subplots(3,3,figsize=(30,30),dpi=200,sharex=True,sharey=True)

for jj, enum in enumerate(list(zip(axs.flatten(),data_array))):
    a = enum[0]
    img = fits.open(data_folder+infiles[jj])
    ima = img[0].data
    ima = ima[ycen_0[jj]-Dy:ycen_0[jj]+Dy,xcen_0[jj]-Dx:xcen_0[jj]+Dx]
    ima = bin_ndarray(ima,new_shape=new_shape,operation='sum') #binning
    exptime = img[0].header['EXPTIME']
    fil = img[0].header['FILTNAM1']
    ima = ima/exptime
    globals()['ima_%s' % jj] = ima
    #plots
    plots(a,ima,vmax=vmax,vmin=vmin)
    #position of centroid
    a.plot([ima.shape[1]/2,ima.shape[1]/2],[0,ima.shape[0]-1],lw=1,color='black')
    a.plot([0,ima.shape[1]-1],[ima.shape[1]/2,ima.shape[1]/2],lw=1,color='black')
    a.text(2,2,infiles[jj][0:8],color='white',fontsize=10)
    a.text(2,5,fil,color='white',fontsize=30)
    a.text(ima.shape[1]-20,1,exptime,color='white',fontsize=20)
fig.subplots_adjust(hspace=0,wspace=0)
fig.savefig(plots_folder+figname+'_test_alignment.png',dpi=300)
#os.system('open test_alignment.png')



### Step 2. average of all images for a single polarizer to have them in the same units e/s.
pol0   = ima_0#(ima_0 + ima_1 + ima_2)/3.
pol60  = ima_1#(ima_3 + ima_4 + ima_5 + ima_6)/4.
pol120 = ima_2#(ima_7 + ima_8)/2.

fig1,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(26,8),dpi=200)
CF = ax1.imshow(pol0,vmin=vmin,vmax=vmax,origin='lower')
cbar = plt.colorbar(CF,ax=ax1)
cbar.ax.tick_params(labelsize=20)
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.text(2,2,'POL0',color='white',fontsize=10)

CF = ax2.imshow(pol60,vmin=vmin,vmax=vmax,origin='lower')
cbar = plt.colorbar(CF,ax=ax2)
cbar.ax.tick_params(labelsize=20)
ax2.tick_params(axis='both', which='major', labelsize=20)
ax2.text(2,2,'POL60',color='white',fontsize=10)

CF = ax3.imshow(pol120,vmin=vmin,vmax=vmax,origin='lower')
cbar = plt.colorbar(CF,ax=ax3)
cbar.ax.tick_params(labelsize=20)
ax3.tick_params(axis='both', which='major', labelsize=20)
ax3.text(2,2,'POL120',color='white',fontsize=10)
fig1.savefig(plots_folder+figname+'_test_combinePol.png',dpi=300)
#os.system('open test_combinePol.png')


### Step 3. Compute Stokes IQU, P, PA, PI
#Stokes parameters
I_stokes = (2./3.)*(pol0 + pol60 + pol120)
Q_stokes = (2./3.)*(2*pol0 - pol60 - pol120)
U_stokes = (2./np.sqrt(3.))*(pol60 - pol120)

#Remove nan
I_stokes[np.isnan(I_stokes)]=0.
Q_stokes[np.isnan(Q_stokes)]=0.
U_stokes[np.isnan(U_stokes)]=0.

#Polarimetry
PI  = np.sqrt(Q_stokes*Q_stokes + U_stokes*U_stokes)
P   = PI/I_stokes*100
PA  = 0.5*arctan2(U_stokes,Q_stokes)*180./np.pi+90
s_P  = np.sqrt(2.)*(I_stokes)**(-0.5)
s_PA = s_P/(P/100.)*180./np.pi

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

fig2.savefig(plots_folder+figname+'_test_Stokes.png',dpi=300)
#os.system('open test_Stokes.png')

### Step 4. Binning and smoothing
#Images can be binned and smoothed to improve SNR. This step can also be done
#using the PolX images.


### Step 5. Roate images to have North up
#Images needs to be reprojected to have North up.
#this procedure implies to rotate the Stokes QU using a rotation matrix


### STEP 6. image to FITS with updated WCS
new_wcs = WCS(naxis=2)
new_wcs.wcs.crpix = [I_stokes.shape[0]/2, I_stokes.shape[1]/2]
new_wcs.wcs.crval = [img[0].header['CRVAL1'], img[0].header['CRVAL2']]
new_wcs.wcs.cunit = ["deg", "deg"]
new_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
new_wcs.wcs.cdelt = [img[0].header['CD1_1']*Dxy, img[0].header['CD1_2']*Dxy]

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
pol.data[SNRp < SNRp_cut] = np.nan

SNRi = stkI.data/np.std(stkI.data[0:10,0:10])
pol.data[SNRi < SNRi_cut] = np.nan
print(np.max(SNRi))
fig = plt.figure(figsize=(11,10))
gc = FITSFigure(stkI,figure=fig)
gc.show_contour(np.log10(SNRi),levels=np.linspace(np.log10(SNRi_cut),np.max(np.log10(SNRi)),20),\
                filled=True,cmap='magma')
gc.show_vectors(pol,pang,scale=scalevec,step=step_vec,color='white',linewidth=1.0)

fig.savefig(plots_folder+figname,dpi=300)
#os.system('open '+figname)
