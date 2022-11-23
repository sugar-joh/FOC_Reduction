#!/usr/bin/env python
from lib.reduction import align_data, crop_array, princ_angle
from lib.deconvolve import zeropad
from matplotlib.colors import LogNorm
from os.path import join as path_join
from os import walk as path_walk
from astropy.io import fits
from astropy.wcs import WCS
from re import compile as regcompile, IGNORECASE
from scipy.ndimage import shift
import numpy as np
import matplotlib.pyplot as plt

root_dir = path_join('/home/t.barnouin/Thesis/HST')
root_dir_K = path_join(root_dir,'Kishimoto','output')
root_dir_S = path_join(root_dir,'FOC_Reduction','output')
root_dir_data_S = path_join(root_dir,'FOC_Reduction','data','NGC1068_x274020')

data_K = {}
data_S = {}
for d,i in zip(['I','Q','U','P','PA','sI','sQ','sU','sP','sPA'],[0,1,2,5,8,(3,0,0),(3,1,1),(3,2,2),6,9]):
    data_K[d] = np.loadtxt(path_join(root_dir_K,d+'.txt'))
    with fits.open(path_join(root_dir_data_S,'NGC1068_K_FOC_bin10px.fits')) as f:
        if not type(i) is int:
            data_S[d] = np.sqrt(f[i[0]].data[i[1],i[2]])
        else:
            data_S[d] = f[i].data
    if i==0:
        header = f[i].header
wcs = WCS(header)
convert_flux = header['photflam']

#zeropad data to get same size of array
shape = data_S['I'].shape
for d in data_K:
    data_K[d] = zeropad(data_K[d],shape)

#shift array to get same information in same pixel
data_arr, error_ar, heads, data_msk, shifts, shifts_err = align_data(np.array([data_S['I'],data_K['I']]), [header, header], upsample_factor=10., return_shifts=True)
for d in data_K:
    data_K[d] = shift(data_K[d],shifts[1],order=1,cval=0.)

#compute pol components from shifted array
for d in [data_S, data_K]:
    for i in d:
        d[i][np.isnan(d[i])] = 0.
    d['P'] = np.where(np.logical_and(np.isfinite(d['I']),d['I']>0.),np.sqrt(d['Q']**2+d['U']**2)/d['I'],0.)
    d['sP'] = np.where(np.logical_and(np.isfinite(d['I']),d['I']>0.),np.sqrt((d['Q']**2*d['sQ']**2+d['U']**2*d['sU']**2)/(d['Q']**2+d['U']**2)+((d['Q']/d['I'])**2+(d['U']/d['I'])**2)*d['sI']**2)/d['I'],0.)
    d['PA'] = princ_angle((90./np.pi)*np.arctan2(d['U'],d['Q'])+180.)
    d['SNRp'] = np.zeros(d['P'].shape)
    d['SNRp'][d['sP']>0.] = d['P'][d['sP']>0.]/d['sP'][d['sP']>0.]
    d['SNRi'] = np.zeros(d['I'].shape)
    d['SNRi'][d['sI']>0.] = d['I'][d['sI']>0.]/d['sI'][d['sI']>0.]
    d['mask'] = np.logical_and(d['SNRi']>30,d['SNRp']>5)
data_S['mask'], data_K['mask'] = np.logical_and(data_S['mask'],data_K['mask']), np.logical_and(data_S['mask'],data_K['mask'])

for d in [data_S, data_K]:
    d['X'], d['Y'] = np.meshgrid(np.arange(d['I'].shape[1]), np.arange(d['I'].shape[0]))
    d['xy_U'], d['xy_V'] = np.where(d['mask'],d['P']*np.cos(np.pi/2.+d['PA']*np.pi/180.), np.nan), np.where(d['mask'],d['P']*np.sin(np.pi/2.+d['PA']*np.pi/180.), np.nan)

#display both polarization maps to check consistencfig = plt.figure()
plt.rcParams.update({'font.size': 20})
fig = plt.figure()
ax = fig.add_subplot(111, projection=wcs)
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.12, 0.01, 0.75])

im0 = ax.imshow(data_S['I']*convert_flux,norm=LogNorm(data_S['I'][data_S['I']>0].min()*convert_flux,data_S['I'][data_S['I']>0].max()*convert_flux),origin='lower',cmap='gray',label=r"$I_{STOKES}$ through this pipeline")
#im0 = ax.imshow(data_K['P']*100.,vmin=0.,vmax=100.,origin='lower',cmap='inferno',label=r"$P$ through Kishimoto's pipeline")
#im0 = ax.imshow(data_S['P']*100.,vmin=0.,vmax=100.,origin='lower',cmap='inferno',label=r"$P$ through this pipeline")
#im0 = ax.imshow(data_K['PA'],vmin=0.,vmax=360.,origin='lower',cmap='inferno',label=r"$\theta_P$ through Kishimoto's pipeline")
#im0 = ax.imshow(data_S['PA'],vmin=0.,vmax=360.,origin='lower',cmap='inferno',label=r"$\theta_P$ through this pipeline")
quiv0 = ax.quiver(data_S['X'],data_S['Y'],data_S['xy_U'],data_S['xy_V'],units='xy',angles='uv',scale=0.5,scale_units='xy',pivot='mid',headwidth=0.,headlength=0.,headaxislength=0.,width=0.1,color='b',alpha=0.75, label="PA through this pipeline")
quiv1 = ax.quiver(data_K['X'],data_K['Y'],data_K['xy_U'],data_K['xy_V'],units='xy',angles='uv',scale=0.5,scale_units='xy',pivot='mid',headwidth=0.,headlength=0.,headaxislength=0.,width=0.1,color='r',alpha=0.75, label="PA through Kishimoto's pipeline")

ax.set_title(r"$SNR_P \geq 5 \; & \; SNR_I \geq 30$")
#ax.coords.grid(True, color='white', ls='dotted', alpha=0.5)
ax.coords[0].set_axislabel('Right Ascension (J2000)')
ax.coords[0].set_axislabel_position('b')
ax.coords[0].set_ticklabel_position('b')
ax.coords[1].set_axislabel('Declination (J2000)')
ax.coords[1].set_axislabel_position('l')
ax.coords[1].set_ticklabel_position('l')
#ax.axis('equal')

cbar = plt.colorbar(im0, cax=cbar_ax, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")
#cbar = plt.colorbar(im0, cax=cbar_ax, label=r"$P$ [%]")
#cbar = plt.colorbar(im0, cax=cbar_ax, label=r"$\theta_P$ [°]")
plt.rcParams.update({'font.size': 15})
ax.legend(loc='upper right')

#compute integrated polarization parameters on a specific cut
for d in [data_S, data_K]:
    d['I_dil'] = np.sum(d['I'][d['mask']])
    d['sI_dil'] = np.sqrt(np.sum(d['sI'][d['mask']]**2))
    d['Q_dil'] = np.sum(d['Q'][d['mask']])
    d['sQ_dil'] = np.sqrt(np.sum(d['sQ'][d['mask']]**2))
    d['U_dil'] = np.sum(d['U'][d['mask']])
    d['sU_dil'] = np.sqrt(np.sum(d['sU'][d['mask']]**2))

    d['P_dil'] = np.sqrt(d['Q_dil']**2+d['U_dil']**2)/d['I_dil']
    d['sP_dil'] = np.sqrt((d['Q_dil']**2*d['sQ_dil']**2+d['U_dil']**2*d['sU_dil']**2)/(d['Q_dil']**2+d['U_dil']**2)+((d['Q_dil']/d['I_dil'])**2+(d['U_dil']/d['I_dil'])**2)*d['sI_dil']**2)/d['I_dil']
    d['PA_dil'] = princ_angle((90./np.pi)*np.arctan2(d['U_dil'],d['Q_dil']))
    d['sPA_dil'] = princ_angle((90./(np.pi*(d['Q_dil']**2+d['U_dil']**2)))*np.sqrt(d['Q_dil']**2*d['sU_dil']**2+d['U_dil']**2*d['sU_dil']**2))
print('From this pipeline :\n', "P = {0:.2f} ± {1:.2f} %\n".format(data_S['P_dil']*100.,data_S['sP_dil']*100.), "PA = {0:.2f} ± {1:.2f} °".format(data_S['PA_dil'],data_S['sPA_dil']))
print("From Kishimoto's pipeline :\n", "P = {0:.2f} ± {1:.2f} %\n".format(data_K['P_dil']*100.,data_K['sP_dil']*100.), "PA = {0:.2f} ± {1:.2f} °".format(data_K['PA_dil'],data_K['sPA_dil']))

#compare different types of error
print("This pipeline : average sI/I={0:.2f} ; sQ/Q={1:.2f} ; sU/U={2:.2f} ; sP/P={3:.2f}".format(np.mean(data_S['sI'][data_S['mask']]/data_S['I'][data_S['mask']]),np.mean(data_S['sQ'][data_S['mask']]/data_S['Q'][data_S['mask']]),np.mean(data_S['sU'][data_S['mask']]/data_S['U'][data_S['mask']]),np.mean(data_S['sP'][data_S['mask']]/data_S['P'][data_S['mask']])))
print("Kishimoto's pipeline : average sI/I={0:.2f} ; sQ/Q={1:.2f} ; sU/U={2:.2f} ; sP/P={3:.2f}".format(np.mean(data_K['sI'][data_S['mask']]/data_K['I'][data_S['mask']]),np.mean(data_K['sQ'][data_S['mask']]/data_K['Q'][data_S['mask']]),np.mean(data_K['sU'][data_S['mask']]/data_K['U'][data_S['mask']]),np.mean(data_K['sP'][data_S['mask']]/data_K['P'][data_S['mask']])))
for d,i in zip(['I','Q','U','P','PA','sI','sQ','sU','sP','sPA'],[0,1,2,5,8,(3,0,0),(3,1,1),(3,2,2),6,9]):
    data_K[d] = np.loadtxt(path_join(root_dir_K,d+'.txt'))
    with fits.open(path_join(root_dir_data_S,'NGC1068_K_FOC_bin10px.fits')) as f:
        if not type(i) is int:
            data_S[d] = np.sqrt(f[i[0]].data[i[1],i[2]])
        else:
            data_S[d] = f[i].data
    if i==0:
        header = f[i].header

#from Kishimoto's pipeline : IQU_dir, IQU_shift, IQU_stat, IQU_trans
#from my pipeline : raw_bg, raw_flat, raw_psf, raw_shift, raw_wav, IQU_dir
# but errors from my pipeline are propagated all along, how to compare then ?

plt.show()