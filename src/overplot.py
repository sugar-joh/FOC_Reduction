#!/usr/bin/python3
from os import system as command
from astropy.io import fits
import numpy as np
from copy import deepcopy
from lib.plots import overplot_radio, overplot_pol, align_pol
from matplotlib.colors import LogNorm

#Stokes_UV = fits.open("../data/IC5063_x3nl030/IC5063_FOC_combine_FWHM020.fits")
#Stokes_18GHz = fits.open("../data/IC5063_x3nl030/radio/IC5063.18GHz.fits")
#Stokes_24GHz = fits.open("../data/IC5063_x3nl030/radio/IC5063.24GHz.fits")
#Stokes_103GHz = fits.open("../data/IC5063_x3nl030/radio/I5063_103GHz.fits")
#Stokes_229GHz = fits.open("../data/IC5063_x3nl030/radio/I5063_229GHz.fits")
#Stokes_357GHz = fits.open("../data/IC5063_x3nl030/radio/I5063_357GHz.fits")
#Stokes_S2 = fits.open("../data/IC5063_x3nl030/POLARIZATION_COMPARISON/S2_rot_crop.fits")
#Stokes_IR = fits.open("../data/IC5063_x3nl030/IR/u2e65g01t_c0f_rot.fits")

#levelsMorganti = np.array([1.,2.,3.,8.,16.,32.,64.,128.])

##levels18GHz = np.array([0.6, 1.5, 3, 6, 12, 24, 48, 96])/100.*Stokes_18GHz[0].data.max()
#levels18GHz = levelsMorganti*0.28*1e-3
#A = overplot_radio(Stokes_UV, Stokes_18GHz)
#A.plot(levels=levels18GHz, SNRp_cut=3.0, SNRi_cut=60.0, savename='../plots/IC5063_x3nl030/18GHz_overplot_forced.png')

##levels24GHz = np.array([1.,1.5, 3, 6, 12, 24, 48, 96])/100.*Stokes_24GHz[0].data.max()
#levels24GHz = levelsMorganti*0.46*1e-3
#B = overplot_radio(Stokes_UV, Stokes_24GHz)
#B.plot(levels=levels24GHz, SNRp_cut=3.0, SNRi_cut=80.0, savename='../plots/IC5063_x3nl030/24GHz_overplot_forced.png')
#
#levels103GHz = np.linspace(1,99,11)/100.*np.max(deepcopy(Stokes_103GHz[0].data[Stokes_103GHz[0].data > 0.]))
#C = overplot_radio(Stokes_UV, Stokes_103GHz)
#C.plot(levels=levels103GHz, SNRp_cut=3.0, SNRi_cut=80.0, savename='../plots/IC5063_x3nl030/103GHz_overplot_forced.png')
#
#levels229GHz = np.linspace(1,99,11)/100.*np.max(deepcopy(Stokes_229GHz[0].data[Stokes_229GHz[0].data > 0.]))
#D = overplot_radio(Stokes_UV, Stokes_229GHz)
#D.plot(levels=levels229GHz, SNRp_cut=3.0, SNRi_cut=80.0, savename='../plots/IC5063_x3nl030/229GHz_overplot_forced.png')
#
#levels357GHz = np.linspace(1,99,11)/100.*np.max(deepcopy(Stokes_357GHz[0].data[Stokes_357GHz[0].data > 0.]))
#E = overplot_radio(Stokes_UV, Stokes_357GHz)
#E.plot(levels=levels357GHz, SNRp_cut=3.0, SNRi_cut=80.0, savename='../plots/IC5063_x3nl030/357GHz_overplot_forced.png')
#
#F = overplot_pol(Stokes_UV, Stokes_S2)
#F.plot(SNRp_cut=3.0, SNRi_cut=80.0, savename='../plots/IC5063_x3nl030/S2_overplot_forced.png', norm=LogNorm(vmin=5e-20,vmax=5e-18))

#G = overplot_pol(Stokes_UV, Stokes_IR, norm=LogNorm(vmin=1e-17,vmax=5e-15), cmap='inferno_r')
#G.plot(SNRp_cut=3.0, SNRi_cut=60.0, savename='../plots/IC5063_x3nl030/IR_overplot_forced.png', norm=LogNorm(vmin=1e-17,vmax=5e-15), cmap='inferno_r')

data_folder1 = "../data/M87/POS1/"
plots_folder1 = "../plots/M87/POS1/"
basename1 = "M87_015_log_core"
M87_1_95 = fits.open(data_folder1+"M87_POS1_1995_FOC_combine_FWHM010.fits")
M87_1_96 = fits.open(data_folder1+"M87_POS1_1996_FOC_combine_FWHM010.fits")
M87_1_97 = fits.open(data_folder1+"M87_POS1_1997_FOC_combine_FWHM010.fits")
M87_1_98 = fits.open(data_folder1+"M87_POS1_1998_FOC_combine_FWHM010.fits")
M87_1_99 = fits.open(data_folder1+"M87_POS1_1999_FOC_combine_FWHM010.fits")

H = align_pol(np.array([M87_1_95,M87_1_96,M87_1_97,M87_1_98,M87_1_99]), norm=LogNorm())
H.plot(SNRp_cut=3.0, SNRi_cut=30.0, savename=plots_folder1+'animated_loop/'+basename1, norm=LogNorm())
command("convert -delay 50 -loop 0 {0:s}animated_loop/{1:s}*.png {0:s}animated_loop/{1:s}.gif".format(plots_folder1, basename1))

#data_folder3 = "../data/M87/POS3/"
#plots_folder3 = "../plots/M87/POS3/"
#basename3 = "M87_005_log_star"
#M87_3_95 = fits.open(data_folder3+"M87_POS3_1995_FOC_combine_FWHM005.fits")
#M87_3_96 = fits.open(data_folder3+"M87_POS3_1996_FOC_combine_FWHM005.fits")
#M87_3_97 = fits.open(data_folder3+"M87_POS3_1997_FOC_combine_FWHM005.fits")
#M87_3_98 = fits.open(data_folder3+"M87_POS3_1998_FOC_combine_FWHM005.fits")
#M87_3_99 = fits.open(data_folder3+"M87_POS3_1999_FOC_combine_FWHM005.fits")
#
#I = align_pol(np.array([M87_3_95,M87_3_96,M87_3_97,M87_3_98,M87_3_99]), norm=LogNorm())
#I.plot(SNRp_cut=3.0, SNRi_cut=30.0, savename=plots_folder3+'animated_loop/'+basename3, norm=LogNorm())
#command("convert -delay 20 -loop 0 {0:s}animated_loop/{1:s}*.png {0:s}animated_loop/{1:s}.gif".format(plots_folder3, basename3))
