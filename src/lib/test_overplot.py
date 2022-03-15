#!/usr/bin/python3
from astropy.io import fits
import numpy as np
from plots import overplot_maps

Stokes_UV = fits.open("../../data/IC5063_x3nl030/IC5063_FOC_combine_FWHM020.fits")
Stokes_18GHz = fits.open("../../data/IC5063_x3nl030/radio/IC5063.18GHz.fits")
Stokes_24GHz = fits.open("../../data/IC5063_x3nl030/radio/IC5063.24GHz.fits")

levelsMorganti = np.array([1.,2.,3.,8.,16.,32.,64.,128.])

#levels18GHz = np.array([0.6, 1.5, 3, 6, 12, 24, 48, 96])/100.*Stokes_18GHz[0].data.max()
levels18GHz = levelsMorganti*0.28*1e-3
A = overplot_maps(Stokes_UV, Stokes_18GHz)
A.plot(levels=levels18GHz, SNRp_cut=6.0, SNRi_cut=180.0, savename='../../plots/IC5063_x3nl030/18GHz_overplot.png')

#levels24GHz = np.array([1.,1.5, 3, 6, 12, 24, 48, 96])/100.*Stokes_24GHz[0].data.max()
levels24GHz = levelsMorganti*0.46*1e-3
B = overplot_maps(Stokes_UV, Stokes_24GHz)
B.plot(levels=levels24GHz, SNRp_cut=6.0, SNRi_cut=180.0, savename='../../plots/IC5063_x3nl030/24GHz_overplot.png')
