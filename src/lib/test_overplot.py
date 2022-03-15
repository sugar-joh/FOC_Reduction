from astropy.io import fits
from plots import overplot_maps

Stokes_UV = fits.open("../../data/IC5063_x3nl030/IC5063_FOC_combine_FWHM020.fits")
Stokes_18GHz = fits.open("../../data/IC5063_x3nl030/radio/IC5063.18GHz.fits")
Stokes_24GHz = fits.open("../../data/IC5063_x3nl030/radio/IC5063.24GHz.fits")

A = overplot_maps(Stokes_UV, Stokes_18GHz)
A.plot(5e-3, 5e-2, 8, SNRp_cut=6.0, SNRi_cut=180.0)
A.fig2.savefig('../../plots/IC5063_x3nl030/1.8GHz_overplot.png',bbox_inches='tight',overwrite=True)

B = overplot_maps(Stokes_UV, Stokes_24GHz)
B.plot(1e-3, 3e-2, 8, SNRp_cut=6.0, SNRi_cut=180.0)
B.fig2.savefig('../../plots/IC5063_x3nl030/2.4GHz_overplot.png',bbox_inches='tight',overwrite=True)
