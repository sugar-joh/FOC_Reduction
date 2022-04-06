#!/usr/bin/python3
from getopt import getopt, error as get_error
from sys import argv

arglist = argv[1:]
options = "hf:p:i:o:"
long_options = ["help","fits=","snrp=","snri=","output="]

fits_path = None
SNRp_cut, SNRi_cut = 3, 30
out_txt = None

try:
    arg, val = getopt(arglist, options, long_options)

    for curr_arg, curr_val in arg:
        if curr_arg in ("-h", "--help"):
            print("python3 analysis.py -f <path_to_reduced_fits> -p <SNRp_cut> -i <SNRi_cut> -o <path_to_output_txt>")
        elif curr_arg in ("-f", "--fits"):
            fits_path = str(curr_val)
        elif curr_arg in ("-p", "--snrp"):
            SNRp_cut = int(curr_val)
        elif curr_arg in ("-i", "--snri"):
            SNRi_cut = int(curr_val)
        elif curr_arg in ("-o", "--output"):
            out_txt = str(curr_val)
except get_error as err:
    print(str(err))

if not fits_path is None:
    from astropy.io import fits
    from lib.plots import pol_map

    Stokes_UV = fits.open(fits_path)
    p = pol_map(Stokes_UV, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut)

    if not out_txt is None:
        import numpy as np

        conv = p.Stokes[0].header['photflam']
        I = p.Stokes[0].data*conv
        Q = p.Stokes[1].data*conv
        U = p.Stokes[2].data*conv
        P = np.zeros(I.shape)
        P[p.cut] = p.Stokes[5].data[p.cut]
        PA = np.zeros(I.shape)
        PA[p.cut] = p.Stokes[8].data[p.cut]

        shape = np.array(I.shape)
        center = (shape/2).astype(int)
        cdelt_arcsec = p.wcs.wcs.cdelt*3600
        xx, yy = np.indices(shape)
        x, y = (xx-center[0])*cdelt_arcsec[0], (yy-center[1])*cdelt_arcsec[1]

        data_list = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                data_list.append([x[i,j], y[i,j], I[i,j], Q[i,j], U[i,j], P[i,j], PA[i,j]])
        data = np.array(data_list)
        np.savetxt(out_txt,data)
else:
    print("python3 analysis.py -f <path_to_reduced_fits> -p <SNRp_cut> -i <SNRi_cut> -o <path_to_output_txt>")
