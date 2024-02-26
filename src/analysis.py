#!/usr/bin/python3
from getopt import getopt, error as get_error
from sys import argv

arglist = argv[1:]
options = "hf:p:i:l:"
long_options = ["help", "fits=", "snrp=", "snri=", "lim="]

fits_path = None
SNRp_cut, SNRi_cut = 3, 30
flux_lim = None
out_txt = None

try:
    arg, val = getopt(arglist, options, long_options)

    for curr_arg, curr_val in arg:
        if curr_arg in ("-h", "--help"):
            print("python3 analysis.py -f <path_to_reduced_fits> -p <SNRp_cut> -i <SNRi_cut> -l <flux_lim>")
        elif curr_arg in ("-f", "--fits"):
            fits_path = str(curr_val)
        elif curr_arg in ("-p", "--snrp"):
            SNRp_cut = int(curr_val)
        elif curr_arg in ("-i", "--snri"):
            SNRi_cut = int(curr_val)
        elif curr_arg in ("-l", "--lim"):
            flux_lim = list("".join(curr_val).split(','))
except get_error as err:
    print(str(err))

if fits_path is not None:
    from astropy.io import fits
    from lib.plots import pol_map

    Stokes_UV = fits.open(fits_path)
    p = pol_map(Stokes_UV, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim)

else:
    print("python3 analysis.py -f <path_to_reduced_fits> -p <SNRp_cut> -i <SNRi_cut> -l <flux_lim>")
