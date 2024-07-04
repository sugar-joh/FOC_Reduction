#!/usr/bin/python
# -*- coding:utf-8 -*-
# Project libraries

import numpy as np


def same_reduction(infiles):
    """
    Test if infiles are pipeline productions with same parameters.
    """
    from astropy.io.fits import open as fits_open

    params = {"IQU": [], "TARGNAME": [], "BKG_SUB": [], "SAMPLING": [], "SMOOTHING": []}
    for file in infiles:
        with fits_open(file) as f:
            # test for presence of I, Q, U images
            datatype = []
            for hdu in f:
                try:
                    datatype.append(hdu.header["datatype"])
                except KeyError:
                    pass
            test_IQU = True
            for look in ["I_stokes", "Q_stokes", "U_stokes", "IQU_cov_matrix"]:
                test_IQU *= look in datatype
            params["IQU"].append(test_IQU)
            # look for information on reduction procedure
            for key in ["TARGNAME", "BKG_SUB", "SAMPLING", "SMOOTHING"]:
                try:
                    params[key].append(f[0].header[key])
                except KeyError:
                    params[key].append("null")
    result = np.all(params["IQU"])
    for key in ["TARGNAME", "BKG_SUB", "SAMPLING", "SMOOTHING"]:
        result *= np.unique(params[key]).size == 1

    return result


def same_obs(infiles, data_folder):
    """
    Group infiles into same observations.
    """

    import astropy.units as u
    from astropy.io.fits import getheader
    from astropy.table import Table
    from astropy.time import Time, TimeDelta

    headers = [getheader("/".join([data_folder,file])) for file in infiles]
    files = {}
    files["PROPOSID"] = np.array([str(head["PROPOSID"]) for head in headers],dtype=str)
    files["ROOTNAME"] = np.array([head["ROOTNAME"].lower()+"_c0f.fits" for head in headers],dtype=str)
    files["EXPSTART"] = np.array([Time(head["EXPSTART"],format='mjd') for head in headers])
    products = Table(files)

    new_infiles = []
    for pid in np.unique(products["PROPOSID"]):
        obs = products[products["PROPOSID"] == pid].copy()
        close_date = np.unique(
            [[np.abs(TimeDelta(obs["EXPSTART"][i].unix - date.unix, format="sec")) < 7.0 * u.d for i in range(len(obs))] for date in obs["EXPSTART"]], axis=0
        )
        if len(close_date) > 1:
            for date in close_date:
                new_infiles.append(list(products["ROOTNAME"][np.any([products["ROOTNAME"] == dataset for dataset in obs["ROOTNAME"][date]], axis=0)]))
        else:
            new_infiles.append(list(products["ROOTNAME"][products["PROPOSID"] == pid]))
    return new_infiles


def combine_Stokes(infiles):
    """
    Combine I, Q, U from different observations of a same object.
    """
    print("not implemented yet")

    return infiles


def main(infiles, target=None, output_dir="./data/"):
    """ """
    if target is None:
        target = input("Target name:\n>")

    if not same_reduction(infiles):
        print("NOT SAME REDUC")
        from FOC_reduction import main as FOC_reduction
        prod = np.array([["/".join(filepath.split("/")[:-1]), filepath.split("/")[-1]] for filepath in infiles], dtype=str)
        data_folder = prod[0][0]
        infiles = [p[1] for p in prod]
        # Reduction parameters
        kwargs = {}
        # Background estimation
        kwargs["error_sub_type"] = "freedman-diaconis"  # sqrt, sturges, rice, scott, freedman-diaconis (default) or shape (example (51, 51))
        kwargs["subtract_error"] = 0.7

        # Data binning
        kwargs["pxsize"] = 0.1
        kwargs["pxscale"] = "arcsec"  # pixel, arcsec or full

        # Smoothing
        kwargs["smoothing_function"] = "combine"  # gaussian_after, weighted_gaussian_after, gaussian, weighted_gaussian or combine
        kwargs["smoothing_FWHM"] = 0.2  # If None, no smoothing is done
        kwargs["smoothing_scale"] = "arcsec"  # pixel or arcsec

        #  Polarization map output
        kwargs["SNRp_cut"] = 3.0  # P measurments with SNR>3
        kwargs["SNRi_cut"] = 1.0  # I measurments with SNR>30, which implies an uncertainty in P of 4.7%.
        kwargs["flux_lim"] = 1e-19, 3e-17  # lowest and highest flux displayed on plot, defaults to bkg and maximum in cut if None
        kwargs["scale_vec"] = 5
        kwargs["step_vec"] = (
            1  # plot all vectors in the array. if step_vec = 2, then every other vector will be plotted if step_vec = 0 then all vectors are displayed at full length
        )
        grouped_infiles = same_obs(infiles, data_folder)
        print(grouped_infiles)

        new_infiles = []
        for i,group in enumerate(grouped_infiles):
            new_infiles.append(FOC_reduction(target=target+"-"+str(i+1), infiles=["/".join([data_folder,file]) for file in group], interactive=True, **kwargs))

        combined_Stokes = combine_Stokes(new_infiles)

    else:
        print("SAME REDUC")
        combined_Stokes = combine_Stokes(infiles)

    return combined_Stokes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine different observations of a single object")
    parser.add_argument("-t", "--target", metavar="targetname", required=False, help="the name of the target", type=str, default=None)
    parser.add_argument("-f", "--files", metavar="path", required=False, nargs="*", help="the full or relative path to the data products", default=None)
    parser.add_argument(
        "-o", "--output_dir", metavar="directory_path", required=False, help="output directory path for the data products", type=str, default="./data"
    )
    args = parser.parse_args()
    exitcode = main(target=args.target, infiles=args.files, output_dir=args.output_dir)
    print("Written to: ", exitcode)
