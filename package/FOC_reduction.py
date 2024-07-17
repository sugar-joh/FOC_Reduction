#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Main script where are progressively added the steps for the FOC pipeline reduction.
"""

# Project libraries

from copy import deepcopy
import os
from os import system
from os.path import exists as path_exists

from matplotlib.colors import LogNorm
import numpy as np

from lib.background import subtract_bkg
import lib.fits as proj_fits        # Functions to handle fits files
import lib.reduction as proj_red    # Functions used in reduction pipeline
import lib.plots as proj_plots      # Functions for plotting data
from lib.utils import sci_not, princ_angle





def main(target=None, proposal_id=None, data_dir=None, infiles=None, output_dir="./data", crop=False, interactive=False):
    # Reduction parameters
    # Deconvolution
    deconvolve = False
    if deconvolve:
        # from lib.deconvolve import from_file_psf
        psf = "gaussian"  # Can be user-defined as well
        # psf = from_file_psf(data_folder+psf_file)
        psf_FWHM = 3.1
        psf_scale = "px"
        psf_shape = None  # (151, 151)
        iterations = 1
        algo = "conjgrad"

    # Initial crop
    display_crop = False

    # Background estimation

    error_sub_type = "freedman-diaconis"  # sqrt, sturges, rice, scott, freedman-diaconis (default) or shape (example (51, 51))
    subtract_error = 1.0

    display_bkg = False

    # Data binning
    pxsize = 0.05
    pxscale = "arcsec"  # pixel, arcsec or full
    rebin_operation = "sum"  # sum or average

    # Alignement
    align_center = "center"  # If None will not align the images

    display_align = False
    display_data = False

    # Transmittance correction
    transmitcorr = True

    # Smoothing
    smoothing_function = "combine"  # gaussian_after, weighted_gaussian_after, gaussian, weighted_gaussian or combine
    smoothing_FWHM = 0.1    # If None, no smoothing is done
    smoothing_scale = "arcsec"  # pixel or arcsec

    # Rotation
    rotate_North = True

    #  Polarization map output
    SNRp_cut = 3.0  # P measurments with SNR>3
    SNRi_cut = 1.0  # I measurments with SNR>30, which implies an uncertainty in P of 4.7%.
    flux_lim = None  # lowest and highest flux displayed on plot, defaults to bkg and maximum in cut if None
    scale_vec = 5
    step_vec = 1  # plot all vectors in the array. if step_vec = 2, then every other vector will be plotted if step_vec = 0 then all vectors are displayed at full length

    # Adaptive binning
    # in order to perfrom optimal binning, there are several steps to follow:
    # 1. Load the data again and preserve the full images
    # 2. Skip the cropping step but use the same error and background estimation
    # 3. Use the same alignment as the routine
    # 4. Skip the rebinning step
    # 5. Calulate the Stokes parameters without smoothing
    optimal_binning = True
    optimize = False
    
    # Pipeline start

    # Step 1:
    #  Get data from fits files and translate to flux in erg/cm²/s/Angstrom.
    outfiles = []
    if data_dir is None:
        if infiles is not None:
            prod = np.array([["/".join(filepath.split('/')[:-1]), filepath.split('/')[-1]] for filepath in infiles], dtype=str)
            obs_dir = "/".join(infiles[0].split("/")[:-1])
            if not path_exists(obs_dir):
                system("mkdir -p {0:s} {1:s}".format(obs_dir, obs_dir.replace("data", "plots")))
            if target is None:
                target = input("Target name:\n>")
        else:
            from lib.query import retrieve_products
            target, products = retrieve_products(target, proposal_id, output_dir=output_dir)
            prod = products.pop()
            for prods in products:
                outfiles.append(main(target=target, infiles=["/".join(pr) for pr in prods], output_dir=output_dir, crop=crop, interactive=interactive))
        data_folder = prod[0][0]
        
        infiles = [p[1] for p in prod]
        data_array, headers = proj_fits.get_obs_data(infiles, data_folder=data_folder, compute_flux=True)
    
    else:
        infiles = [f for f in os.listdir(data_dir) if f.endswith('.fits') and f.startswith('x')]
        data_folder = data_dir
        if target is None:
                target = input("Target name:\n>")
        
        data_array, headers = proj_fits.get_obs_data(infiles, data_folder=data_folder, compute_flux=True)

    try:
        plots_folder = data_folder.replace("data", "plots")
    except ValueError:
        plots_folder = "."
    if not path_exists(plots_folder):
        system("mkdir -p {0:s} ".format(plots_folder))

    figname = "_".join([target, "FOC"])
    figtype = ""
    if (pxsize is not None) and not (pxsize == 1 and pxscale.lower() in ["px", "pixel", "pixels"]):
        if pxscale not in ["full"]:
            figtype = "".join(["b", "{0:.2f}".format(pxsize), pxscale])  # additionnal informations
        else:
            figtype = "full"

    if smoothing_FWHM is not None and smoothing_scale is not None:
        smoothstr = "".join([*[s[0] for s in smoothing_function.split("_")], "{0:.2f}".format(smoothing_FWHM), smoothing_scale])
        figtype = "_".join([figtype, smoothstr] if figtype != "" else [smoothstr])

    if deconvolve:
        figtype = "_".join([figtype, "deconv"] if figtype != "" else ["deconv"])

    if align_center is None:
        figtype = "_".join([figtype, "not_aligned"] if figtype != "" else ["not_aligned"])
    
    if optimal_binning:
        options = {'optimize': optimize, 'optimal_binning': True}
        
        # Step 1: Load the data again and preserve the full images
        _data_array, _headers = deepcopy(data_array), deepcopy(headers) # Preserve full images
        _data_mask = np.ones(_data_array[0].shape, dtype=bool)
        
        # Step 2: Skip the cropping step but use the same error and background estimation (I don't understand why this is wrong)
        data_array, error_array, headers = proj_red.crop_array(data_array, headers, step=5, null_val=0., inside=True,
                                                                display=display_crop, savename=figname, plots_folder=plots_folder)
        data_mask = np.ones(data_array[0].shape, dtype=bool)
        
        background = None
        _, _, _, background, error_bkg = proj_red.get_error(data_array, headers, error_array, data_mask=data_mask, sub_type=error_sub_type, subtract_error=subtract_error, display=display_bkg, savename="_".join([figname, "errors"]), plots_folder=plots_folder, return_background=True)
        
        # _background is the same as background, but for the optimal binning
        _background = None
        _data_array, _error_array, _, = proj_red.get_error(_data_array, _headers, error_array=None, data_mask=_data_mask, sub_type=error_sub_type, subtract_error=False, display=display_bkg, savename="_".join([figname, "errors"]), plots_folder=plots_folder, return_background=False)
        _error_bkg = np.ones_like(_data_array) * error_bkg[:, 0, 0, np.newaxis, np.newaxis]
        _data_array, _error_array, _background, _ = subtract_bkg(_data_array, _error_array, _data_mask, background, _error_bkg)

        # Step 3: Align and rescale images with oversampling. (has to disable croping in align_data function)
        _data_array, _error_array, _headers, _, shifts, error_shifts = proj_red.align_data(_data_array, _headers, error_array=_error_array, background=_background,
                                                                                                upsample_factor=10, ref_center=align_center, return_shifts=True, optimal_binning=True)
        print("Image shifts: {} \nShifts uncertainty: {}".format(shifts, error_shifts))
        _data_mask = np.ones(_data_array[0].shape, dtype=bool)
        
        # Step 4: Compute Stokes I, Q, U
        _background = np.array([np.array(bkg).reshape(1, 1) for bkg in _background])
        _background_error = np.array([np.array(np.sqrt((bkg-_background[np.array([h['filtnam1'] == head['filtnam1'] for h in _headers], dtype=bool)].mean())
                                    ** 2/np.sum([h['filtnam1'] == head['filtnam1'] for h in _headers]))).reshape(1, 1) for bkg, head in zip(_background, _headers)])
        
        _I_stokes, _Q_stokes, _U_stokes, _Stokes_cov, _header_stokes = proj_red.compute_Stokes(_data_array, _error_array, _data_mask, _headers, 
                                                                                FWHM=None, scale=smoothing_scale, smoothing=smoothing_function, transmitcorr=transmitcorr)
        _I_bkg, _Q_bkg, _U_bkg, _S_cov_bkg, _header_bkg = proj_red.compute_Stokes(_background, _background_error, np.array(True).reshape(1, 1), _headers, 
                                                                    FWHM=None, scale=smoothing_scale, smoothing=smoothing_function, transmitcorr=False)
        
        # Step 5: Compute polarimetric parameters (polarization degree and angle).
        _P, _debiased_P, _s_P, _s_P_P, _PA, _s_PA, _s_PA_P = proj_red.compute_pol(_I_stokes, _Q_stokes, _U_stokes, _Stokes_cov, _header_stokes)
        _P_bkg, _debiased_P_bkg, _s_P_bkg, _s_P_P_bkg, _PA_bkg, _s_PA_bkg, _s_PA_P_bkg = proj_red.compute_pol(_I_bkg, _Q_bkg, _U_bkg, _S_cov_bkg, _header_bkg)
        
        # Step 6: Save image to FITS.
        figname = "_".join([figname, figtype]) if figtype != "" else figname
        _Stokes_hdul = proj_fits.save_Stokes(_I_stokes, _Q_stokes, _U_stokes, _Stokes_cov, _P, _debiased_P, _s_P, _s_P_P, _PA, _s_PA, _s_PA_P,
                                    _header_stokes, _data_mask, figname, data_folder=data_folder, return_hdul=True)
        
        # Step 6:
        _data_mask = _Stokes_hdul['data_mask'].data.astype(bool)
        print(
            "F_int({0:.0f} Angs) = ({1} ± {2})e{3} ergs.cm^-2.s^-1.Angs^-1".format(
                _header_stokes["PHOTPLAM"],
                *sci_not(
                    _Stokes_hdul[0].data[_data_mask].sum() * _header_stokes["PHOTFLAM"],
                    np.sqrt(_Stokes_hdul[3].data[0, 0][_data_mask].sum()) * _header_stokes["PHOTFLAM"],
                    2,
                    out=int,
                ),
            )
        )
        print("P_int = {0:.1f} ± {1:.1f} %".format(_header_stokes["p_int"] * 100.0, np.ceil(_header_stokes["sP_int"] * 1000.0) / 10.0))
        print("PA_int = {0:.1f} ± {1:.1f} °".format(princ_angle(_header_stokes["pa_int"]), princ_angle(np.ceil(_header_stokes["sPA_int"] * 10.0) / 10.0)))
        #  Background values
        print(
            "F_bkg({0:.0f} Angs) = ({1} ± {2})e{3} ergs.cm^-2.s^-1.Angs^-1".format(
                _header_stokes["PHOTFLAM"], *sci_not(_I_bkg[0, 0] * _header_stokes["PHOTFLAM"], np.sqrt(_S_cov_bkg[0, 0][0, 0]) * _header_stokes["PHOTFLAM"], 2, out=int)
            )
        )
        print("P_bkg = {0:.1f} ± {1:.1f} %".format(_debiased_P_bkg[0, 0] * 100.0, np.ceil(_s_P_bkg[0, 0] * 1000.0) / 10.0))
        print("PA_bkg = {0:.1f} ± {1:.1f} °".format(princ_angle(_PA_bkg[0, 0]), princ_angle(np.ceil(_s_PA_bkg[0, 0] * 10.0) / 10.0)))
        
        #  Plot polarization map (Background is either total Flux, Polarization degree or Polarization degree error).
        if pxscale.lower() not in ['full', 'integrate'] and not interactive:
            proj_plots.polarization_map(deepcopy(_Stokes_hdul), _data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim,
                                        step_vec=step_vec, vec_scale=scale_vec, savename="_".join([figname]), plots_folder=plots_folder, **options)
            proj_plots.polarization_map(deepcopy(_Stokes_hdul), _data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim, step_vec=step_vec,
                                        vec_scale=scale_vec, savename="_".join([figname, "I"]), plots_folder=plots_folder, display='Intensity', **options)
            proj_plots.polarization_map(deepcopy(_Stokes_hdul), _data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim, step_vec=step_vec,
                                        vec_scale=scale_vec, savename="_".join([figname, "P_flux"]), plots_folder=plots_folder, display='Pol_Flux', **options)
            proj_plots.polarization_map(deepcopy(_Stokes_hdul), _data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim, step_vec=step_vec,
                                        vec_scale=scale_vec, savename="_".join([figname, "P"]), plots_folder=plots_folder, display='Pol_deg', **options)
            proj_plots.polarization_map(deepcopy(_Stokes_hdul), _data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim, step_vec=step_vec,
                                        vec_scale=scale_vec, savename="_".join([figname, "PA"]), plots_folder=plots_folder, display='Pol_ang', **options)
            proj_plots.polarization_map(deepcopy(_Stokes_hdul), _data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim, step_vec=step_vec,
                                        vec_scale=scale_vec, savename="_".join([figname, "I_err"]), plots_folder=plots_folder, display='I_err', **options)
            proj_plots.polarization_map(deepcopy(_Stokes_hdul), _data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim, step_vec=step_vec,
                                        vec_scale=scale_vec, savename="_".join([figname, "P_err"]), plots_folder=plots_folder, display='Pol_deg_err', **options)
            proj_plots.polarization_map(deepcopy(_Stokes_hdul), _data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim, step_vec=step_vec,
                                        vec_scale=scale_vec, savename="_".join([figname, "SNRi"]), plots_folder=plots_folder, display='SNRi', **options)
            proj_plots.polarization_map(deepcopy(_Stokes_hdul), _data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim, step_vec=step_vec,
                                        vec_scale=scale_vec, savename="_".join([figname, "SNRp"]), plots_folder=plots_folder, display='SNRp', **options)
        elif not interactive:
            proj_plots.polarization_map(deepcopy(_Stokes_hdul), _data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut,
                                        savename=figname, plots_folder=plots_folder, display='integrate', **options)
        elif pxscale.lower() not in ['full', 'integrate']:
            proj_plots.pol_map(_Stokes_hdul, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim)
    
    else:
        options = {'optimize': optimize, 'optimal_binning': False}
        #  Crop data to remove outside blank margins.
        data_array, error_array, headers = proj_red.crop_array(data_array, headers, step=5, null_val=0., inside=True,
                                                                display=display_crop, savename=figname, plots_folder=plots_folder)
        data_mask = np.ones(data_array[0].shape, dtype=bool)

        #  Deconvolve data using Richardson-Lucy iterative algorithm with a gaussian PSF of given FWHM.
        if deconvolve:
            data_array = proj_red.deconvolve_array(data_array, headers, psf=psf, FWHM=psf_FWHM, scale=psf_scale, shape=psf_shape, iterations=iterations, algo=algo)

        #  Estimate error from data background, estimated from sub-image of desired sub_shape.
        background = None
        data_array, error_array, headers, background, error_bkg = proj_red.get_error(data_array, headers, error_array, data_mask=data_mask, sub_type=error_sub_type, subtract_error=subtract_error, display=display_bkg, savename="_".join([figname, "errors"]), plots_folder=plots_folder, return_background=True)

        #  Align and rescale images with oversampling.
        data_array, error_array, headers, data_mask, shifts, error_shifts = proj_red.align_data(
            data_array, headers, error_array=error_array, background=background, upsample_factor=10, ref_center=align_center, return_shifts=True)
            
        if display_align:
            print("Image shifts: {} \nShifts uncertainty: {}".format(shifts, error_shifts))
            proj_plots.plot_obs(data_array, headers, savename="_".join([figname, str(align_center)]), plots_folder=plots_folder, norm=LogNorm(
                vmin=data_array[data_array > 0.].min()*headers[0]['photflam'], vmax=data_array[data_array > 0.].max()*headers[0]['photflam']))

        #  Rebin data to desired pixel size.
        if (pxsize is not None) and not (pxsize == 1 and pxscale.lower() in ["px", "pixel", "pixels"]):
            data_array, error_array, headers, Dxy, data_mask = proj_red.rebin_array(
                data_array, error_array, headers, pxsize=pxsize, scale=pxscale, operation=rebin_operation, data_mask=data_mask)

        # Rotate data to have same orientation
        rotate_data = np.unique([np.round(float(head["ORIENTAT"]), 3) for head in headers]).size != 1
        if rotate_data:
            ang = np.mean([head["ORIENTAT"] for head in headers])
            for head in headers:
                head["ORIENTAT"] -= ang
            data_array, error_array, data_mask, headers = proj_red.rotate_data(data_array, error_array, data_mask, headers)
            if display_data:
                proj_plots.plot_obs(
                    data_array,
                    headers,
                    savename="_".join([figname, "rotate_data"]),
                    plots_folder=plots_folder,
                    norm=LogNorm(
                        vmin=data_array[data_array > 0.0].min() * headers[0]["photflam"], vmax=data_array[data_array > 0.0].max() * headers[0]["photflam"]
                    ),
                )

        # Plot array for checking output
        if display_data and pxscale.lower() not in ['full', 'integrate']:
            proj_plots.plot_obs(data_array, headers, savename="_".join([figname, "rebin"]), plots_folder=plots_folder, norm=LogNorm(
                vmin=data_array[data_array > 0.].min()*headers[0]['photflam'], vmax=data_array[data_array > 0.].max()*headers[0]['photflam']))

        background = np.array([np.array(bkg).reshape(1, 1) for bkg in background])
        background_error = np.array([np.array(np.sqrt((bkg-background[np.array([h['filtnam1'] == head['filtnam1'] for h in headers], dtype=bool)].mean())
                                    ** 2/np.sum([h['filtnam1'] == head['filtnam1'] for h in headers]))).reshape(1, 1) for bkg, head in zip(background, headers)])

        # Step 2:
        # Compute Stokes I, Q, U with smoothed polarized images
        # SMOOTHING DISCUSSION :
        # FWHM of FOC have been estimated at about 0.03" across 1500-5000 Angstrom band, which is about 2 detector pixels wide
        # see Jedrzejewski, R.; Nota, A.; Hack, W. J., A Comparison Between FOC and WFPC2
        # Bibcode : 1995chst.conf...10J
        I_stokes, Q_stokes, U_stokes, Stokes_cov, header_stokes = proj_red.compute_Stokes(
            data_array, error_array, data_mask, headers, FWHM=smoothing_FWHM, scale=smoothing_scale, smoothing=smoothing_function, transmitcorr=transmitcorr)
        I_bkg, Q_bkg, U_bkg, S_cov_bkg, header_bkg = proj_red.compute_Stokes(background, background_error, np.array(True).reshape(
            1, 1), headers, FWHM=None, scale=smoothing_scale, smoothing=smoothing_function, transmitcorr=False)

        # Step 3:
        # Rotate images to have North up
        if rotate_North:
            I_stokes, Q_stokes, U_stokes, Stokes_cov, data_mask, header_stokes = proj_red.rotate_Stokes(
                I_stokes, Q_stokes, U_stokes, Stokes_cov, data_mask, header_stokes, SNRi_cut=None)
            I_bkg, Q_bkg, U_bkg, S_cov_bkg, data_mask_bkg, header_bkg = proj_red.rotate_Stokes(I_bkg, Q_bkg, U_bkg, S_cov_bkg, np.array(True).reshape(1, 1), header_bkg, SNRi_cut=None)

        # Compute polarimetric parameters (polarization degree and angle).
        P, debiased_P, s_P, s_P_P, PA, s_PA, s_PA_P = proj_red.compute_pol(I_stokes, Q_stokes, U_stokes, Stokes_cov, header_stokes)
        P_bkg, debiased_P_bkg, s_P_bkg, s_P_P_bkg, PA_bkg, s_PA_bkg, s_PA_P_bkg = proj_red.compute_pol(I_bkg, Q_bkg, U_bkg, S_cov_bkg, header_bkg)

        # Step 4:
        # Save image to FITS.
        figname = "_".join([figname, figtype]) if figtype != "" else figname
        Stokes_hdul = proj_fits.save_Stokes(I_stokes, Q_stokes, U_stokes, Stokes_cov, P, debiased_P, s_P, s_P_P, PA, s_PA, s_PA_P,
                                            header_stokes, data_mask, figname, data_folder=data_folder, return_hdul=True)
        outfiles.append("/".join([data_folder, Stokes_hdul[0].header["FILENAME"] + ".fits"]))

        # Step 5:
        # crop to desired region of interest (roi)
        if crop:
            figname += "_crop"
            stokescrop = proj_plots.crop_Stokes(deepcopy(Stokes_hdul), norm=LogNorm())
            stokescrop.crop()
            stokescrop.write_to("/".join([data_folder, figname+".fits"]))
            Stokes_hdul, header_stokes = stokescrop.hdul_crop, [dataset.header for dataset in stokescrop.hdul_crop]
            outfiles.append("/".join([data_folder, Stokes_hdul[0].header["FILENAME"] + ".fits"]))

        data_mask = Stokes_hdul['data_mask'].data.astype(bool)
        print(
            "F_int({0:.0f} Angs) = ({1} ± {2})e{3} ergs.cm^-2.s^-1.Angs^-1".format(
                header_stokes["PHOTPLAM"],
                *sci_not(
                    Stokes_hdul[0].data[data_mask].sum() * header_stokes["PHOTFLAM"],
                    np.sqrt(Stokes_hdul[3].data[0, 0][data_mask].sum()) * header_stokes["PHOTFLAM"],
                    2,
                    out=int,
                ),
            )
        )
        print("P_int = {0:.1f} ± {1:.1f} %".format(header_stokes["p_int"] * 100.0, np.ceil(header_stokes["sP_int"] * 1000.0) / 10.0))
        print("PA_int = {0:.1f} ± {1:.1f} °".format(princ_angle(header_stokes["pa_int"]), princ_angle(np.ceil(header_stokes["sPA_int"] * 10.0) / 10.0)))
        #  Background values
        print(
            "F_bkg({0:.0f} Angs) = ({1} ± {2})e{3} ergs.cm^-2.s^-1.Angs^-1".format(
                header_stokes["PHOTPLAM"], *sci_not(I_bkg[0, 0] * header_stokes["PHOTPLAM"], np.sqrt(S_cov_bkg[0, 0][0, 0]) * header_stokes["PHOTPLAM"], 2, out=int)
            )
        )
        print("P_bkg = {0:.1f} ± {1:.1f} %".format(debiased_P_bkg[0, 0] * 100.0, np.ceil(s_P_bkg[0, 0] * 1000.0) / 10.0))
        print("PA_bkg = {0:.1f} ± {1:.1f} °".format(princ_angle(PA_bkg[0, 0]), princ_angle(np.ceil(s_PA_bkg[0, 0] * 10.0) / 10.0)))
        #  Plot polarization map (Background is either total Flux, Polarization degree or Polarization degree error).
        if pxscale.lower() not in ['full', 'integrate'] and not interactive:
            proj_plots.polarization_map(deepcopy(Stokes_hdul), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim,
                                        step_vec=step_vec, scale_vec=scale_vec, savename="_".join([figname]), plots_folder=plots_folder, **options)
            proj_plots.polarization_map(deepcopy(Stokes_hdul), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim, step_vec=step_vec,
                                        scale_vec=scale_vec, savename="_".join([figname, "I"]), plots_folder=plots_folder, display='Intensity', **options)
            proj_plots.polarization_map(deepcopy(Stokes_hdul), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim, step_vec=step_vec,
                                        scale_vece=scale_vec, savename="_".join([figname, "P_flux"]), plots_folder=plots_folder, display='Pol_Flux', **options)
            proj_plots.polarization_map(deepcopy(Stokes_hdul), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim, step_vec=step_vec,
                                        scale_vec=scale_vec, savename="_".join([figname, "P"]), plots_folder=plots_folder, display='Pol_deg', **options)
            proj_plots.polarization_map(deepcopy(Stokes_hdul), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim, step_vec=step_vec,
                                        scale_vec=scale_vec, savename="_".join([figname, "PA"]), plots_folder=plots_folder, display='Pol_ang', **options)
            proj_plots.polarization_map(deepcopy(Stokes_hdul), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim, step_vec=step_vec,
                                        scale_vec=scale_vec, savename="_".join([figname, "I_err"]), plots_folder=plots_folder, display='I_err', **options)
            proj_plots.polarization_map(deepcopy(Stokes_hdul), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim, step_vec=step_vec,
                                        scale_vec=scale_vec, savename="_".join([figname, "P_err"]), plots_folder=plots_folder, display='Pol_deg_err', **options)
            proj_plots.polarization_map(deepcopy(Stokes_hdul), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim, step_vec=step_vec,
                                        scale_vec=scale_vec, savename="_".join([figname, "SNRi"]), plots_folder=plots_folder, display='SNRi', **options)
            proj_plots.polarization_map(deepcopy(Stokes_hdul), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim, step_vec=step_vec,
                                        scale_vec=scale_vec, savename="_".join([figname, "SNRp"]), plots_folder=plots_folder, display='SNRp', **options)
        elif not interactive:
            proj_plots.polarization_map(deepcopy(Stokes_hdul), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut,
                                        savename=figname, plots_folder=plots_folder, display='integrate', **options)
        elif pxscale.lower() not in ['full', 'integrate']:
            proj_plots.pol_map(Stokes_hdul, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim)


    return outfiles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Query MAST for target products')
    parser.add_argument('-t', '--target', metavar='targetname', required=False, help='the name of the target', type=str, default=None)
    parser.add_argument('-p', '--proposal_id', metavar='proposal_id', required=False, help='the proposal id of the data products', type=int, default=None)
    parser.add_argument('-d', '--data_dir', metavar='directory_path', required=False, help='directory path to the data products', type=str, default=None)
    parser.add_argument('-f', '--files', metavar='path', required=False, nargs='*', help='the full or relative path to the data products', default=None)
    parser.add_argument('-o', '--output_dir', metavar='directory_path', required=False,
                        help='output directory path for the data products', type=str, default="./data")
    parser.add_argument('-c', '--crop', action='store_true', required=False, help='whether to crop the analysis region')
    parser.add_argument('-i', '--interactive', action='store_true', required=False, help='whether to output to the interactive analysis tool')
    
    args = parser.parse_args()
    exitcode = main(target=args.target, proposal_id=args.proposal_id, data_dir=args.data_dir, infiles=args.files,
                    output_dir=args.output_dir, crop=args.crop, interactive=args.interactive)
    print("Finished with ExitCode: ", exitcode)