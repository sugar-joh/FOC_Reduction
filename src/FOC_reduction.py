#!/usr/bin/python
#-*- coding:utf-8 -*-
"""
Main script where are progressively added the steps for the FOC pipeline reduction.
"""

#Project libraries
import sys
import numpy as np
import copy
import matplotlib.pyplot as plt
import lib.fits as proj_fits        #Functions to handle fits files
import lib.reduction as proj_red    #Functions used in reduction pipeline
import lib.plots as proj_plots      #Functions for plotting data


def main():
    ##### User inputs
    ## Input and output locations
#    globals()['data_folder'] = "../data/NGC1068_x274020/"
#    infiles = ['x274020at.c0f.fits','x274020bt.c0f.fits','x274020ct.c0f.fits',
#            'x274020dt.c0f.fits','x274020et.c0f.fits','x274020ft.c0f.fits',
#            'x274020gt.c0f.fits','x274020ht.c0f.fits','x274020it.c0f.fits']
#    globals()['plots_folder'] = "../plots/NGC1068_x274020/"

#    globals()['data_folder'] = "../data/NGC1068_x14w010/"
#    infiles = ['x14w0101t_c0f.fits','x14w0102t_c0f.fits','x14w0103t_c0f.fits',
#            'x14w0104t_c0f.fits','x14w0105p_c0f.fits','x14w0106t_c0f.fits']
#    globals()['plots_folder'] = "../plots/NGC1068_x14w010/"

#    globals()['data_folder'] = "../data/3C405_x136060/"
#    infiles = ['x1360601t_c0f.fits','x1360602t_c0f.fits','x1360603t_c0f.fits']
#    globals()['plots_folder'] = "../plots/3C405_x136060/"

#    globals()['data_folder'] = "../data/CygnusA_x43w0/"
#    infiles = ['x43w0101r_c0f.fits', 'x43w0102r_c0f.fits', 'x43w0103r_c0f.fits',
#            'x43w0104r_c0f.fits', 'x43w0105r_c0f.fits', 'x43w0106r_c0f.fits',
#            'x43w0107r_c0f.fits', 'x43w0108r_c0f.fits', 'x43w0109r_c0f.fits']
#    infiles = ['x43w0201r_c0f.fits', 'x43w0202r_c0f.fits', 'x43w0203r_c0f.fits',
#            'x43w0204r_c0f.fits', 'x43w0205r_c0f.fits', 'x43w0206r_c0f.fits']
#    globals()['plots_folder'] = "../plots/CygnusA_x43w0/"

#    globals()['data_folder'] = "../data/3C109_x3mc010/"
#    infiles = ['x3mc0101m_c0f.fits','x3mc0102m_c0f.fits','x3mc0103m_c0f.fits']
#    globals()['plots_folder'] = "../plots/3C109_x3mc010/"

    globals()['data_folder'] = "../data/MKN463_x2rp030/"
    infiles = ['x2rp0201t_c0f.fits', 'x2rp0202t_c0f.fits', 'x2rp0203t_c0f.fits',
            'x2rp0204t_c0f.fits', 'x2rp0205t_c0f.fits', 'x2rp0206t_c0f.fits',
            'x2rp0207t_c0f.fits', 'x2rp0301t_c0f.fits', 'x2rp0302t_c0f.fits',
            'x2rp0303t_c0f.fits', 'x2rp0304t_c0f.fits', 'x2rp0305t_c0f.fits',
            'x2rp0306t_c0f.fits', 'x2rp0307t_c0f.fits']
    globals()['plots_folder'] = "../plots/MKN463_x2rp030/"

#    globals()['data_folder'] = "../data/PG1630+377_x39510/"
#    infiles = ['x3990201m_c0f.fits', 'x3990205m_c0f.fits', 'x3995101r_c0f.fits',
#            'x3995105r_c0f.fits', 'x3995109r_c0f.fits', 'x3995201r_c0f.fits',
#            'x3995205r_c0f.fits', 'x3990202m_c0f.fits', 'x3990206m_c0f.fits',
#            'x3995102r_c0f.fits', 'x3995106r_c0f.fits', 'x399510ar_c0f.fits',
#            'x3995202r_c0f.fits','x3995206r_c0f.fits']
#    globals()['plots_folder'] = "../plots/PG1630+377_x39510/"

#    globals()['data_folder'] = "../data/IC5063_x3nl030/"
#    infiles = ['x3nl0301r_c0f.fits','x3nl0302r_c0f.fits','x3nl0303r_c0f.fits']
#    globals()['plots_folder'] = "../plots/IC5063_x3nl030/"

#    globals()['data_folder'] = "../data/MKN3_x3nl010/"
#    infiles = ['x3nl0101r_c0f.fits','x3nl0102r_c0f.fits','x3nl0103r_c0f.fits']
#    globals()['plots_folder'] = "../plots/MKN3_x3nl010/"

#    globals()['data_folder'] = "../data/MKN3_x3md010/"
#    infiles = ['x3md0101r_c0f.fits', 'x3md0102r_c0f.fits', 'x3md0103r_c0f.fits']
#    infiles = ['x3md0104r_c0f.fits', 'x3md0105r_c0f.fits', 'x3md0106r_c0f.fits']
#    globals()['plots_folder'] = "../plots/MKN3_x3md010/"

#    globals()['data_folder'] = "../data/MKN78_x3nl020/"
#    infiles = ['x3nl0201r_c0f.fits','x3nl0202r_c0f.fits','x3nl0203r_c0f.fits']
#    globals()['plots_folder'] = "../plots/MKN78_x3nl020/"

    ## Reduction parameters
    # Deconvolution
    deconvolve = False
    if deconvolve:
        psf = 'gaussian'  #Can be user-defined as well
        psf_FWHM = 0.10
        psf_scale = 'arcsec'
        psf_shape=(9,9)
        iterations = 10
    # Cropping
    display_crop = False
    # Error estimation
    error_sub_shape = (75,75)
    display_error = False
    # Data binning
    rebin = True
    if rebin:
        pxsize = 0.10
        px_scale = 'arcsec'         #pixel or arcsec
        rebin_operation = 'sum'     #sum or average
    # Alignement
    align_center = 'image'        #If None will align image to image center
    display_data = False
    # Smoothing
    smoothing_function = 'combine'  #gaussian_after, gaussian or combine
    smoothing_FWHM = 0.20           #If None, no smoothing is done
    smoothing_scale = 'arcsec'       #pixel or arcsec
    # Rotation
    rotate_stokes = True           #rotation to North convention can give erroneous results
    rotate_data = False              #rotation to North convention can give erroneous results
    # Polarization map output
    figname = 'MKN463_FOC'         #target/intrument name
    figtype = '_combine_FWHM020_rot'    #additionnal informations
    SNRp_cut = 3    #P measurments with SNR>3
    SNRi_cut = 30   #I measurments with SNR>30, which implies an uncertainty in P of 4.7%.
    step_vec = 1    #plot all vectors in the array. if step_vec = 2, then every other vector will be plotted

    ##### Pipeline start
    ## Step 1:
    # Get data from fits files and translate to flux in erg/cm²/s/Angstrom.
    data_array, headers = proj_fits.get_obs_data(infiles, data_folder=data_folder, compute_flux=True)
    for data in data_array:
        if (data < 0.).any():
            print("ETAPE 1 : ", data)
    # Crop data to remove outside blank margins.
    data_array, error_array = proj_red.crop_array(data_array, headers, step=5, null_val=0., inside=True, display=display_crop, savename=figname, plots_folder=plots_folder)
    for data in data_array:
        if (data < 0.).any():
            print("ETAPE 2 : ", data)
    # Deconvolve data using Richardson-Lucy iterative algorithm with a gaussian PSF of given FWHM.
    if deconvolve:
        data_array = proj_red.deconvolve_array(data_array, headers, psf=psf, FWHM=psf_FWHM, scale=psf_scale, shape=psf_shape, iterations=iterations)
    # Estimate error from data background, estimated from sub-image of desired sub_shape.
    data_array, error_array = proj_red.get_error(data_array, sub_shape=error_sub_shape, display=display_error, headers=headers, savename=figname+"_errors", plots_folder=plots_folder)
    for data in data_array:
        if (data < 0.).any():
            print("ETAPE 3 : ", data)
    # Rebin data to desired pixel size.
    if rebin:
        data_array, error_array, headers, Dxy = proj_red.rebin_array(data_array, error_array, headers, pxsize=pxsize, scale=px_scale, operation=rebin_operation)
    for data in data_array:
        if (data < 0.).any():
            print("ETAPE 4 : ", data)
    #Align and rescale images with oversampling.
    data_array, error_array, data_mask = proj_red.align_data(data_array, headers, error_array, upsample_factor=int(Dxy.min()), ref_center=align_center, return_shifts=False)
    for data in data_array:
        if (data < 0.).any():
            print("ETAPE 5 : ", data)
    # Rotate data to have North up
    ref_header = copy.deepcopy(headers[0])
    if rotate_data:
        data_array, error_array, data_mask, headers = proj_red.rotate_data(data_array, error_array, data_mask, headers, -ref_header['orientat'])
        for data in data_array:
            if (data < 0.).any():
                print("ETAPE 6 : ", data)
    #Plot array for checking output
    if display_data:
        proj_plots.plot_obs(data_array, headers, vmin=data_array.min(), vmax=data_array.max(), savename=figname+"_center_"+align_center, plots_folder=plots_folder)

    ## Step 2:
    # Compute Stokes I, Q, U with smoothed polarized images
    # SMOOTHING DISCUSSION :
    # FWHM of FOC have been estimated at about 0.03" across 1500-5000 Angstrom band, which is about 2 detector pixels wide
    # see Jedrzejewski, R.; Nota, A.; Hack, W. J., A Comparison Between FOC and WFPC2
    # Bibcode : 1995chst.conf...10J
    I_stokes, Q_stokes, U_stokes, Stokes_cov = proj_red.compute_Stokes(data_array, error_array, data_mask, headers, FWHM=smoothing_FWHM, scale=smoothing_scale, smoothing=smoothing_function)

    ## Step 3:
    # Rotate images to have North up
    if rotate_stokes:
        ref_header = copy.deepcopy(headers[0])
        I_stokes, Q_stokes, U_stokes, Stokes_cov, data_mask, headers = proj_red.rotate_Stokes(I_stokes, Q_stokes, U_stokes, Stokes_cov, data_mask, headers, -ref_header['orientat'], SNRi_cut=None)
    # Compute polarimetric parameters (polarization degree and angle).
    P, debiased_P, s_P, s_P_P, PA, s_PA, s_PA_P = proj_red.compute_pol(I_stokes, Q_stokes, U_stokes, Stokes_cov, headers)

    ## Step 4:
    # Save image to FITS.
    Stokes_test = proj_fits.save_Stokes(I_stokes, Q_stokes, U_stokes, Stokes_cov, P, debiased_P, s_P, s_P_P, PA, s_PA, s_PA_P, headers, figname+figtype, data_folder=data_folder, return_hdul=True)

    ## Step 5:
    # Plot polarization map (Background is either total Flux, Polarization degree or Polarization degree error).
    proj_plots.polarization_map(copy.deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, step_vec=step_vec, savename=figname+figtype, plots_folder=plots_folder, display=None)
    proj_plots.polarization_map(copy.deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, step_vec=step_vec, savename=figname+figtype+"_P", plots_folder=plots_folder, display='Pol_deg')
    proj_plots.polarization_map(copy.deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, step_vec=step_vec, savename=figname+figtype+"_P_err", plots_folder=plots_folder, display='Pol_deg_err')
    proj_plots.polarization_map(copy.deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, step_vec=step_vec, savename=figname+figtype+"_SNRi", plots_folder=plots_folder, display='SNRi')
    proj_plots.polarization_map(copy.deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, step_vec=step_vec, savename=figname+figtype+"_SNRp", plots_folder=plots_folder, display='SNRp')

    return 0

if __name__ == "__main__":
    sys.exit(main())
