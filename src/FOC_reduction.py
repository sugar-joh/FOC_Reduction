#!/usr/bin/python3
#-*- coding:utf-8 -*-
"""
Main script where are progressively added the steps for the FOC pipeline reduction.
"""

#Project libraries
import sys
import numpy as np
from copy import deepcopy
import lib.fits as proj_fits        #Functions to handle fits files
import lib.reduction as proj_red    #Functions used in reduction pipeline
import lib.plots as proj_plots      #Functions for plotting data
from lib.convex_hull import image_hull
from lib.deconvolve import from_file_psf
import matplotlib.pyplot as plt
from astropy.wcs import WCS

##### User inputs
## Input and output locations
globals()['data_folder'] = "../data/NGC1068_x274020/"
globals()['infiles'] = ['x274020at_c0f.fits','x274020bt_c0f.fits','x274020ct_c0f.fits',
   'x274020dt_c0f.fits','x274020et_c0f.fits','x274020ft_c0f.fits',
   'x274020gt_c0f.fits','x274020ht_c0f.fits','x274020it_c0f.fits']
#psf_file = 'NGC1068_f253m00.fits'
globals()['plots_folder'] = "../plots/NGC1068_x274020/"

#globals()['data_folder'] = "../data/IC5063_x3nl030/"
#globals()['infiles'] = ['x3nl0301r_c0f.fits','x3nl0302r_c0f.fits','x3nl0303r_c0f.fits']
##psf_file = 'IC5063_f502m00.fits'
#globals()['plots_folder'] = "../plots/IC5063_x3nl030/"

#globals()['data_folder'] = "../data/NGC1068_x14w010/"
#globals()['infiles'] = ['x14w0101t_c0f.fits','x14w0102t_c0f.fits','x14w0103t_c0f.fits',
#   'x14w0104t_c0f.fits','x14w0105p_c0f.fits','x14w0106t_c0f.fits']
#globals()['plots_folder'] = "../plots/NGC1068_x14w010/"

#globals()['data_folder'] = "../data/3C405_x136060/"
#globals()['infiles'] = ['x1360601t_c0f.fits','x1360602t_c0f.fits','x1360603t_c0f.fits']
#globals()['plots_folder'] = "../plots/3C405_x136060/"

#globals()['data_folder'] = "../data/CygnusA_x43w0/"
#globals()['infiles'] = ['x43w0101r_c0f.fits', 'x43w0102r_c0f.fits', 'x43w0103r_c0f.fits',
#   'x43w0104r_c0f.fits', 'x43w0105r_c0f.fits', 'x43w0106r_c0f.fits',
#   'x43w0107r_c0f.fits', 'x43w0108r_c0f.fits', 'x43w0109r_c0f.fits'] #F342W
##globals()['infiles'] = ['x43w0201r_c0f.fits', 'x43w0202r_c0f.fits', 'x43w0203r_c0f.fits',
##   'x43w0204r_c0f.fits', 'x43w0205r_c0f.fits', 'x43w0206r_c0f.fits'] #F275W
#globals()['plots_folder'] = "../plots/CygnusA_x43w0/"

#globals()['data_folder'] = "../data/3C109_x3mc010/"
#globals()['infiles'] = ['x3mc0101m_c0f.fits','x3mc0102m_c0f.fits','x3mc0103m_c0f.fits']
#globals()['plots_folder'] = "../plots/3C109_x3mc010/"

#globals()['data_folder'] = "../data/MKN463_x2rp030/"
#globals()['infiles'] = ['x2rp0201t_c0f.fits', 'x2rp0202t_c0f.fits', 'x2rp0203t_c0f.fits',
#   'x2rp0204t_c0f.fits', 'x2rp0205t_c0f.fits', 'x2rp0206t_c0f.fits',
#   'x2rp0207t_c0f.fits', 'x2rp0301t_c0f.fits', 'x2rp0302t_c0f.fits',
#   'x2rp0303t_c0f.fits', 'x2rp0304t_c0f.fits', 'x2rp0305t_c0f.fits',
#   'x2rp0306t_c0f.fits', 'x2rp0307t_c0f.fits']
#globals()['plots_folder'] = "../plots/MKN463_x2rp030/"

#globals()['data_folder'] = "../data/PG1630+377_x39510/"
#globals()['infiles'] = ['x3990201m_c0f.fits', 'x3990205m_c0f.fits', 'x3995101r_c0f.fits',
#   'x3995105r_c0f.fits', 'x3995109r_c0f.fits', 'x3995201r_c0f.fits',
#   'x3995205r_c0f.fits', 'x3990202m_c0f.fits', 'x3990206m_c0f.fits',
#   'x3995102r_c0f.fits', 'x3995106r_c0f.fits', 'x399510ar_c0f.fits',
#   'x3995202r_c0f.fits','x3995206r_c0f.fits']
#globals()['plots_folder'] = "../plots/PG1630+377_x39510/"

#globals()['data_folder'] = "../data/MKN3_x3nl010/"
#globals()['infiles'] = ['x3nl0101r_c0f.fits','x3nl0102r_c0f.fits','x3nl0103r_c0f.fits']
#globals()['plots_folder'] = "../plots/MKN3_x3nl010/"

#globals()['data_folder'] = "../data/MKN3_x3md010/"
#globals()['infiles'] = ['x3md0101r_c0f.fits', 'x3md0102r_c0f.fits', 'x3md0103r_c0f.fits'] #F275W
##globals()['infiles'] = ['x3md0104r_c0f.fits', 'x3md0105r_c0f.fits', 'x3md0106r_c0f.fits'] #F342W
#globals()['plots_folder'] = "../plots/MKN3_x3md010/"

#globals()['data_folder'] = "../data/MKN78_x3nl020/"
#globals()['infiles'] = ['x3nl0201r_c0f.fits','x3nl0202r_c0f.fits','x3nl0203r_c0f.fits']
#globals()['plots_folder'] = "../plots/MKN78_x3nl020/"

#globals()['data_folder'] = "../data/MRK231_x4qr010/"
#globals()['infiles'] = ['x4qr010ar_c0f.fits', 'x4qr010br_c0f.fits', 'x4qr010dr_c0f.fits',
#   'x4qr010er_c0f.fits', 'x4qr010gr_c0f.fits', 'x4qr010hr_c0f.fits',
#   'x4qr010jr_c0f.fits', 'x4qr010kr_c0f.fits', 'x4qr0104r_c0f.fits',
#   'x4qr0105r_c0f.fits', 'x4qr0107r_c0f.fits', 'x4qr0108r_c0f.fits']
#globals()['plots_folder'] = "../plots/MRK231_x4qr010/"

#globals()['data_folder'] = "../data/3C273_x0u20/"
#globals()['infiles'] = ['x0u20101t_c0f.fits','x0u20102t_c0f.fits','x0u20103t_c0f.fits',
#   'x0u20104t_c0f.fits','x0u20105t_c0f.fits','x0u20106t_c0f.fits',
#   'x0u20201t_c0f.fits','x0u20202t_c0f.fits','x0u20203t_c0f.fits',
#   'x0u20204t_c0f.fits','x0u20205t_c0f.fits','x0u20206t_c0f.fits',
#   'x0u20301t_c0f.fits','x0u20302t_c0f.fits','x0u20303t_c0f.fits',
#   'x0u20304t_c0f.fits','x0u20305t_c0f.fits','x0u20306t_c0f.fits']
#globals()['plots_folder'] = "../plots/3C273_x0u20/"

#BEWARE: 5 observations separated by 1 year each (1995, 1996, 1997, 1998, 1999)
#globals()['data_folder'] = "../data/M87/POS1/"
##globals()['infiles'] = ['x2py010ct_c0f.fits','x2py010dt_c0f.fits','x2py010et_c0f.fits','x2py010ft_c0f.fits'] #1995
##globals()['infiles'] = ['x3be010ct_c0f.fits','x3be010dt_c0f.fits','x3be010et_c0f.fits','x3be010ft_c0f.fits'] #1996
##globals()['infiles'] = ['x43r010km_c0f.fits','x43r010mm_c0f.fits','x43r010om_c0f.fits','x43r010rm_c0f.fits'] #1997
##globals()['infiles'] = ['x43r110kr_c0f.fits','x43r110mr_c0f.fits','x43r110or_c0f.fits','x43r110rr_c0f.fits'] #1998
#globals()['infiles'] = ['x43r210kr_c0f.fits','x43r210mr_c0f.fits','x43r210or_c0f.fits','x43r210rr_c0f.fits'] #1999
#globals()['plots_folder'] = "../plots/M87/POS1/"

#BEWARE: 5 observations separated by 1 year each (1995, 1996, 1997, 1998, 1999)
#globals()['data_folder'] = "../data/M87/POS3/"
##globals()['infiles'] = ['x2py030at_c0f.fits','x2py030bt_c0f.fits','x2py030ct_c0f.fits','x2py0309t_c0f.fits'] #1995
##globals()['infiles'] = ['x3be030at_c0f.fits','x3be030bt_c0f.fits','x3be030ct_c0f.fits','x3be0309t_c0f.fits'] #1996
##globals()['infiles'] = ['x43r030em_c0f.fits','x43r030gm_c0f.fits','x43r030im_c0f.fits','x43r030lm_c0f.fits'] #1997
##globals()['infiles'] = ['x43r130er_c0f.fits','x43r130fr_c0f.fits','x43r130ir_c0f.fits','x43r130lr_c0f.fits'] #1998
#globals()['infiles'] = ['x43r230er_c0f.fits','x43r230fr_c0f.fits','x43r230ir_c0f.fits','x43r230lr_c0f.fits'] #1999
#globals()['plots_folder'] = "../plots/M87/POS3/"


def main():
    ## Reduction parameters
    # Deconvolution
    deconvolve = False
    if deconvolve:
        psf = 'gaussian'  #Can be user-defined as well
        #psf = from_file_psf(data_folder+psf_file)
        psf_FWHM = 0.15
        psf_scale = 'arcsec'
        psf_shape=(25,25)
        iterations = 5
        algo="richardson"
    # Initial crop
    display_crop = False
    # Error estimation
    error_sub_type = 'freedman-diaconis'   #sqrt, sturges, rice, scott, freedman-diaconis (default) or shape (example (15,15))
    subtract_error = True
    display_error = False
    # Data binning
    rebin = True
    pxsize = 0.10
    px_scale = 'arcsec'         #pixel, arcsec or full
    rebin_operation = 'sum'     #sum or average
    # Alignement
    align_center = 'image'          #If None will align image to image center
    display_data = False
    # Smoothing
    smoothing_function = 'combine'  #gaussian_after, weighted_gaussian_after, gaussian, weighted_gaussian or combine
    smoothing_FWHM = 0.20           #If None, no smoothing is done
    smoothing_scale = 'arcsec'      #pixel or arcsec
    # Rotation
    rotate_stokes = True
    rotate_data = False             #rotation to North convention can give erroneous results
    # Final crop
    crop = False                    #Crop to desired ROI
    final_display = True
    # Polarization map output
    figname = 'NGC1068_FOC'         #target/intrument name
    figtype = '_c_020'    #additionnal informations
    SNRp_cut = 5.    #P measurments with SNR>3
    SNRi_cut = 50.   #I measurments with SNR>30, which implies an uncertainty in P of 4.7%.
    step_vec = 1    #plot all vectors in the array. if step_vec = 2, then every other vector will be plotted
                    # if step_vec = 0 then all vectors are displayed at full length

    ##### Pipeline start
    ## Step 1:
    # Get data from fits files and translate to flux in erg/cm²/s/Angstrom.
    data_array, headers = proj_fits.get_obs_data(infiles, data_folder=data_folder, compute_flux=True)

    # Crop data to remove outside blank margins.
    data_array, error_array, headers = proj_red.crop_array(data_array, headers, step=5, null_val=0., inside=True, display=display_crop, savename=figname, plots_folder=plots_folder)

    # Deconvolve data using Richardson-Lucy iterative algorithm with a gaussian PSF of given FWHM.
    if deconvolve:
        data_array = proj_red.deconvolve_array(data_array, headers, psf=psf, FWHM=psf_FWHM, scale=psf_scale, shape=psf_shape, iterations=iterations, algo=algo)

    # Estimate error from data background, estimated from sub-image of desired sub_shape.
    background = None
    data_array, error_array, headers, background = proj_red.get_error(data_array, headers, error_array, sub_type=error_sub_type, subtract_error=subtract_error, display=display_error, savename=figname+"_errors", plots_folder=plots_folder, return_background=True)

    # Align and rescale images with oversampling.
    data_array, error_array, headers, data_mask = proj_red.align_data(data_array, headers, error_array=error_array, background=background, upsample_factor=10, ref_center=align_center, return_shifts=False)

    # Rebin data to desired pixel size.
    if rebin:
        data_array, error_array, headers, Dxy, data_mask = proj_red.rebin_array(data_array, error_array, headers, pxsize=pxsize, scale=px_scale, operation=rebin_operation, data_mask=data_mask)

    # Rotate data to have North up
    if rotate_data:
        data_mask = np.ones(data_array.shape[1:]).astype(bool)
        alpha = headers[0]['orientat']
        data_array, error_array, data_mask, headers = proj_red.rotate_data(data_array, error_array, data_mask, headers, -alpha)

    #Plot array for checking output
    if display_data and px_scale.lower() not in ['full','integrate']:
        vertex = image_hull(data_mask,step=5,null_val=0.,inside=True)
        shape = np.array([vertex[1]-vertex[0],vertex[3]-vertex[2]])
        rectangle = [vertex[2], vertex[0], shape[1], shape[0], 0., 'g']

        proj_plots.plot_obs(data_array, headers, vmin=data_array.min(), vmax=data_array.max(), rectangle =[rectangle,]*data_array.shape[0], savename=figname+"_center_"+align_center, plots_folder=plots_folder)

    ## Step 2:
    # Compute Stokes I, Q, U with smoothed polarized images
    # SMOOTHING DISCUSSION :
    # FWHM of FOC have been estimated at about 0.03" across 1500-5000 Angstrom band, which is about 2 detector pixels wide
    # see Jedrzejewski, R.; Nota, A.; Hack, W. J., A Comparison Between FOC and WFPC2
    # Bibcode : 1995chst.conf...10J
    I_stokes, Q_stokes, U_stokes, Stokes_cov = proj_red.compute_Stokes(data_array, error_array, data_mask, headers, FWHM=smoothing_FWHM, scale=smoothing_scale, smoothing=smoothing_function,transmitcorr=True)
    I_bkg, Q_bkg, U_bkg, S_cov_bkg = proj_red.compute_Stokes(background, background_error, np.array(True).reshape(1,1), headers, FWHM=None, scale=smoothing_scale, smoothing=smoothing_function,transmitcorr=True)

    ## Step 3:
    # Rotate images to have North up
    if rotate_stokes:
        I_stokes, Q_stokes, U_stokes, Stokes_cov, data_mask, headers = proj_red.rotate_Stokes(I_stokes, Q_stokes, U_stokes, Stokes_cov, data_mask, headers, SNRi_cut=None)
        I_bkg, Q_bkg, U_bkg, S_cov_bkg, _, _ = proj_red.rotate_Stokes(I_bkg, Q_bkg, U_bkg, S_cov_bkg, np.array(True).reshape(1,1), headers, SNRi_cut=None)

    # Compute polarimetric parameters (polarization degree and angle).
    P, debiased_P, s_P, s_P_P, PA, s_PA, s_PA_P = proj_red.compute_pol(I_stokes, Q_stokes, U_stokes, Stokes_cov, headers)
    P_bkg, debiased_P_bkg, s_P_bkg, s_P_P_bkg, PA_bkg, s_PA_bkg, s_PA_P_bkg = proj_red.compute_pol(I_bkg, Q_bkg, U_bkg, S_cov_bkg, headers)

    ## Step 4:
    # Save image to FITS.
    Stokes_test = proj_fits.save_Stokes(I_stokes, Q_stokes, U_stokes, Stokes_cov, P, debiased_P, s_P, s_P_P, PA, s_PA, s_PA_P, headers, data_mask, figname+figtype, data_folder=data_folder, return_hdul=True)
    data_mask = Stokes_test[-1].data.astype(bool)

    ## Step 5:
    # crop to desired region of interest (roi)
    if crop:
        figtype += "_crop"
        stokescrop = proj_plots.crop_Stokes(deepcopy(Stokes_test))
        stokescrop.crop()
        stokescrop.writeto(data_folder+figname+figtype+".fits")
        Stokes_test, data_mask = stokescrop.hdul_crop, stokescrop.data_mask

    print("F_int({0:.0f} Angs) = ({1} ± {2})e{3} ergs.cm^-2.s^-1.Angs^-1".format(headers[0]['photplam'],*proj_plots.sci_not(Stokes_test[0].data[data_mask].sum()*headers[0]['photflam'],np.sqrt(Stokes_test[3].data[0,0][data_mask].sum())*headers[0]['photflam'],2,out=int)))
    print("P_int = {0:.1f} ± {1:.1f} %".format(headers[0]['p_int']*100.,np.ceil(headers[0]['p_int_err']*1000.)/10.))
    print("PA_int = {0:.1f} ± {1:.1f} °".format(headers[0]['pa_int'],np.ceil(headers[0]['pa_int_err']*10.)/10.))
    # Background values
    print("F_bkg({0:.0f} Angs) = ({1} ± {2})e{3} ergs.cm^-2.s^-1.Angs^-1".format(headers[0]['photplam'],*proj_plots.sci_not(I_bkg[0,0]*headers[0]['photflam'],np.sqrt(S_cov_bkg[0,0][0,0])*headers[0]['photflam'],2,out=int)))
    print("P_bkg = {0:.1f} ± {1:.1f} %".format(debiased_P_bkg[0,0]*100.,np.ceil(s_P_bkg[0,0]*1000.)/10.))
    print("PA_bkg = {0:.1f} ± {1:.1f} °".format(PA_bkg[0,0],np.ceil(s_PA_bkg[0,0]*10.)/10.))
    # Plot polarization map (Background is either total Flux, Polarization degree or Polarization degree error).
    if px_scale.lower() not in ['full','integrate'] and final_display:
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, step_vec=step_vec, savename=figname+figtype, plots_folder=plots_folder)
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, step_vec=step_vec, savename=figname+figtype+"_I", plots_folder=plots_folder, display='Intensity')
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, step_vec=step_vec, savename=figname+figtype+"_P_flux", plots_folder=plots_folder, display='Pol_Flux')
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, step_vec=step_vec, savename=figname+figtype+"_P", plots_folder=plots_folder, display='Pol_deg')
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, step_vec=step_vec, savename=figname+figtype+"_PA", plots_folder=plots_folder, display='Pol_ang')
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, step_vec=step_vec, savename=figname+figtype+"_I_err", plots_folder=plots_folder, display='I_err')
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, step_vec=step_vec, savename=figname+figtype+"_P_err", plots_folder=plots_folder, display='Pol_deg_err')
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, step_vec=step_vec, savename=figname+figtype+"_SNRi", plots_folder=plots_folder, display='SNRi')
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, step_vec=step_vec, savename=figname+figtype+"_SNRp", plots_folder=plots_folder, display='SNRp')
    elif final_display:
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, step_vec=step_vec, savename=figname+figtype, plots_folder=plots_folder, display='integrate')
    elif px_scale.lower() not in ['full', 'integrate']:
        pol_map = proj_plots.pol_map(Stokes_test, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut)

    return 0

if __name__ == "__main__":
    sys.exit(main())
