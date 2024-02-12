#!/usr/bin/python3
#-*- coding:utf-8 -*-
"""
Main script where are progressively added the steps for the FOC pipeline reduction.
"""

#Project libraries
import numpy as np
from copy import deepcopy
import lib.fits as proj_fits        #Functions to handle fits files
import lib.reduction as proj_red    #Functions used in reduction pipeline
import lib.plots as proj_plots      #Functions for plotting data
from lib.deconvolve import from_file_psf
from lib.query import retrieve_products, path_exists, system
from matplotlib.colors import LogNorm


def main(target=None, proposal_id=None, infiles=None, output_dir="./data", crop=0, interactive=0):
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
    
    # Background estimation
    error_sub_type = 'freedman-diaconis'   #sqrt, sturges, rice, scott, freedman-diaconis (default) or shape (example (51,51))
    subtract_error = 1.00
    display_error = False
    
    # Data binning
    rebin = True
    pxsize = 0.10
    px_scale = 'arcsec'         #pixel, arcsec or full
    rebin_operation = 'sum'     #sum or average
    
    # Alignement
    align_center = 'center'          #If None will not align the images
    display_bkg = False
    display_align = False
    display_data = False
    
    # Smoothing
    smoothing_function = 'combine'  #gaussian_after, weighted_gaussian_after, gaussian, weighted_gaussian or combine
    smoothing_FWHM = 0.20           #If None, no smoothing is done
    smoothing_scale = 'arcsec'      #pixel or arcsec
    
    # Rotation
    rotate_data = False             #rotation to North convention can give erroneous results
    rotate_stokes = True
    
    # Final crop
    #crop = False                    #Crop to desired ROI
    #interactive = False             #Whether to output to intercative analysis tool
    
    # Polarization map output
    SNRp_cut = 3.    #P measurments with SNR>3
    SNRi_cut = 30.   #I measurments with SNR>30, which implies an uncertainty in P of 4.7%.
    flux_lim = None    #lowest and highest flux displayed on plot, defaults to bkg and maximum in cut if None
    vec_scale = 3
    step_vec = 1    #plot all vectors in the array. if step_vec = 2, then every other vector will be plotted
                    # if step_vec = 0 then all vectors are displayed at full length

    ##### Pipeline start
    ## Step 1:
    # Get data from fits files and translate to flux in erg/cm²/s/Angstrom.
    if not infiles is None:
        prod = np.array([["/".join(filepath.split('/')[:-1]),filepath.split('/')[-1]] for filepath in infiles],dtype=str)
        obs_dir = "/".join(infiles[0].split("/")[:-1])
        if not path_exists(obs_dir):
            system("mkdir -p {0:s} {1:s}".format(obs_dir,obs_dir.replace("data","plots")))
        if target is None:
            target = input("Target name:\n>")
    else:
        target, products = retrieve_products(target,proposal_id,output_dir=output_dir)
        prod = products.pop()
        for prods in products:
            main(target=target,infiles=["/".join(pr) for pr in prods],output_dir=output_dir)
    data_folder = prod[0][0]
    try:
        plots_folder = data_folder.replace("data","plots")
    except:
        plots_folder = "."
    if not path_exists(plots_folder):
        system("mkdir -p {0:s} ".format(plots_folder))
    infiles = [p[1] for p in prod]
    data_array, headers = proj_fits.get_obs_data(infiles, data_folder=data_folder, compute_flux=True)

    figname = "_".join([target,"FOC"])
    if rebin:
        if not px_scale in ['full']:
            figtype = "".join(["b","{0:.2f}".format(pxsize),px_scale])    #additionnal informations
        else:
            figtype = "full"
    if not smoothing_FWHM is None:
        figtype += "_"+"".join(["".join([s[0] for s in smoothing_function.split("_")]),"{0:.2f}".format(smoothing_FWHM),smoothing_scale])    #additionnal informations
    if align_center is None:
        figtype += "_not_aligned"

    # Crop data to remove outside blank margins.
    data_array, error_array, headers = proj_red.crop_array(data_array, headers, step=5, null_val=0., inside=True, display=display_crop, savename=figname, plots_folder=plots_folder)

    # Deconvolve data using Richardson-Lucy iterative algorithm with a gaussian PSF of given FWHM.
    if deconvolve:
        data_array = proj_red.deconvolve_array(data_array, headers, psf=psf, FWHM=psf_FWHM, scale=psf_scale, shape=psf_shape, iterations=iterations, algo=algo)

    # Estimate error from data background, estimated from sub-image of desired sub_shape.
    background = None
    data_array, error_array, headers, background = proj_red.get_error(data_array, headers, error_array, sub_type=error_sub_type, subtract_error=subtract_error, display=display_error, savename="_".join([figname,"errors"]), plots_folder=plots_folder, return_background=True)

    if display_bkg:
        proj_plots.plot_obs(data_array, headers, vmin=data_array[data_array>0.].min()*headers[0]['photflam'], vmax=data_array[data_array>0.].max()*headers[0]['photflam'], savename="_".join([figname,"bkg"]), plots_folder=plots_folder)

    # Align and rescale images with oversampling.
    data_array, error_array, headers, data_mask = proj_red.align_data(data_array, headers, error_array=error_array, background=background, upsample_factor=10, ref_center=align_center, return_shifts=False)

    if display_align:
        proj_plots.plot_obs(data_array, headers, vmin=data_array[data_array>0.].min()*headers[0]['photflam'], vmax=data_array[data_array>0.].max()*headers[0]['photflam'], savename="_".join([figname,str(align_center)]), plots_folder=plots_folder)

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
        proj_plots.plot_obs(data_array, headers, vmin=data_array[data_array>0.].min()*headers[0]['photflam'], vmax=data_array[data_array>0.].max()*headers[0]['photflam'], savename="_".join([figname,"rebin"]), plots_folder=plots_folder)

    background = np.array([np.array(bkg).reshape(1,1) for bkg in background])
    background_error = np.array([np.array(np.sqrt((bkg-background[np.array([h['filtnam1']==head['filtnam1'] for h in headers],dtype=bool)].mean())**2/np.sum([h['filtnam1']==head['filtnam1'] for h in headers]))).reshape(1,1) for bkg,head in zip(background,headers)])

    ## Step 2:
    # Compute Stokes I, Q, U with smoothed polarized images
    # SMOOTHING DISCUSSION :
    # FWHM of FOC have been estimated at about 0.03" across 1500-5000 Angstrom band, which is about 2 detector pixels wide
    # see Jedrzejewski, R.; Nota, A.; Hack, W. J., A Comparison Between FOC and WFPC2
    # Bibcode : 1995chst.conf...10J
    I_stokes, Q_stokes, U_stokes, Stokes_cov = proj_red.compute_Stokes(data_array, error_array, data_mask, headers, FWHM=smoothing_FWHM, scale=smoothing_scale, smoothing=smoothing_function,transmitcorr=False)
    I_bkg, Q_bkg, U_bkg, S_cov_bkg = proj_red.compute_Stokes(background, background_error, np.array(True).reshape(1,1), headers, FWHM=None, scale=smoothing_scale, smoothing=smoothing_function,transmitcorr=False)

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
    Stokes_test = proj_fits.save_Stokes(I_stokes, Q_stokes, U_stokes, Stokes_cov, P, debiased_P, s_P, s_P_P, PA, s_PA, s_PA_P, headers, data_mask, "_".join([figname,figtype]), data_folder=data_folder, return_hdul=True)
    data_mask = Stokes_test[-1].data.astype(bool)

    ## Step 5:
    # crop to desired region of interest (roi)
    if crop:
        figtype += "_crop"
        stokescrop = proj_plots.crop_Stokes(deepcopy(Stokes_test),norm=LogNorm())
        stokescrop.crop()
        stokescrop.writeto("/".join([data_folder,"_".join([figname,figtype+".fits"])]))
        Stokes_test, data_mask, headers = stokescrop.hdul_crop, stokescrop.data_mask, [dataset.header for dataset in stokescrop.hdul_crop]

    print("F_int({0:.0f} Angs) = ({1} ± {2})e{3} ergs.cm^-2.s^-1.Angs^-1".format(headers[0]['photplam'],*proj_plots.sci_not(Stokes_test[0].data[data_mask].sum()*headers[0]['photflam'],np.sqrt(Stokes_test[3].data[0,0][data_mask].sum())*headers[0]['photflam'],2,out=int)))
    print("P_int = {0:.1f} ± {1:.1f} %".format(headers[0]['p_int']*100.,np.ceil(headers[0]['p_int_err']*1000.)/10.))
    print("PA_int = {0:.1f} ±t {1:.1f} °".format(headers[0]['pa_int'],np.ceil(headers[0]['pa_int_err']*10.)/10.))
    # Background values
    print("F_bkg({0:.0f} Angs) = ({1} ± {2})e{3} ergs.cm^-2.s^-1.Angs^-1".format(headers[0]['photplam'],*proj_plots.sci_not(I_bkg[0,0]*headers[0]['photflam'],np.sqrt(S_cov_bkg[0,0][0,0])*headers[0]['photflam'],2,out=int)))
    print("P_bkg = {0:.1f} ± {1:.1f} %".format(debiased_P_bkg[0,0]*100.,np.ceil(s_P_bkg[0,0]*1000.)/10.))
    print("PA_bkg = {0:.1f} ± {1:.1f} °".format(PA_bkg[0,0],np.ceil(s_PA_bkg[0,0]*10.)/10.))
    # Plot polarization map (Background is either total Flux, Polarization degree or Polarization degree error).
    if px_scale.lower() not in ['full','integrate'] and not interactive:
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim, step_vec=step_vec, vec_scale=vec_scale, savename="_".join([figname,figtype]), plots_folder=plots_folder)
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim, step_vec=step_vec, vec_scale=vec_scale, savename="_".join([figname,figtype,"I"]), plots_folder=plots_folder, display='Intensity')
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim, step_vec=step_vec, vec_scale=vec_scale, savename="_".join([figname,figtype,"P_flux"]), plots_folder=plots_folder, display='Pol_Flux')
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim, step_vec=step_vec, vec_scale=vec_scale, savename="_".join([figname,figtype,"P"]), plots_folder=plots_folder, display='Pol_deg')
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim, step_vec=step_vec, vec_scale=vec_scale, savename="_".join([figname,figtype,"PA"]), plots_folder=plots_folder, display='Pol_ang')
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim, step_vec=step_vec, vec_scale=vec_scale, savename="_".join([figname,figtype,"I_err"]), plots_folder=plots_folder, display='I_err')
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim, step_vec=step_vec, vec_scale=vec_scale, savename="_".join([figname,figtype,"P_err"]), plots_folder=plots_folder, display='Pol_deg_err')
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim, step_vec=step_vec, vec_scale=vec_scale, savename="_".join([figname,figtype,"SNRi"]), plots_folder=plots_folder, display='SNRi')
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim, step_vec=step_vec, vec_scale=vec_scale, savename="_".join([figname,figtype,"SNRp"]), plots_folder=plots_folder, display='SNRp')
    elif not interactive:
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, savename="_".join([figname,figtype]), plots_folder=plots_folder, display='integrate')
    elif px_scale.lower() not in ['full', 'integrate']:
        pol_map = proj_plots.pol_map(Stokes_test, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, flux_lim=flux_lim)

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Query MAST for target products')
    parser.add_argument('-t','--target', metavar='targetname', required=False,
                        help='the name of the target', type=str, default=None)
    parser.add_argument('-p','--proposal_id', metavar='proposal_id', required=False,
                        help='the proposal id of the data products', type=int, default=None)
    parser.add_argument('-f','--files', metavar='path', required=False, nargs='*',
                        help='the full or relative path to the data products', default=None)
    parser.add_argument('-o','--output_dir', metavar='directory_path', required=False,
                        help='output directory path for the data products', type=str, default="./data")
    parser.add_argument('-c','--crop', metavar='crop_boolean', required=False,
                        help='whether to crop the analysis region', type=int, default=0)
    parser.add_argument('-i','--interactive', metavar='interactive_boolean', required=False,
                        help='whether to output to the interactive analysis tool', type=int, default=0)
    args = parser.parse_args()
    exitcode = main(target=args.target, proposal_id=args.proposal_id, infiles=args.files, output_dir=args.output_dir, crop=args.crop, interactive=args.interactive)
    print("Finished with ExitCode: ",exitcode)
