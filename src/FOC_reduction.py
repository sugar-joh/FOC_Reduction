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


def main(target=None, proposal_id=None, infiles=None, output_dir="./data"):
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
    subtract_error = 1.25
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
    smoothing_function = 'gaussian'  #gaussian_after, weighted_gaussian_after, gaussian, weighted_gaussian or combine
    smoothing_FWHM = None           #If None, no smoothing is done
    smoothing_scale = 'arcsec'      #pixel or arcsec
    
    # Rotation
    rotate_stokes = True
    rotate_data = False             #rotation to North convention can give erroneous results
    
    # Final crop
    crop = False                    #Crop to desired ROI
    final_display = True           #Whether to display all polarization map outputs
    
    # Polarization map output
    SNRp_cut = 3.    #P measurments with SNR>3
    SNRi_cut = 30.   #I measurments with SNR>30, which implies an uncertainty in P of 4.7%.
    vec_scale = 2.0
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
        prod = products[0]
        for prods in products[1:]:
            main(target=target,infiles=["/".join(pr) for pr in prods],output_dir=output_dir)
    data_folder = prod[0,0]
    try:
        plots_folder = data_folder.replace("data","plots")
    except:
        plots_folder = "."
    infiles = prod[:,1]
    data_array, headers = proj_fits.get_obs_data(infiles, data_folder=data_folder, compute_flux=True)

    figname = "_".join([target,"FOC"])
    if smoothing_FWHM is None:
        if px_scale in ['px','pixel','pixels']:
            figtype = "".join(["b_",str(pxsize),'px'])
        elif px_scale in ['arcsec','arcseconds','arcs']:
            figtype = "".join(["b_","{0:.2f}".format(pxsize).replace(".",""),'arcsec'])
        else:
            figtype = "full"
    else:
        figtype = "_".join(["".join([s[0] for s in smoothing_function.split("_")]),"".join(["{0:.2f}".format(smoothing_FWHM).replace(".",""),smoothing_scale])])    #additionnal informations

    # Crop data to remove outside blank margins.
    data_array, error_array, headers = proj_red.crop_array(data_array, headers, step=5, null_val=0., inside=True, display=display_crop, savename=figname, plots_folder=plots_folder)

    # Deconvolve data using Richardson-Lucy iterative algorithm with a gaussian PSF of given FWHM.
    if deconvolve:
        data_array = proj_red.deconvolve_array(data_array, headers, psf=psf, FWHM=psf_FWHM, scale=psf_scale, shape=psf_shape, iterations=iterations, algo=algo)

    # Estimate error from data background, estimated from sub-image of desired sub_shape.
    background = None
    data_array, error_array, headers, background = proj_red.get_error(data_array, headers, error_array, sub_type=error_sub_type, subtract_error=subtract_error, display=display_error, savename="_".join([figname,"errors"]), plots_folder=plots_folder, return_background=True)

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
        proj_plots.plot_obs(data_array, headers, vmin=data_array[data_array>0.].min()*headers[0]['photflam'], vmax=data_array[data_array>0.].max()*headers[0]['photflam'], savename="_".join([figname,"center",align_center]), plots_folder=plots_folder)

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
        figtype += "crop"
        stokescrop = proj_plots.crop_Stokes(deepcopy(Stokes_test))
        stokescrop.crop()
        stokescrop.writeto("/".join([data_folder,"_".join([figname,figtype+".fits"])]))
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
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, step_vec=step_vec, vec_scale=vec_scale, savename="_".join([figname,figtype]), plots_folder=plots_folder)
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, step_vec=step_vec, vec_scale=vec_scale, savename="_".join([figname,figtype,"I"]), plots_folder=plots_folder, display='Intensity')
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, step_vec=step_vec, vec_scale=vec_scale, savename="_".join([figname,figtype,"P_flux"]), plots_folder=plots_folder, display='Pol_Flux')
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, step_vec=step_vec, vec_scale=vec_scale, savename="_".join([figname,figtype,"P"]), plots_folder=plots_folder, display='Pol_deg')
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, step_vec=step_vec, vec_scale=vec_scale, savename="_".join([figname,figtype,"PA"]), plots_folder=plots_folder, display='Pol_ang')
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, step_vec=step_vec, vec_scale=vec_scale, savename="_".join([figname,figtype,"I_err"]), plots_folder=plots_folder, display='I_err')
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, step_vec=step_vec, vec_scale=vec_scale, savename="_".join([figname,figtype,"P_err"]), plots_folder=plots_folder, display='Pol_deg_err')
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, step_vec=step_vec, vec_scale=vec_scale, savename="_".join([figname,figtype,"SNRi"]), plots_folder=plots_folder, display='SNRi')
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, step_vec=step_vec, vec_scale=vec_scale, savename="_".join([figname,figtype,"SNRp"]), plots_folder=plots_folder, display='SNRp')
    elif final_display:
        proj_plots.polarization_map(deepcopy(Stokes_test), data_mask, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, savename="_".join([figname,figtype]), plots_folder=plots_folder, display='integrate')
    elif px_scale.lower() not in ['full', 'integrate']:
        pol_map = proj_plots.pol_map(Stokes_test, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut)

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
    args = parser.parse_args()
    exitcode = main(target=args.target, proposal_id=args.proposal_id, infiles=args.files, output_dir=args.output_dir)
    print("Finished with ExitCode: ",exitcode)
