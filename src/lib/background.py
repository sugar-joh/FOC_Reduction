"""
Library function for background estimation steps of the reduction pipeline.

prototypes :
    - display_bkg(data, background, std_bkg, headers, histograms, bin_centers, rectangle, savename, plots_folder)
        Display and save how the background noise is computed.
    - bkg_hist(data, error, mask, headers, sub_shape, display, savename, plots_folder) -> n_data_array, n_error_array, headers, background)
        Compute the error (noise) of the input array by computing the base mode of each image.
    - bkg_mini(data, error, mask, headers, sub_shape, display, savename, plots_folder) -> n_data_array, n_error_array, headers, background)
        Compute the error (noise) of the input array by looking at the sub-region of minimal flux in every image and of shape sub_shape.
"""
import sys
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from datetime import datetime
from lib.plots import plot_obs

def display_bkg(data, background, std_bkg, headers, histograms=None, bin_centers=None, rectangle=None, savename=None, plots_folder="./"):
    plt.rcParams.update({'font.size': 15})
    convert_flux = np.array([head['photflam'] for head in headers])
    date_time = np.array([headers[i]['date-obs']+';'+headers[i]['time-obs']
        for i in range(len(headers))])
    date_time = np.array([datetime.strptime(d,'%Y-%m-%d;%H:%M:%S')
        for d in date_time])
    filt = np.array([headers[i]['filtnam1'] for i in range(len(headers))])
    dict_filt = {"POL0":'r', "POL60":'g', "POL120":'b'}
    c_filt = np.array([dict_filt[f] for f in filt])

    fig,ax = plt.subplots(figsize=(10,6), constrained_layout=True)
    for f in np.unique(filt):
        mask = [fil==f for fil in filt]
        ax.scatter(date_time[mask], background[mask]*convert_flux[mask],
                color=dict_filt[f],label="{0:s}".format(f))
    ax.errorbar(date_time, background*convert_flux,
            yerr=std_bkg*convert_flux, fmt='+k',
            markersize=0, ecolor=c_filt)
    # Date handling
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel("Observation date and time")
    ax.set_ylabel(r"Flux [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")
    plt.legend()
    if not(savename is None):
        fig.savefig(plots_folder+savename+"_background_flux.png", bbox_inches='tight')

    if not(histograms is None):
        filt_obs = {"POL0":0, "POL60":0, "POL120":0}
        fig_h, ax_h = plt.subplots(figsize=(10,6), constrained_layout=True)
        for i, (hist, bins) in enumerate(zip(histograms, bin_centers)):
            filt_obs[headers[i]['filtnam1']] += 1
            ax_h.plot(bins,hist,'+',color="C{0:d}".format(i),alpha=0.8,label=headers[i]['filtnam1']+' (Obs '+str(filt_obs[headers[i]['filtnam1']])+')')
            ax_h.plot([background[i],background[i]],[hist.min(), hist.max()],'x--',color="C{0:d}".format(i),alpha=0.8)
        ax_h.set_xscale('log')
        ax_h.set_xlim([background.mean()*1e-2,background.mean()*1e2])
        ax_h.set_xlabel(r"Count rate [$s^{-1}$]")
        ax_h.set_ylabel(r"Number of pixels in bin")
        ax_h.set_title("Histogram for each observation")
        plt.legend()
        if not(savename is None):
            fig_h.savefig(plots_folder+savename+'_histograms.png', bbox_inches='tight')

    fig2, ax2 = plt.subplots(figsize=(10,10))
    data0 = data[0]*convert_flux[0]
    bkg_data0 = data0 <= background[0]*convert_flux[0]
    instr = headers[0]['instrume']
    rootname = headers[0]['rootname']
    exptime = headers[0]['exptime']
    filt = headers[0]['filtnam1']
    #plots
    im = ax2.imshow(data0, norm=LogNorm(data0[data0>0.].mean()/10.,data0.max()), origin='lower', cmap='gray')
    bkg_im = ax2.imshow(bkg_data0, origin='lower', cmap='Reds', alpha=0.7)
    if not(rectangle is None):
        x, y, width, height, angle, color = rectangle[0]
        ax2.add_patch(Rectangle((x, y),width,height,edgecolor=color,fill=False))
    ax2.annotate(instr+":"+rootname, color='white', fontsize=10, xy=(0.02, 0.95), xycoords='axes fraction')
    ax2.annotate(filt, color='white', fontsize=14, xy=(0.02, 0.02), xycoords='axes fraction')
    ax2.annotate(str(exptime)+" s", color='white', fontsize=10, xy=(0.80, 0.02), xycoords='axes fraction')
    ax2.set(xlabel='pixel offset', ylabel='pixel offset')

    fig2.subplots_adjust(hspace=0, wspace=0, right=0.85)
    cbar_ax = fig2.add_axes([0.9, 0.12, 0.02, 0.75])
    fig2.colorbar(im, cax=cbar_ax, label=r"Flux [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")

    if not(savename is None):
        fig2.savefig(plots_folder+savename+'_'+filt+'_background_location.png', bbox_inches='tight')
        if not(rectangle is None):
            plot_obs(data, headers, vmin=data.min(), vmax=data.max(), rectangle=rectangle,
                    savename=savename+"_background_location",plots_folder=plots_folder)
    elif not(rectangle is None):
        plot_obs(data, headers, vmin=vmin, vmax=vmax, rectangle=rectangle)

    plt.show()


def bkg_hist(data, error, mask, headers, sub_type=None, display=False, savename=None, plots_folder=""):
    """
    ----------
    Inputs:
    data : numpy.ndarray
        Array containing the data to study (2D float arrays).
    error : numpy.ndarray
        Array of images (2D floats, aligned and of the same shape) containing
        the error in each pixel of the observation images in data_array.
    mask : numpy.ndarray
        2D boolean array delimiting the data to work on.
    headers : header list
        Headers associated with the images in data_array.
    sub_type : str or int, optional
        If str, statistic rule to be used for the number of bins in counts/s.
        If int, number of bins for the counts/s histogram.
        Defaults to "Freedman-Diaconis".
    display : boolean, optional
        If True, data_array will be displayed with a rectangle around the
        sub-image selected for background computation.
        Defaults to False.
    savename : str, optional
        Name of the figure the map should be saved to. If None, the map won't
        be saved (only displayed). Only used if display is True.
        Defaults to None.
    plots_folder : str, optional
        Relative (or absolute) filepath to the folder in wich the map will
        be saved. Not used if savename is None.
        Defaults to current folder.
    ----------
    Returns:
    data_array : numpy.ndarray
        Array containing the data to study minus the background.
    headers : header list
        Updated headers associated with the images in data_array.
    error_array : numpy.ndarray
        Array containing the background values associated to the images in
        data_array.
    background : numpy.ndarray
        Array containing the pixel background value for each image in
        data_array.
    """
    n_data_array, n_error_array = deepcopy(data), deepcopy(error)
    error_bkg = np.ones(n_data_array.shape)
    std_bkg = np.zeros((data.shape[0]))
    background = np.zeros((data.shape[0]))
    histograms, bin_centers = [], []
    
    for i, image in enumerate(data):
        #Compute the Count-rate histogram for the image
        n_mask = np.logical_and(mask,image>0.)
        if not (sub_type is None):
            if type(sub_type) == int:
                n_bins = sub_type
            elif sub_type.lower() in ['sqrt']:
                n_bins = np.fix(np.sqrt(image[n_mask].size)).astype(int) # Square-root
            elif sub_type.lower() in ['sturges']:
                n_bins = np.ceil(np.log2(image[n_mask].size)).astype(int)+1 # Sturges
            elif sub_type.lower() in ['rice']:
                n_bins = 2*np.fix(np.power(image[n_mask].size,1/3)).astype(int) # Rice
            elif sub_type.lower() in ['scott']:
                n_bins = np.fix((image[n_mask].max()-image[n_mask].min())/(3.5*image[n_mask].std()/np.power(image[n_mask].size,1/3))).astype(int) # Scott
            else:
                n_bins = np.fix((image[n_mask].max()-image[n_mask].min())/(2*np.subtract(*np.percentile(image[n_mask], [75, 25]))/np.power(image[n_mask].size,1/3))).astype(int) # Freedman-Diaconis
        else:
            n_bins = np.fix((image[n_mask].max()-image[n_mask].min())/(2*np.subtract(*np.percentile(image[n_mask], [75, 25]))/np.power(image[n_mask].size,1/3))).astype(int) # Freedman-Diaconis
        
        hist, bin_edges = np.histogram(np.log(image[n_mask]),bins=n_bins)
        histograms.append(hist)
        bin_centers.append(np.exp((bin_edges[:-1]+bin_edges[1:])/2))
        #Take the background as the count-rate with the maximum number of pixels
        hist_max = bin_centers[-1][np.argmax(hist)]
        bkg = np.sqrt(np.sum(image[np.abs(image-hist_max)/hist_max<0.5]**2)/image[np.abs(image-hist_max)/hist_max<0.5].size)
        error_bkg[i] *= bkg
       
        # Quadratically add uncertainties in the "correction factors" (see Kishimoto 1999)
        #wavelength dependence of the polariser filters
        #estimated to less than 1%
        err_wav = data[i]*0.01
        #difference in PSFs through each polarizers
        #estimated to less than 3%
        err_psf = data[i]*0.03
        #flatfielding uncertainties
        #estimated to less than 3%
        err_flat = data[i]*0.03

        n_error_array[i] = np.sqrt(n_error_array[i]**2 + error_bkg[i]**2 + err_wav**2 + err_psf**2 + err_flat**2)
        
        #Substract background
        n_data_array[i][mask] = n_data_array[i][mask] - bkg
        n_data_array[i][np.logical_and(mask,n_data_array[i] <= 0.01*bkg)] = 0.01*bkg
 
        std_bkg[i] = image[np.abs(image-bkg)/bkg<1.].std()
        background[i] = bkg

    if display:
        display_bkg(data, background, std_bkg, headers, histograms=histograms, bin_centers=bin_centers, savename=savename, plots_folder=plots_folder)
    return n_data_array, n_error_array, headers, background


def bkg_mini(data, error, mask, headers, sub_shape=(15,15), display=False, savename=None, plots_folder=""):
    """
    Look for sub-image of shape sub_shape that have the smallest integrated
    flux (no source assumption) and define the background on the image by the
    standard deviation on this sub-image.
    ----------
    Inputs:
    data : numpy.ndarray
        Array containing the data to study (2D float arrays).
    error : numpy.ndarray
        Array of images (2D floats, aligned and of the same shape) containing
        the error in each pixel of the observation images in data_array.
    mask : numpy.ndarray
        2D boolean array delimiting the data to work on.
    headers : header list
        Headers associated with the images in data_array.
    sub_shape : tuple, optional
        Shape of the sub-image to look for. Must be odd.
        Defaults to 10% of input array.
    display : boolean, optional
        If True, data_array will be displayed with a rectangle around the
        sub-image selected for background computation.
        Defaults to False.
    savename : str, optional
        Name of the figure the map should be saved to. If None, the map won't
        be saved (only displayed). Only used if display is True.
        Defaults to None.
    plots_folder : str, optional
        Relative (or absolute) filepath to the folder in wich the map will
        be saved. Not used if savename is None.
        Defaults to current folder.
    ----------
    Returns:
    data_array : numpy.ndarray
        Array containing the data to study minus the background.
    headers : header list
        Updated headers associated with the images in data_array.
    error_array : numpy.ndarray
        Array containing the background values associated to the images in
        data_array.
    background : numpy.ndarray
        Array containing the pixel background value for each image in
        data_array.
    """
    sub_shape = np.array(sub_shape)
    # Make sub_shape of odd values
    if not(np.all(sub_shape%2)):
        sub_shape += 1-sub_shape%2
    shape = np.array(data.shape)
    diff = (sub_shape-1).astype(int)
    temp = np.zeros((shape[0],shape[1]-diff[0],shape[2]-diff[1]))

    n_data_array, n_error_array = deepcopy(data), deepcopy(error)
    error_bkg = np.ones(n_data_array.shape)
    std_bkg = np.zeros((data.shape[0]))
    background = np.zeros((data.shape[0]))
    rectangle = []

    for i,image in enumerate(data):
        # Find the sub-image of smallest integrated flux (suppose no source)
        #sub-image dominated by background
        fmax = np.finfo(np.double).max
        img = deepcopy(image)
        img[1-mask] = fmax/(diff[0]*diff[1])
        for r in range(temp.shape[1]):
            for c in range(temp.shape[2]):
                temp[i][r,c] = np.where(mask[r,c], img[r:r+diff[0],c:c+diff[1]].sum(), fmax/(diff[0]*diff[1]))

    minima = np.unravel_index(np.argmin(temp.sum(axis=0)),temp.shape[1:])

    for i, image in enumerate(data):
        rectangle.append([minima[1], minima[0], sub_shape[1], sub_shape[0], 0., 'r'])
        # Compute error : root mean square of the background
        sub_image = image[minima[0]:minima[0]+sub_shape[0],minima[1]:minima[1]+sub_shape[1]]
        #bkg =  np.std(sub_image)    # Previously computed using standard deviation over the background
        bkg = np.sqrt(np.sum(sub_image**2)/sub_image.size)
        error_bkg[i] *= bkg
       
        # Quadratically add uncertainties in the "correction factors" (see Kishimoto 1999)
        #wavelength dependence of the polariser filters
        #estimated to less than 1%
        err_wav = data[i]*0.01
        #difference in PSFs through each polarizers
        #estimated to less than 3%
        err_psf = data[i]*0.03
        #flatfielding uncertainties
        #estimated to less than 3%
        err_flat = data[i]*0.03

        n_error_array[i] = np.sqrt(n_error_array[i]**2 + error_bkg[i]**2 + err_wav**2 + err_psf**2 + err_flat**2)
        
        #Substract background
        n_data_array[i][mask] = n_data_array[i][mask] - bkg
        n_data_array[i][np.logical_and(mask,n_data_array[i] <= 0.01*bkg)] = 0.01*bkg
 
        std_bkg[i] = image[np.abs(image-bkg)/bkg<1.].std()
        background[i] = bkg

    if display:
        display_bkg(data, background, std_bkg, headers, rectangle=rectangle, savename=savename, plots_folder=plots_folder)
    return n_data_array, n_error_array, headers, background

