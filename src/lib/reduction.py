"""
Library function computing various steps of the reduction pipeline.

prototypes :
    - bin_ndarray(ndarray, new_shape, operation) -> ndarray
        Bins an ndarray to new_shape.

    - crop_array(data_array, error_array, data_mask, step, null_val, inside, display, savename, plots_folder) -> crop_data_array, crop_error_array (, crop_mask), crop_headers
        Homogeneously crop out null edges off a data array.

    - deconvolve_array(data_array, headers, psf, FWHM, scale, shape, iterations, algo) -> deconvolved_data_array
        Homogeneously deconvolve a data array using a chosen deconvolution algorithm.

    - get_error(data_array, headers, error_array, data_mask, sub_shape, display, savename, plots_folder, return_background) -> data_array, error_array, headers (, background)
        Compute the error (noise) on each image of the input array.

    - rebin_array(data_array, error_array, headers, pxsize, scale, operation, data_mask) -> rebinned_data, rebinned_error, rebinned_headers, Dxy (, data_mask)
        Homegeneously rebin a data array given a target pixel size in scale units.

    - align_data(data_array, error_array, upsample_factor, ref_data, ref_center, return_shifts) -> data_array, error_array (, shifts, errors)
        Align data_array on ref_data by cross-correlation.

    - smooth_data(data_array, error_array, data_mask, headers, FWHM, scale, smoothing) -> smoothed_array, smoothed_error
        Smooth data by convoluting with a gaussian or by combining weighted images

    - polarizer_avg(data_array, error_array, data_mask, headers, FWHM, scale, smoothing) -> polarizer_array, polarizer_cov
        Average images in data_array on each used polarizer filter and compute correlated errors.

    - compute_Stokes(data_array, error_array, data_mask, headers, FWHM, scale, smoothing) -> I_stokes, Q_stokes, U_stokes, Stokes_cov
        Compute Stokes parameters I, Q and U and their respective correlated errors from data_array.

    - compute_pol(I_stokes, Q_stokes, U_stokes, Stokes_cov, headers) -> P, debiased_P, s_P, s_P_P, PA, s_PA, s_PA_P
        Compute polarization degree (in %) and angle (in degree) and their respective errors.

    - rotate_Stokes(I_stokes, Q_stokes, U_stokes, Stokes_cov, data_mask, headers, ang, SNRi_cut) -> I_stokes, Q_stokes, U_stokes, Stokes_cov, data_mask, headers
        Rotate I, Q, U given an angle in degrees using scipy functions.
    
    - rotate_data(data_array, error_array, data_mask, headers, ang) -> data_array, error_array, data_mask, headers
        Rotate data before reduction given an angle in degrees using scipy functions.
"""

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime
from scipy.ndimage import rotate as sc_rotate, shift as sc_shift
from scipy.signal import convolve2d
from astropy.wcs import WCS
from astropy import log
log.setLevel('ERROR')
import warnings
from lib.deconvolve import deconvolve_im, gaussian_psf, gaussian2d, zeropad
from lib.convex_hull import image_hull, clean_ROI
from lib.plots import plot_obs
from lib.cross_correlation import phase_cross_correlation


# Useful tabulated values
#FOC instrument
globals()['trans2'] = {'f140w' : 0.21, 'f175w' : 0.24, 'f220w' : 0.39, 'f275w' : 0.40, 'f320w' : 0.89, 'f342w' : 0.81, 'f430w' : 0.74, 'f370lp' : 0.83, 'f486n' : 0.63, 'f501n' : 0.68, 'f480lp' : 0.82, 'clear2' : 1.0}
globals()['trans3'] = {'f120m' : 0.10, 'f130m' : 0.10, 'f140m' : 0.08, 'f152m' : 0.08, 'f165w' : 0.28, 'f170m' : 0.18, 'f195w' : 0.42, 'f190m' : 0.15, 'f210m' : 0.18, 'f231m' : 0.18, 'clear3' : 1.0}
globals()['trans4'] = {'f253m' : 0.18, 'f278m' : 0.26, 'f307m' : 0.26, 'f130lp' : 0.92, 'f346m' : 0.58, 'f372m' : 0.73, 'f410m' : 0.58, 'f437m' : 0.71, 'f470m' : 0.79, 'f502m' : 0.82, 'f550m' : 0.77, 'clear4' : 1.0}
globals()['pol_efficiency'] = {'pol0' : 0.92, 'pol60' : 0.92, 'pol120' : 0.91}
# POL0 = 0deg, POL60 = 60deg, POL120=120deg
globals()['theta'] = np.array([180.*np.pi/180., 60.*np.pi/180., 120.*np.pi/180.])
# Uncertainties on the orientation of the polarizers' axes taken to be 3deg (see Nota et. al 1996, p36; Robinson & Thomson 1995)
globals()['sigma_theta'] = np.array([3.*np.pi/180., 3.*np.pi/180., 3.*np.pi/180.])


def princ_angle(ang):
    """
    Return the principal angle in the 0-180° quadrant.
    """
    while ang < 0.:
        ang += 180.
    while ang > 180.:
        ang -= 180.
    return ang


def get_row_compressor(old_dimension, new_dimension, operation='sum'):
    """
    Return the matrix that allows to compress an array from an old dimension of
    rows to a new dimension of rows, can be done by summing the original
    components or averaging them.
    ----------
    Inputs:
    old_dimension, new_dimension : int
        Number of rows in the original and target matrices.
    operation : str, optional
        Set the way original components of the matrix are put together
        between summing ('sum') and averaging ('average', 'avg', 'mean') them.
        Defaults to 'sum'.
    ----------
    Returns:
    dim_compressor : numpy.ndarray
        2D matrix allowing row compression by matrix multiplication to the left
        of the original matrix.
    """
    dim_compressor = np.zeros((new_dimension, old_dimension))
    bin_size = float(old_dimension) / new_dimension
    next_bin_break = bin_size
    which_row, which_column = 0, 0

    while which_row < dim_compressor.shape[0] and which_column < dim_compressor.shape[1]:
        if round(next_bin_break - which_column, 10) >= 1:
            dim_compressor[which_row, which_column] = 1
            which_column += 1
        elif next_bin_break == which_column:

            which_row += 1
            next_bin_break += bin_size
        else:
            partial_credit = next_bin_break - which_column
            dim_compressor[which_row, which_column] = partial_credit
            which_row += 1
            dim_compressor[which_row, which_column] = 1 - partial_credit
            which_column += 1
            next_bin_break += bin_size

    if operation.lower() in ["mean", "average", "avg"]:
        dim_compressor /= bin_size

    return dim_compressor


def get_column_compressor(old_dimension, new_dimension, operation='sum'):
    """
    Return the matrix that allows to compress an array from an old dimension of
    columns to a new dimension of columns, can be done by summing the original
    components or averaging them.
    ----------
    Inputs:
    old_dimension, new_dimension : int
        Number of columns in the original and target matrices.
    operation : str, optional
        Set the way original components of the matrix are put together
        between summing ('sum') and averaging ('average', 'avg', 'mean') them.
        Defaults to 'sum'.
    ----------
    Returns:
    dim_compressor : numpy.ndarray
        2D matrix allowing columns compression by matrix multiplication to the
        right of the original matrix.
    """
    return get_row_compressor(old_dimension, new_dimension, operation).transpose()


def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    if not operation.lower() in ['sum', 'mean', 'average', 'avg']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    if (np.array(ndarray.shape)%np.array(new_shape) == np.array([0.,0.])).all():
        compression_pairs = [(d, c//d) for d,c in zip(new_shape, ndarray.shape)]
        flattened = [l for p in compression_pairs for l in p]
        ndarray = ndarray.reshape(flattened)

        for i in range(len(new_shape)):
            if operation.lower() == "sum":
                ndarray = ndarray.sum(-1*(i+1))
            elif operation.lower() in ["mean", "average", "avg"]:
                ndarray = ndarray.mean(-1*(i+1))
    else:
        row_comp = np.mat(get_row_compressor(ndarray.shape[0], new_shape[0], operation))
        col_comp = np.mat(get_column_compressor(ndarray.shape[1], new_shape[1], operation))
        ndarray = np.array(row_comp * np.mat(ndarray) * col_comp)

    return ndarray


def crop_array(data_array, headers, error_array=None, data_mask=None, step=5,
            null_val=None, inside=False, display=False, savename=None,
            plots_folder=""):
    """
    Homogeneously crop an array: all contained images will have the same shape.
    'inside' parameter will decide how much should be cropped.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array containing the observation data (2D float arrays) to
        homogeneously crop.
    headers : header list
        Headers associated with the images in data_array.
    error_array : numpy.ndarray, optional
        Array of images (2D floats, aligned and of the same shape) containing
        the error in each pixel of the observation images in data_array.
        If None, will be initialized to zeros.
        Defaults to None.
    data_mask : numpy.ndarray, optional
        2D boolean array delimiting the data to work on.
        If None, will be initialized with a full true mask.
        Defaults to None.
    step : int, optional
        For images with straight edges, not all lines and columns need to be
        browsed in order to have a good convex hull. Step value determine
        how many row/columns can be jumped. With step=2 every other line will
        be browsed.
        Defaults to 5.
    null_val : float or array-like, optional
        Pixel value determining the threshold for what is considered 'outside'
        the image. All border pixels below this value will be taken out.
        If None, will be put to 75% of the mean value of the associated error
        array.
        Defaults to None.
    inside : boolean, optional
        If True, the cropped image will be the maximum rectangle included
        inside the image. If False, the cropped image will be the minimum
        rectangle in which the whole image is included.
        Defaults to False.
    display : boolean, optional
        If True, data_array will be displayed with a rectangle around the
        sub-image selected for region of interest.
        Defaults to False.
    ----------
    Returns:
    cropped_array : numpy.ndarray
        Array containing the observationnal data homogeneously cropped.
    headers : header list
        Updated headers associated with the images in data_array.
    cropped_error : numpy.ndarray
        Array containing the error on the observationnal data homogeneously cropped.
    """
    if error_array is None:
        error_array = np.zeros(data_array.shape)
    if null_val is None:
        null_val = [1.00*error.mean() for error in error_array]
    elif type(null_val) is float:
        null_val = [null_val,]*error_array.shape[0]

    vertex = np.zeros((data_array.shape[0],4),dtype=int)
    for i,image in enumerate(data_array):   # Get vertex of the rectangular convex hull of each image
        vertex[i] = image_hull(image,step=step,null_val=null_val[i],inside=inside)
    v_array = np.zeros(4,dtype=int)
    if inside:  # Get vertex of the maximum convex hull for all images
        v_array[0] = np.max(vertex[:,0]).astype(int)
        v_array[1] = np.min(vertex[:,1]).astype(int)
        v_array[2] = np.max(vertex[:,2]).astype(int)
        v_array[3] = np.min(vertex[:,3]).astype(int)
    else:       # Get vertex of the minimum convex hull for all images
        v_array[0] = np.min(vertex[:,0]).astype(int)
        v_array[1] = np.max(vertex[:,1]).astype(int)
        v_array[2] = np.min(vertex[:,2]).astype(int)
        v_array[3] = np.max(vertex[:,3]).astype(int)

    new_shape = np.array([v_array[1]-v_array[0],v_array[3]-v_array[2]])
    rectangle = [v_array[2], v_array[0], new_shape[1], new_shape[0], 0., 'b']
    if display:
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(figsize=(10,10))
        data = data_array[0]
        instr = headers[0]['instrume']
        rootname = headers[0]['rootname']
        exptime = headers[0]['exptime']
        filt = headers[0]['filtnam1']
        #plots
        im = ax.imshow(data, vmin=data.min(), vmax=data.max(), origin='lower', cmap='gray')
        x, y, width, height, angle, color = rectangle
        ax.add_patch(Rectangle((x, y),width,height,edgecolor=color,fill=False))
        #position of centroid
        ax.plot([data.shape[1]/2, data.shape[1]/2], [0,data.shape[0]-1], '--', lw=1,
                color='grey', alpha=0.3)
        ax.plot([0,data.shape[1]-1], [data.shape[1]/2, data.shape[1]/2], '--', lw=1,
                color='grey', alpha=0.3)
        ax.annotate(instr+":"+rootname, color='white', fontsize=10,
                xy=(0.02, 0.95), xycoords='axes fraction')
        ax.annotate(filt, color='white', fontsize=14, xy=(0.02, 0.02),
                xycoords='axes fraction')
        ax.annotate(str(exptime)+" s", color='white', fontsize=10, xy=(0.80, 0.02),
                xycoords='axes fraction')
        ax.set(#title="Location of cropped image.",
                xlabel='pixel offset',
                ylabel='pixel offset')

        fig.subplots_adjust(hspace=0, wspace=0, right=0.85)
        cbar_ax = fig.add_axes([0.9, 0.12, 0.02, 0.75])
        fig.colorbar(im, cax=cbar_ax, label=r'$Counts \cdot s^{-1}$')

        if not(savename is None):
            #fig.suptitle(savename+'_'+filt+'_crop_region')
            fig.savefig(plots_folder+savename+'_'+filt+'_crop_region.png',
                    bbox_inches='tight')
            plot_obs(data_array, headers, vmin=data_array.min(),
                    vmax=data_array.max(), rectangle=[rectangle,]*len(headers),
                    savename=savename+'_crop_region',plots_folder=plots_folder)
        plt.show()

    crop_headers = deepcopy(headers)
    crop_array = np.zeros((data_array.shape[0],new_shape[0],new_shape[1]))
    crop_error_array = np.zeros((data_array.shape[0],new_shape[0],new_shape[1]))
    for i,image in enumerate(data_array):
        #Put the image data in the cropped array
        crop_array[i] = image[v_array[0]:v_array[1],v_array[2]:v_array[3]]
        crop_error_array[i] = error_array[i][v_array[0]:v_array[1],v_array[2]:v_array[3]]
        #Update CRPIX value in the associated header
        curr_wcs = deepcopy(WCS(crop_headers[i]))
        curr_wcs.wcs.crpix = curr_wcs.wcs.crpix - np.array([v_array[2], v_array[0]])
        crop_headers[i].update(curr_wcs.to_header())
    if not data_mask is None:
        crop_mask = data_mask[v_array[0]:v_array[1],v_array[2]:v_array[3]]
        return crop_array, crop_error_array, crop_mask, crop_headers
    else:
        return crop_array, crop_error_array, crop_headers


def deconvolve_array(data_array, headers, psf='gaussian', FWHM=1., scale='px',
        shape=(9,9), iterations=20, algo='richardson'):
    """
    Homogeneously deconvolve a data array using Richardson-Lucy iterative algorithm.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array containing the observation data (2D float arrays) to
        homogeneously deconvolve.
    headers : header list
        Headers associated with the images in data_array.
    psf : str or numpy.ndarray, optionnal
        String designing the type of desired Point Spread Function or array
        of dimension 2 corresponding to the weights of a PSF.
        Defaults to 'gaussian' type PSF.
    FWHM : float, optional
        Full Width at Half Maximum for desired PSF in 'scale units. Only used
        for relevant values of 'psf' variable.
        Defaults to 1.
    scale : str, optional
        Scale units for the FWHM of the PSF between 'pixel' and 'arcsec'.
        Defaults to 'pixel'.
    shape : tuple, optional
        Shape of the kernel of the PSF. Must be of dimension 2. Only used for
        relevant values of 'psf' variable.
        Defaults to (9,9).
    iterations : int, optional
        Number of iterations for iterative deconvolution algorithms. Act as
        as a regulation of the process.
        Defaults to 20.
    algo : str, optional
        Name of the deconvolution algorithm that will be used. Implemented 
        algorithms are the following : 'Wiener', 'Van-Cittert',
        'One Step Gradient', 'Conjugate Gradient' and 'Richardson-Lucy'.
        Defaults to 'Richardson-Lucy'.
    ----------
    Returns:
    deconv_array : numpy.ndarray
        Array containing the deconvolved data (2D float arrays) using given
        point spread function.
    """
    # If chosen FWHM scale is 'arcsec', compute FWHM in pixel scale
    if scale.lower() in ['arcsec','arcseconds']:
        pxsize = np.zeros((data_array.shape[0],2))
        for i,header in enumerate(headers):
            # Get current pixel size
            w = WCS(header).deepcopy()
            pxsize[i] = np.round(w.wcs.cdelt/3600.,15)
        if (pxsize != pxsize[0]).any():
            raise ValueError("Not all images in array have same pixel size")
        FWHM /= pxsize[0].min()

    # Define Point-Spread-Function kernel
    if psf.lower() in ['gauss','gaussian']:
        kernel = gaussian_psf(FWHM=FWHM, shape=shape)
    elif (type(psf) == np.ndarray) and (len(psf.shape) == 2):
        kernel = psf
    else:
        raise ValueError("{} is not a valid value for 'psf'".format(psf))

    # Deconvolve images in the array using given PSF
    deconv_array = np.zeros(data_array.shape)
    for i,image in enumerate(data_array):
        deconv_array[i] = deconvolve_im(image, kernel, iterations=iterations,
                clip=True, filter_epsilon=None, algo='richardson')

    return deconv_array


def get_error(data_array, headers, error_array=None, data_mask=None,
            sub_shape=None, display=False, savename=None, plots_folder="",
            return_background=False):
    """
    Look for sub-image of shape sub_shape that have the smallest integrated
    flux (no source assumption) and define the background on the image by the
    standard deviation on this sub-image.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array containing the data to study (2D float arrays).
    headers : header list
        Headers associated with the images in data_array.
    error_array : numpy.ndarray, optional
        Array of images (2D floats, aligned and of the same shape) containing
        the error in each pixel of the observation images in data_array.
        If None, will be initialized to zeros.
        Defaults to None.
    data_mask : numpy.ndarray, optional
        2D boolean array delimiting the data to work on.
        If None, will be initialized with a full true mask.
        Defaults to None.
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
    return_background : boolean, optional
        If True, the pixel background value for each image in data_array is
        returned.
        Defaults to False.
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
        Only returned if return_background is True.
    """
    # Crop out any null edges
    if error_array is None:
        error_array = np.zeros(data_array.shape)
    if not data_mask is None:
        data, error, mask = data_array, error_array, data_mask
    else:
        data, error, _ = crop_array(data_array, headers, error_array, step=5, null_val=0., inside=False)
        mask = np.ones(data[0].shape, dtype=bool)
    if sub_shape is None:
        sub_shape = (np.array(data_array.shape[1:])/10).astype(int)
    sub_shape = np.array(sub_shape)
    # Make sub_shape of odd values
    if not(np.all(sub_shape%2)):
        sub_shape += 1-sub_shape%2

    shape = np.array(data.shape)
    diff = (sub_shape-1).astype(int)
    temp = np.zeros((shape[0],shape[1]-diff[0],shape[2]-diff[1]))
    error_bkg = np.ones(data_array.shape)
    rectangle = []
    background = np.zeros((shape[0]))

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
        #error =  np.std(sub_image)    # Previously computed using standard deviation over the background
        bkg = np.sqrt(np.sum((sub_image-sub_image.mean())**2)/sub_image.size)
        error_bkg[i] *= bkg

        data_array[i] = np.abs(data_array[i] - sub_image.mean())
        # Quadratically add uncertainties in the "correction factors" (see Kishimoto 1999)
        #wavelength dependence of the polariser filters
        #estimated to less than 1%
        err_wav = data_array[i]*0.01
        #difference in PSFs through each polarizers
        #estimated to less than 3%
        err_psf = data_array[i]*0.03
        #flatfielding uncertainties
        #estimated to less than 3%
        err_flat = data_array[i]*0.03

        error_array[i] = np.sqrt(error_array[i]**2 + error_bkg[i]**2 + err_wav**2 + err_psf**2 + err_flat**2)

        background[i] = sub_image.sum()
        if (data_array[i] < 0.).any():
            print(data_array[i])

    if display:
        plt.rcParams.update({'font.size': 10})
        convert_flux = headers[0]['photflam']
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
            ax.scatter(date_time[mask], background[mask]*convert_flux,
                    color=dict_filt[f],label="{0:s}".format(f))
        ax.errorbar(date_time, background*convert_flux,
                yerr=error_array[:,0,0]*convert_flux, fmt='+k',
                markersize=0, ecolor=c_filt)
        # Date handling
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlabel("Observation date and time")
        ax.set_ylabel(r"Flux [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")
        #ax.set_title("Background flux and error computed for each image")
        plt.legend()

        fig2, ax2 = plt.subplots(figsize=(10,10))
        data0 = data[0]*convert_flux
        instr = headers[0]['instrume']
        rootname = headers[0]['rootname']
        exptime = headers[0]['exptime']
        filt = headers[0]['filtnam1']
        #plots
        im = ax2.imshow(data0, vmin=data0.min(), vmax=data0.max(), origin='lower', cmap='gray')
        x, y, width, height, angle, color = rectangle[0]
        ax2.add_patch(Rectangle((x, y),width,height,edgecolor=color,fill=False))
        ax2.annotate(instr+":"+rootname, color='white', fontsize=10,
                xy=(0.02, 0.95), xycoords='axes fraction')
        ax2.annotate(filt, color='white', fontsize=14, xy=(0.02, 0.02),
                xycoords='axes fraction')
        ax2.annotate(str(exptime)+" s", color='white', fontsize=10, xy=(0.80, 0.02),
                xycoords='axes fraction')
        ax2.set(#title="Location of background computation.",
                xlabel='pixel offset',
                ylabel='pixel offset')

        fig2.subplots_adjust(hspace=0, wspace=0, right=0.85)
        cbar_ax = fig2.add_axes([0.9, 0.12, 0.02, 0.75])
        fig2.colorbar(im, cax=cbar_ax, label=r"Flux [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")

        if not(savename is None):
            fig.savefig(plots_folder+savename+"_background_flux.png",
                    bbox_inches='tight')
            fig2.savefig(plots_folder+savename+'_'+filt+'_background_location.png',
                    bbox_inches='tight')
            vmin = np.min(np.log10(data[data > 0.]))
            vmax = np.max(np.log10(data[data > 0.]))
            plot_obs(data, headers, vmin=data.min(), vmax=data.max(),
                    rectangle=rectangle,
                    savename=savename+"_background_location",
                    plots_folder=plots_folder)
        else:
            vmin = np.min(np.log10(data[data > 0.]))
            vmax = np.max(np.log10(data[data > 0.]))
            plot_obs(np.log10(data), headers, vmin=vmin, vmax=vmax,
                    rectangle=rectangle)

        plt.show()

    if return_background:
        return data_array, error_array, headers, np.array([error_array[i][0,0] for i in range(error_array.shape[0])])
    else:
        return data_array, error_array, headers


def rebin_array(data_array, error_array, headers, pxsize, scale,
        operation='sum', data_mask=None):
    """
    Homogeneously rebin a data array to get a new pixel size equal to pxsize
    where pxsize is given in arcsec.
    ----------
    Inputs:
    data_array, error_array : numpy.ndarray
        Arrays containing the images (2D float arrays) and their associated
        errors that will be rebinned.
    headers : header list
        List of headers corresponding to the images in data_array.
    pxsize : float
        Physical size of the pixel in arcseconds that should be obtain with
        the rebinning.
    scale : str, optional
        Scale units for the FWHM between 'pixel' and 'arcsec'.
        Defaults to 'pixel'.
    operation : str, optional
        Set the way original components of the matrix are put together
        between summing ('sum') and averaging ('average', 'avg', 'mean') them.
        Defaults to 'sum'.
    data_mask : numpy.ndarray, optional
        2D boolean array delimiting the data to work on.
        If None, will be initialized with a full true mask.
        Defaults to None.
    ----------
    Returns:
    rebinned_data, rebinned_error : numpy.ndarray
        Rebinned arrays containing the images and associated errors.
    rebinned_headers : header list
        Updated headers corresponding to the images in rebinned_data.
    Dxy : numpy.ndarray
        Array containing the rebinning factor in each direction of the image.
    data_mask : numpy.ndarray, optional
        Updated 2D boolean array delimiting the data to work on.
        Only returned if inputed data_mask is not None.
    """
    # Check that all images are from the same instrument
    ref_header = headers[0]
    instr = ref_header['instrume']
    same_instr = np.array([instr == header['instrume'] for header in headers]).all()
    if not same_instr:
        raise ValueError("All images in data_array are not from the same\
                instrument, cannot proceed.")
    if not instr in ['FOC']:
        raise ValueError("Cannot reduce images from {0:s} instrument\
                (yet)".format(instr))

    rebinned_data, rebinned_error, rebinned_headers = [], [], []
    Dxy = np.array([1, 1],dtype=int)

    # Routine for the FOC instrument
    if instr == 'FOC':
        HST_aper = 2400.    # HST aperture in mm
        for i, enum in enumerate(list(zip(data_array, error_array, headers))):
            image, error, header = enum
            # Get current pixel size
            w = WCS(header).deepcopy()

            # Compute binning ratio
            if scale.lower() in ['px', 'pixel']:
                Dxy = np.array([pxsize,]*2)
            elif scale.lower() in ['arcsec','arcseconds']:
                Dxy = np.floor(pxsize/np.abs(w.wcs.cdelt)/3600.).astype(int)
            elif scale.lower() in ['full','integrate']:
                Dxy = np.floor(image.shape).astype(int)
            else:
                raise ValueError("'{0:s}' invalid scale for binning.".format(scale))

            if (Dxy <= 1.).any():
                raise ValueError("Requested pixel size is below resolution.")
            new_shape = (image.shape//Dxy).astype(int)

            # Rebin data
            rebin_data = bin_ndarray(image, new_shape=new_shape,
                operation=operation)
            rebinned_data.append(rebin_data)

            # Propagate error
            rms_image = np.sqrt(bin_ndarray(image**2, new_shape=new_shape,
                operation='average'))
            sum_image = bin_ndarray(image, new_shape=new_shape,
                operation='sum')
            mask = sum_image > 0.
            new_error = np.zeros(rms_image.shape)
            if operation.lower() in ["mean", "average", "avg"]:
                new_error[mask] = np.sqrt(bin_ndarray(error**2*image,
                    new_shape=new_shape, operation='average')[mask]/sum_image[mask])
                #new_error[mask] = np.sqrt(bin_ndarray(error**2,
                #    new_shape=new_shape, operation='average')[mask])
            else:
                new_error[mask] = np.sqrt(bin_ndarray(error**2*image,
                    new_shape=new_shape, operation='sum')[mask]/sum_image[mask])
                #new_error[mask] = np.sqrt(bin_ndarray(error**2,
                #    new_shape=new_shape, operation='sum')[mask])
            rebinned_error.append(np.sqrt(rms_image**2 + new_error**2))

            # Update header
            w = w.slice((np.s_[::Dxy[0]], np.s_[::Dxy[1]]))
            header['NAXIS1'],header['NAXIS2'] = w.array_shape
            header.update(w.to_header())
            rebinned_headers.append(header)
        if not data_mask is None:
            data_mask = bin_ndarray(data_mask,new_shape=new_shape,operation='average') > 0.80

    rebinned_data = np.array(rebinned_data)
    rebinned_error = np.array(rebinned_error)

    if data_mask is None:
        return rebinned_data, rebinned_error, rebinned_headers, Dxy
    else:
        return rebinned_data, rebinned_error, rebinned_headers, Dxy, data_mask


def align_data(data_array, headers, error_array=None, upsample_factor=1.,
        ref_data=None, ref_center=None, return_shifts=False):
    """
    Align images in data_array using cross correlation, and rescale them to
    wider images able to contain any rotation of the reference image.
    All images in data_array must have the same shape.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array containing the data to align (2D float arrays).
    headers : header list
        List of headers corresponding to the images in data_array.
    error_array : numpy.ndarray, optional
        Array of images (2D floats, aligned and of the same shape) containing
        the error in each pixel of the observation images in data_array.
        If None, will be initialized to zeros.
        Defaults to None.
    upsample_factor : float, optional
        Oversampling factor for the cross-correlation, will allow sub-
        pixel alignement as small as one over the factor of a pixel.
        Defaults to one (no over-sampling).
    ref_data : numpy.ndarray, optional
        Reference image (2D float array) the data_array should be
        aligned to. If "None", the ref_data will be the first image
        of the data_array.
        Defaults to None.
    ref_center : numpy.ndarray, optional
        Array containing the coordinates of the center of the reference
        image or a string in 'max', 'flux', 'maxflux', 'max_flux'. If None,
        will fallback to the center of the image.
        Defaults to None.
    return_shifts : boolean, optional
        If False, calling the function will only return the array of
        rescaled images. If True, will also return the shifts and
        errors.
        Defaults to True.
    ----------
    Returns:
    rescaled : numpy.ndarray
        Array containing the aligned data from data_array, rescaled to wider
        image with margins of value 0.
    rescaled_error : numpy.ndarray
        Array containing the errors on the aligned images in the rescaled array.
    data_mask : numpy.ndarray
        2D boolean array delimiting the data to work on.
    shifts : numpy.ndarray
        Array containing the pixel shifts on the x and y directions from
        the reference image.
        Only returned if return_shifts is True.
    errors : numpy.ndarray
        Array containing the relative error computed on every shift value.
        Only returned if return_shifts is True.
    """
    if ref_data is None:
        # Define the reference to be the first image of the inputed array
        #if None have been specified
        ref_data = data_array[0]
    same = 1
    for array in data_array:
        # Check if all images have the same shape. If not, cross-correlation
        #cannot be computed.
        same *= (array.shape == ref_data.shape)
    if not same:
        raise ValueError("All images in data_array must have same shape as\
            ref_data")
    if error_array is None:
        _, error_array, headers, background = get_error(data_array, headers, return_background=True)
    else:
        _, _, headers, background = get_error(data_array, headers, return_background=True)

    # Crop out any null edges
    #(ref_data must be cropped as well)
    full_array = np.concatenate((data_array,[ref_data]),axis=0)
    full_headers = deepcopy(headers)
    full_headers.append(headers[0])
    err_array = np.concatenate((error_array,[np.zeros(ref_data.shape)]),axis=0)

    full_array, err_array, full_headers = crop_array(full_array, full_headers, err_array, step=5,
            inside=False)

    data_array, ref_data, headers = full_array[:-1], full_array[-1], full_headers[:-1]
    error_array = err_array[:-1]

    if ref_center is None:
        # Define the center of the reference image to be the center pixel
        #if None have been specified
        ref_center = (np.array(ref_data.shape)/2).astype(int)
    elif ref_center.lower() in ['max', 'flux', 'maxflux', 'max_flux']:
        # Define the center of the reference image to be the pixel of max flux.
        ref_center = np.unravel_index(np.argmax(ref_data),ref_data.shape)
    else:
        # Default to image center.
        ref_center = (np.array(ref_data.shape)/2).astype(int)

    # Create a rescaled null array that can contain any rotation of the
    #original image (and shifted images)
    shape = data_array.shape
    res_shape = int(np.ceil(np.sqrt(2.)*np.max(shape[1:])))
    rescaled_image = np.zeros((shape[0],res_shape,res_shape))
    rescaled_error = np.ones((shape[0],res_shape,res_shape))
    rescaled_mask = np.zeros((shape[0],res_shape,res_shape),dtype=bool)
    res_center = (np.array(rescaled_image.shape[1:])/2).astype(int)
    res_shift = res_center-ref_center
    res_mask = np.zeros((res_shape,res_shape),dtype=bool)
    res_mask[res_shift[0]:res_shift[0]+shape[1], res_shift[1]:res_shift[1]+shape[2]] = True

    shifts, errors = [], []
    for i,image in enumerate(data_array):
        # Initialize rescaled images to background values
        rescaled_error[i] *= background[i]
        # Get shifts and error by cross-correlation to ref_data
        shift, error, phase_diff = phase_cross_correlation(ref_data/ref_data.max(), image/image.max(),
                upsample_factor=upsample_factor)
        # Rescale image to requested output
        rescaled_image[i,res_shift[0]:res_shift[0]+shape[1],
                res_shift[1]:res_shift[1]+shape[2]] = deepcopy(image)
        rescaled_error[i,res_shift[0]:res_shift[0]+shape[1],
                res_shift[1]:res_shift[1]+shape[2]] = deepcopy(error_array[i])
        # Shift images to align
        rescaled_image[i] = sc_shift(rescaled_image[i], shift, order=1, cval=0.)
        rescaled_error[i] = sc_shift(rescaled_error[i], shift, order=1, cval=background[i])
        curr_mask = sc_shift(res_mask, shift, order=1, cval=False)
        mask_vertex = clean_ROI(curr_mask)
        rescaled_mask[i,mask_vertex[2]:mask_vertex[3],mask_vertex[0]:mask_vertex[1]] = True

        rescaled_image[i][rescaled_image[i] < 0.] = 0.
        rescaled_image[i][(1-rescaled_mask[i]).astype(bool)] = 0.

        # Uncertainties from shifting
        prec_shift = np.array([1.,1.])/upsample_factor
        shifted_image = sc_shift(rescaled_image[i], prec_shift, cval=0.)
        error_shift = np.abs(rescaled_image[i] - shifted_image)/2.
        #sum quadratically the errors
        rescaled_error[i] = np.sqrt(rescaled_error[i]**2 + error_shift**2)

        shifts.append(shift)
        errors.append(error)

    shifts = np.array(shifts)
    errors = np.array(errors)

    # Update headers CRPIX value
    headers_wcs = [deepcopy(WCS(header)) for header in headers]
    new_crpix = np.array([wcs.wcs.crpix for wcs in headers_wcs]) + shifts[:,::-1] + res_shift[::-1]
    for i in range(len(headers_wcs)):
        headers_wcs[i].wcs.crpix = new_crpix[0]
        headers[i].update(headers_wcs[i].to_header())

    data_mask = rescaled_mask.all(axis=0)
    data_array, error_array, data_mask, headers = crop_array(rescaled_image, headers, rescaled_error, data_mask)

    if return_shifts:
        return data_array, error_array, headers, data_mask, shifts, errors
    else:
        return data_array, error_array, headers, data_mask


def smooth_data(data_array, error_array, data_mask, headers, FWHM=1.,
        scale='pixel', smoothing='gaussian'):
    """
    Smooth a data_array using selected function.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array containing the data to smooth (2D float arrays).
    error_array : numpy.ndarray
        Array of images (2D floats, aligned and of the same shape) containing
        the error in each pixel of the observation images in data_array.
    data_mask : numpy.ndarray
        2D boolean array delimiting the data to work on.
    headers : header list
        List of headers corresponding to the images in data_array.
    FWHM : float, optional
        Full Width at Half Maximum for desired smoothing in 'scale' units.
        Defaults to 1.
    scale : str, optional
        Scale units for the FWHM between 'pixel' and 'arcsec'.
        Defaults to 'pixel'.
    smoothing : str, optional
        Smoothing algorithm to be used on the input data array.
        -'combine','combining' use the N images combining algorithm with
          weighted pixels (inverse square of errors).
        -'gauss','gaussian' convolve any input image with a gaussian of
          standard deviation stdev = FWHM/(2*sqrt(2*log(2))).
        Defaults to 'gaussian'. Won't be used if FWHM is None.
    ----------
    Returns:
    smoothed_array : numpy.ndarray
        Array containing the smoothed images.
    error_array : numpy.ndarray
        Array containing the error images corresponding to the images in
        smoothed_array.
    """
    # If chosen FWHM scale is 'arcsec', compute FWHM in pixel scale
    if scale.lower() in ['arcsec','arcseconds']:
        pxsize = np.zeros((data_array.shape[0],2))
        for i,header in enumerate(headers):
            # Get current pixel size
            w = WCS(header).deepcopy()
            pxsize[i] = np.round(w.wcs.cdelt*3600.,4)
        if (pxsize != pxsize[0]).any():
            raise ValueError("Not all images in array have same pixel size")
        FWHM /= pxsize[0].min()

    # Define gaussian stdev
    stdev = FWHM/(2.*np.sqrt(2.*np.log(2)))
    fmax = np.finfo(np.double).max

    if smoothing.lower() in ['combine','combining']:
        # Smooth using N images combination algorithm
        # Weight array
        weight = 1./error_array**2
        # Prepare pixel distance matrix
        xx, yy = np.indices(data_array[0].shape)
        # Initialize smoothed image and error arrays
        smoothed = np.zeros(data_array[0].shape)
        error = np.zeros(data_array[0].shape)

        # Combination smoothing algorithm
        for r in range(smoothed.shape[0]):
            for c in range(smoothed.shape[1]):
                # Compute distance from current pixel
                dist_rc = np.where(data_mask, np.sqrt((r-xx)**2+(c-yy)**2), fmax)
                # Catch expected "OverflowWarning" as we overflow values that are not in the image
                with warnings.catch_warnings(record=True) as w:
                    g_rc = np.array([np.exp(-0.5*(dist_rc/stdev)**2),]*data_array.shape[0])
                    # Apply weighted combination
                    smoothed[r,c] = np.where(data_mask[r,c], np.sum(data_array*weight*g_rc)/np.sum(weight*g_rc), 0.)
                    error[r,c] = np.where(data_mask[r,c], np.sqrt(np.sum(weight*g_rc**2))/np.sum(weight*g_rc), 0.)

        # Nan handling
        error[np.isnan(smoothed)] = 0.
        smoothed[np.isnan(smoothed)] = 0.
        error[np.isnan(error)] = 0.

    elif smoothing.lower() in ['weight_gauss','weighted_gaussian','gauss','gaussian']:
        # Convolution with gaussian function
        smoothed = np.zeros(data_array.shape)
        error = np.zeros(error_array.shape)
        for i,image in enumerate(data_array):
            x, y = np.meshgrid(np.arange(-image.shape[1]/2,image.shape[1]/2),np.arange(-image.shape[0]/2,image.shape[0]/2))
            weights = np.ones(error_array[i].shape)
            if smoothing.lower()[:6] in ['weight']:
                weights = 1./error_array[i]**2
                #weights /= weights.max()
            kernel = gaussian2d(x, y, stdev)
            kernel /= kernel.sum()
            smoothed[i] = convolve2d(image*weights/image.sum(),kernel,'same')*image.sum()
            error[i] = np.sqrt(convolve2d((error_array[i]/error_array[i].sum())**2*weights**2,kernel**2,'same'))*error_array[i].sum()

            # Nan handling
            error[i][np.isnan(smoothed[i])] = 0.
            smoothed[i][np.isnan(smoothed[i])] = 0.
            error[i][np.isnan(error[i])] = 0.

    else:
        raise ValueError("{} is not a valid smoothing option".format(smoothing))

    return smoothed, error


def polarizer_avg(data_array, error_array, data_mask, headers, FWHM=None,
        scale='pixel', smoothing='gaussian'):
    """
    Make the average image from a single polarizer for a given instrument.
    -----------
    Inputs:
    data_array : numpy.ndarray
        Array of images (2D floats, aligned and of the same shape) of a
        single observation with multiple polarizers of an instrument.
    error_array : numpy.ndarray
        Array of images (2D floats, aligned and of the same shape) containing
        the error in each pixel of the observation images in data_array.
    data_mask : numpy.ndarray
        2D boolean array delimiting the data to work on.
    headers : header list
        List of headers corresponding to the images in data_array.
    FWHM : float, optional
        Full Width at Half Maximum of the detector for smoothing of the
        data on each polarizer filter in 'scale' units. If None, no smoothing
        is done.
        Defaults to None.
    scale : str, optional
        Scale units for the FWHM between 'pixel' and 'arcsec'.
        Defaults to 'pixel'.
    smoothing : str, optional
        Smoothing algorithm to be used on the input data array.
        -'combine','combining' use the N images combining algorithm with
          weighted pixels (inverse square of errors).
        -'gaussian' convolve any input image with a gaussian of standard
          deviation stdev = FWHM/(2*sqrt(2*log(2))).
        Defaults to 'gaussian'. Won't be used if FWHM is None.
    ----------
    Returns:
    polarizer_array : numpy.ndarray
        Array of images averaged on each polarizer filter of the instrument
    polarizer_cov : numpy.ndarray
        Covariance matrix between the polarizer images in polarizer_array
    """
    # Check that all images are from the same instrument
    instr = headers[0]['instrume']
    same_instr = np.array([instr == header['instrume'] for header in headers]).all()
    if not same_instr:
        raise ValueError("All images in data_array are not from the same\
                instrument, cannot proceed.")
    if not instr in ['FOC']:
        raise ValueError("Cannot reduce images from {0:s} instrument\
                (yet)".format(instr))

    # Routine for the FOC instrument
    if instr == 'FOC':
        # Sort images by polarizer filter : can be 0deg, 60deg, 120deg for the FOC
        is_pol0 = np.array([header['filtnam1']=='POL0' for header in headers])
        if (1-is_pol0).all():
            print("Warning : no image for POL0 of FOC found, averaged data\
                    will be NAN")
        is_pol60 = np.array([header['filtnam1']=='POL60' for header in headers])
        if (1-is_pol60).all():
            print("Warning : no image for POL60 of FOC found, averaged data\
                    will be NAN")
        is_pol120 = np.array([header['filtnam1']=='POL120' for header in headers])
        if (1-is_pol120).all():
            print("Warning : no image for POL120 of FOC found, averaged data\
                    will be NAN")
        # Put each polarizer images in separate arrays
        pol0_array = data_array[is_pol0]
        pol60_array = data_array[is_pol60]
        pol120_array = data_array[is_pol120]

        err0_array = error_array[is_pol0]
        err60_array = error_array[is_pol60]
        err120_array = error_array[is_pol120]

        headers0 = [header for header in headers if header['filtnam1']=='POL0']
        headers60 = [header for header in headers if header['filtnam1']=='POL60']
        headers120 = [header for header in headers if header['filtnam1']=='POL120']

        if not(FWHM is None) and (smoothing.lower() in ['combine','combining']):
            # Smooth by combining each polarizer images
            pol0, err0 = smooth_data(pol0_array, err0_array, data_mask, headers0,
                    FWHM=FWHM, scale=scale, smoothing=smoothing)
            pol60, err60 = smooth_data(pol60_array, err60_array, data_mask, headers60,
                    FWHM=FWHM, scale=scale, smoothing=smoothing)
            pol120, err120 = smooth_data(pol120_array, err120_array, data_mask, headers120,
                    FWHM=FWHM, scale=scale, smoothing=smoothing)

        else:
            # Sum on each polarization filter.
            pol0_t = np.sum([header['exptime'] for header in headers0])
            pol60_t = np.sum([header['exptime'] for header in headers60])
            pol120_t = np.sum([header['exptime'] for header in headers120])

            for i in range(pol0_array.shape[0]):
                pol0_array[i] *= headers0[i]['exptime']/pol0_t
                err0_array[i] *= headers0[i]['exptime']/pol0_t
            for i in range(pol60_array.shape[0]):
                pol60_array[i] *= headers60[i]['exptime']/pol60_t
                err60_array[i] *= headers60[i]['exptime']/pol60_t
            for i in range(pol120_array.shape[0]):
                pol120_array[i] *= headers120[i]['exptime']/pol120_t
                err120_array[i] *= headers120[i]['exptime']/pol120_t

            pol0 = pol0_array.sum(axis=0)
            pol60 = pol60_array.sum(axis=0)
            pol120 = pol120_array.sum(axis=0)
            pol_array = np.array([pol0, pol60, pol120])

            # Propagate uncertainties quadratically
            err0 = np.sqrt(np.sum(err0_array**2,axis=0))
            err60 = np.sqrt(np.sum(err60_array**2,axis=0))
            err120 = np.sqrt(np.sum(err120_array**2,axis=0))
            polerr_array = np.array([err0, err60, err120])

            # Update headers
            for header in headers:
                if header['filtnam1']=='POL0':
                    list_head = headers0
                elif header['filtnam1']=='POL60':
                    list_head = headers60
                else:
                    list_head = headers120
                header['exptime'] = np.sum([head['exptime'] for head in list_head])/len(list_head)
            headers_array = [headers0[0], headers60[0], headers120[0]]

            if not(FWHM is None) and (smoothing.lower() in ['gaussian','gauss']):
                # Smooth by convoluting with a gaussian each polX image.
                pol_array, polerr_array = smooth_data(pol_array, polerr_array,
                        data_mask, headers_array, FWHM=FWHM, scale=scale)
                pol0, pol60, pol120 = pol_array
                err0, err60, err120 = polerr_array

        # Get image shape
        shape = pol0.shape

        # Construct the polarizer array
        polarizer_array = np.zeros((3,shape[0],shape[1]))
        polarizer_array[0] = pol0
        polarizer_array[1] = pol60
        polarizer_array[2] = pol120

        # Define the covariance matrix for the polarizer images
        #We assume cross terms are null
        polarizer_cov = np.zeros((3,3,shape[0],shape[1]))
        polarizer_cov[0,0] = err0**2
        polarizer_cov[1,1] = err60**2
        polarizer_cov[2,2] = err120**2

    return polarizer_array, polarizer_cov


def compute_Stokes(data_array, error_array, data_mask, headers,
        FWHM=None, scale='pixel', smoothing='gaussian_after'):
    """
    Compute the Stokes parameters I, Q and U for a given data_set
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array of images (2D floats, aligned and of the same shape) of a
        single observation with multiple polarizers of an instrument.
    error_array : numpy.ndarray
        Array of images (2D floats, aligned and of the same shape) containing
        the error in each pixel of the observation images in data_array.
    data_mask : numpy.ndarray
        2D boolean array delimiting the data to work on.
    headers : header list
        List of headers corresponding to the images in data_array.
    FWHM : float, optional
        Full Width at Half Maximum of the detector for smoothing of the
        data on each polarizer filter in scale units. If None, no smoothing
        is done.
        Defaults to None.
    scale : str, optional
        Scale units for the FWHM between 'pixel' and 'arcsec'.
        Defaults to 'pixel'.
    smoothing : str, optional
        Smoothing algorithm to be used on the input data array.
        -'combine','combining' use the N images combining algorithm with
          weighted pixels (inverse square of errors).
        -'gaussian' convolve any input image with a gaussian of standard
          deviation stdev = FWHM/(2*sqrt(2*log(2))).
        -'gaussian_after' convolve output Stokes I/Q/U with a gaussian of
          standard deviation stdev = FWHM/(2*sqrt(2*log(2))).
        Defaults to 'gaussian_after'. Won't be used if FWHM is None.
    ----------
    Returns:
    I_stokes : numpy.ndarray
        Image (2D floats) containing the Stokes parameters accounting for
        total intensity
    Q_stokes : numpy.ndarray
        Image (2D floats) containing the Stokes parameters accounting for
        vertical/horizontal linear polarization intensity
    U_stokes : numpy.ndarray
        Image (2D floats) containing the Stokes parameters accounting for
        +45/-45deg linear polarization intensity
    Stokes_cov : numpy.ndarray
        Covariance matrix of the Stokes parameters I, Q, U.
    """
    # Check that all images are from the same instrument
    instr = headers[0]['instrume']
    same_instr = np.array([instr == header['instrume'] for header in headers]).all()
    if not same_instr:
        raise ValueError("All images in data_array are not from the same\
                instrument, cannot proceed.")
    if not instr in ['FOC']:
        raise ValueError("Cannot reduce images from {0:s} instrument\
                (yet)".format(instr))

    # Routine for the FOC instrument
    if instr == 'FOC':
        # Get image from each polarizer and covariance matrix
        pol_array, pol_cov = polarizer_avg(data_array, error_array, data_mask,
                headers, FWHM=FWHM, scale=scale, smoothing=smoothing)
        pol0, pol60, pol120 = pol_array

        if (pol0 < 0.).any() or (pol60 < 0.).any() or (pol120 < 0.).any():
            print("WARNING : Negative value in polarizer array.")

        # Stokes parameters
        #transmittance corrected
        transmit = np.ones((3,))   #will be filter dependant
        filt2, filt3, filt4 = headers[0]['filtnam2'], headers[0]['filtnam3'], headers[0]['filtnam4']
        same_filt2 = np.array([filt2 == header['filtnam2'] for header in headers]).all()
        same_filt3 = np.array([filt3 == header['filtnam3'] for header in headers]).all()
        same_filt4 = np.array([filt4 == header['filtnam4'] for header in headers]).all()
        if (same_filt2 and same_filt3 and same_filt4):
            transmit2, transmit3, transmit4 = trans2[filt2.lower()], trans3[filt3.lower()], trans4[filt4.lower()]
        else:
            print("WARNING : All images in data_array are not from the same \
                    band filter, the limiting transmittance will be taken.")
            transmit2 = np.min([trans2[header['filtnam2'].lower()] for header in headers])
            transmit3 = np.min([trans3[header['filtnam3'].lower()] for header in headers])
            transmit4 = np.min([trans4[header['filtnam4'].lower()] for header in headers])
        transmit *= transmit2*transmit3*transmit4

        pol_eff = np.array([pol_efficiency['pol0'], pol_efficiency['pol60'], pol_efficiency['pol120']])

        # Orientation and error for each polarizer
        fmax = np.finfo(np.float64).max
        pol_flux = np.array([pol0, pol60, pol120])

        coeff_stokes = np.zeros((3,3))
        # Coefficients linking each polarizer flux to each Stokes parameter
        for i in range(3):
            coeff_stokes[0,i] = pol_eff[(i+1)%3]*pol_eff[(i+2)%3]*np.sin(-2.*theta[(i+1)%3]+2.*theta[(i+2)%3])*2./transmit[i]
            coeff_stokes[1,i] = (-pol_eff[(i+1)%3]*np.sin(2.*theta[(i+1)%3]) + pol_eff[(i+2)%3]*np.sin(2.*theta[(i+2)%3]))*2./transmit[i]
            coeff_stokes[2,i] = (pol_eff[(i+1)%3]*np.cos(2.*theta[(i+1)%3]) - pol_eff[(i+2)%3]*np.cos(2.*theta[(i+2)%3]))*2./transmit[i]

        # Normalization parameter for Stokes parameters computation
        A = (coeff_stokes[0,:]*transmit/2.).sum()
        coeff_stokes = coeff_stokes/A
        I_stokes = np.zeros(pol_array[0].shape)
        Q_stokes = np.zeros(pol_array[0].shape)
        U_stokes = np.zeros(pol_array[0].shape)
        Stokes_cov = np.zeros((3,3,I_stokes.shape[0],I_stokes.shape[1]))

        for i in range(I_stokes.shape[0]):
            for j in range(I_stokes.shape[1]):
                I_stokes[i,j], Q_stokes[i,j], U_stokes[i,j] = np.dot(coeff_stokes, pol_flux[:,i,j]).T
                Stokes_cov[:,:,i,j] = np.dot(coeff_stokes, np.dot(pol_cov[:,:,i,j], coeff_stokes.T))

        mask = (Q_stokes**2 + U_stokes**2) > I_stokes**2
        if mask.any():
            print("WARNING : found {0:d} pixels for which I_pol > I_stokes".format(I_stokes[mask].size))

        # Compute the derivative of each Stokes parameter with respect to the polarizer orientation
        dI_dtheta1 = 2.*pol_eff[0]/A*(pol_eff[2]*np.cos(-2.*theta[2]+2.*theta[0])*(pol_flux[1]-I_stokes) - pol_eff[1]*np.cos(-2.*theta[0]+2.*theta[1])*(pol_flux[2]-I_stokes))
        dI_dtheta2 = 2.*pol_eff[1]/A*(pol_eff[0]*np.cos(-2.*theta[0]+2.*theta[1])*(pol_flux[2]-I_stokes) - pol_eff[2]*np.cos(-2.*theta[1]+2.*theta[2])*(pol_flux[0]-I_stokes))
        dI_dtheta3 = 2.*pol_eff[2]/A*(pol_eff[1]*np.cos(-2.*theta[1]+2.*theta[2])*(pol_flux[0]-I_stokes) - pol_eff[0]*np.cos(-2.*theta[2]+2.*theta[0])*(pol_flux[1]-I_stokes))
        dI_dtheta = np.array([dI_dtheta1, dI_dtheta2, dI_dtheta3])

        dQ_dtheta1 = 2.*pol_eff[0]/A*(np.cos(2.*theta[0])*(pol_flux[1]-pol_flux[2]) - (pol_eff[2]*np.cos(-2.*theta[2]+2.*theta[0]) - pol_eff[1]*np.cos(-2.*theta[0]+2.*theta[1]))*Q_stokes)
        dQ_dtheta2 = 2.*pol_eff[1]/A*(np.cos(2.*theta[1])*(pol_flux[2]-pol_flux[0]) - (pol_eff[0]*np.cos(-2.*theta[0]+2.*theta[1]) - pol_eff[2]*np.cos(-2.*theta[1]+2.*theta[2]))*Q_stokes)
        dQ_dtheta3 = 2.*pol_eff[2]/A*(np.cos(2.*theta[2])*(pol_flux[0]-pol_flux[1]) - (pol_eff[1]*np.cos(-2.*theta[1]+2.*theta[2]) - pol_eff[0]*np.cos(-2.*theta[2]+2.*theta[0]))*Q_stokes)
        dQ_dtheta = np.array([dQ_dtheta1, dQ_dtheta2, dQ_dtheta3])

        dU_dtheta1 = 2.*pol_eff[0]/A*(np.sin(2.*theta[0])*(pol_flux[1]-pol_flux[2]) - (pol_eff[2]*np.cos(-2.*theta[2]+2.*theta[0]) - pol_eff[1]*np.cos(-2.*theta[0]+2.*theta[1]))*U_stokes)
        dU_dtheta2 = 2.*pol_eff[1]/A*(np.sin(2.*theta[1])*(pol_flux[2]-pol_flux[0]) - (pol_eff[0]*np.cos(-2.*theta[0]+2.*theta[1]) - pol_eff[2]*np.cos(-2.*theta[1]+2.*theta[2]))*U_stokes)
        dU_dtheta3 = 2.*pol_eff[2]/A*(np.sin(2.*theta[2])*(pol_flux[0]-pol_flux[1]) - (pol_eff[1]*np.cos(-2.*theta[1]+2.*theta[2]) - pol_eff[0]*np.cos(-2.*theta[2]+2.*theta[0]))*U_stokes)
        dU_dtheta = np.array([dU_dtheta1, dU_dtheta2, dU_dtheta3])

        # Compute the uncertainty associated with the polarizers' orientation (see Kishimoto 1999)
        s_I2_axis = np.sum([dI_dtheta[i]**2 * sigma_theta[i]**2 for i in range(len(sigma_theta))],axis=0)
        s_Q2_axis = np.sum([dQ_dtheta[i]**2 * sigma_theta[i]**2 for i in range(len(sigma_theta))],axis=0)
        s_U2_axis = np.sum([dU_dtheta[i]**2 * sigma_theta[i]**2 for i in range(len(sigma_theta))],axis=0)

        # Add quadratically the uncertainty to the Stokes covariance matrix
        Stokes_cov[0,0] += s_I2_axis
        Stokes_cov[1,1] += s_Q2_axis
        Stokes_cov[2,2] += s_U2_axis

        if not(FWHM is None) and (smoothing.lower() in ['weighted_gaussian_after','weight_gauss_after','gaussian_after','gauss_after']):
            smoothing = smoothing.lower()[:-6]
            Stokes_array = np.array([I_stokes, Q_stokes, U_stokes])
            Stokes_error = np.array([np.sqrt(Stokes_cov[i,i]) for i in range(3)])
            Stokes_headers = headers[0:3]

            Stokes_array, Stokes_error = smooth_data(Stokes_array, Stokes_error, data_mask,
                    headers=Stokes_headers, FWHM=FWHM, scale=scale, smoothing=smoothing)

            I_stokes, Q_stokes, U_stokes = Stokes_array
            Stokes_cov[0,0], Stokes_cov[1,1], Stokes_cov[2,2] = Stokes_error**2

        #Compute integrated values for P, PA before any rotation
        mask = np.logical_and(data_mask.astype(bool), (I_stokes > 0.))
        n_pix = I_stokes[mask].size
        I_diluted = I_stokes[mask].sum()
        Q_diluted = Q_stokes[mask].sum()
        U_diluted = U_stokes[mask].sum()
        I_diluted_err = np.sqrt(np.sum(Stokes_cov[0,0][mask]))
        Q_diluted_err = np.sqrt(np.sum(Stokes_cov[1,1][mask]))
        U_diluted_err = np.sqrt(np.sum(Stokes_cov[2,2][mask]))
        IQ_diluted_err = np.sqrt(np.sum(Stokes_cov[0,1][mask]**2))
        IU_diluted_err = np.sqrt(np.sum(Stokes_cov[0,2][mask]**2))
        QU_diluted_err = np.sqrt(np.sum(Stokes_cov[1,2][mask]**2))

        P_diluted = np.sqrt(Q_diluted**2+U_diluted**2)/I_diluted
        P_diluted_err = (1./I_diluted)*np.sqrt((Q_diluted**2*Q_diluted_err**2 + U_diluted**2*U_diluted_err**2 + 2.*Q_diluted*U_diluted*QU_diluted_err)/(Q_diluted**2 + U_diluted**2) + ((Q_diluted/I_diluted)**2 + (U_diluted/I_diluted)**2)*I_diluted_err**2 - 2.*(Q_diluted/I_diluted)*IQ_diluted_err - 2.*(U_diluted/I_diluted)*IU_diluted_err)

        PA_diluted = princ_angle((90./np.pi)*np.arctan2(U_diluted,Q_diluted))
        PA_diluted_err = (90./(np.pi*(Q_diluted**2 + U_diluted**2)))*np.sqrt(U_diluted**2*Q_diluted_err**2 + Q_diluted**2*U_diluted_err**2 - 2.*Q_diluted*U_diluted*QU_diluted_err)

        for header in headers:
            header['P_int'] = (P_diluted, 'Integrated polarization degree')
            header['P_int_err'] = (P_diluted_err, 'Integrated polarization degree error')
            header['PA_int'] = (PA_diluted, 'Integrated polarization angle')
            header['PA_int_err'] = (PA_diluted_err, 'Integrated polarization angle error')

    return I_stokes, Q_stokes, U_stokes, Stokes_cov


def compute_pol(I_stokes, Q_stokes, U_stokes, Stokes_cov, headers):
    """
    Compute the polarization degree (in %) and angle (in deg) and their
    respective errors from given Stokes parameters.
    ----------
    Inputs:
    I_stokes : numpy.ndarray
        Image (2D floats) containing the Stokes parameters accounting for
        total intensity
    Q_stokes : numpy.ndarray
        Image (2D floats) containing the Stokes parameters accounting for
        vertical/horizontal linear polarization intensity
    U_stokes : numpy.ndarray
        Image (2D floats) containing the Stokes parameters accounting for
        +45/-45deg linear polarization intensity
    Stokes_cov : numpy.ndarray
        Covariance matrix of the Stokes parameters I, Q, U.
    headers : header list
        List of headers corresponding to the images in data_array.
    ----------
    Returns:
    P : numpy.ndarray
        Image (2D floats) containing the polarization degree (in %).
    debiased_P : numpy.ndarray
        Image (2D floats) containing the debiased polarization degree (in %).
    s_P : numpy.ndarray
        Image (2D floats) containing the error on the polarization degree.
    s_P_P : numpy.ndarray
        Image (2D floats) containing the Poisson noise error on the
        polarization degree.
    PA : numpy.ndarray
        Image (2D floats) containing the polarization angle.
    s_PA : numpy.ndarray
        Image (2D floats) containing the error on the polarization angle.
    s_PA_P : numpy.ndarray
        Image (2D floats) containing the Poisson noise error on the
        polarization angle.
    new_headers : header list
        Updated list of headers corresponding to the reduced images accounting
        for the new orientation angle.
    """
    #Polarization degree and angle computation
    mask = I_stokes>0.
    I_pol = np.zeros(I_stokes.shape)
    I_pol[mask] = np.sqrt(Q_stokes[mask]**2 + U_stokes[mask]**2)
    P = np.zeros(I_stokes.shape)
    P[mask] = I_pol[mask]/I_stokes[mask]
    PA = np.zeros(I_stokes.shape)
    PA[mask] = (90./np.pi)*np.arctan2(U_stokes[mask],Q_stokes[mask])

    if (P>1).any():
        print("WARNING : found {0:d} pixels for which P > 1".format(P[P>1.].size))

    #Associated errors
    fmax = np.finfo(np.float64).max
    s_P = np.ones(I_stokes.shape)*fmax
    s_PA = np.ones(I_stokes.shape)*fmax

    # Propagate previously computed errors
    s_P[mask] = (1/I_stokes[mask])*np.sqrt((Q_stokes[mask]**2*Stokes_cov[1,1][mask] + U_stokes[mask]**2*Stokes_cov[2,2][mask] + 2.*Q_stokes[mask]*U_stokes[mask]*Stokes_cov[1,2][mask])/(Q_stokes[mask]**2 + U_stokes[mask]**2) + ((Q_stokes[mask]/I_stokes[mask])**2 + (U_stokes[mask]/I_stokes[mask])**2)*Stokes_cov[0,0][mask] - 2.*(Q_stokes[mask]/I_stokes[mask])*Stokes_cov[0,1][mask] - 2.*(U_stokes[mask]/I_stokes[mask])*Stokes_cov[0,2][mask])
    s_PA[mask] = (90./(np.pi*(Q_stokes[mask]**2 + U_stokes[mask]**2)))*np.sqrt(U_stokes[mask]**2*Stokes_cov[1,1][mask] + Q_stokes[mask]**2*Stokes_cov[2,2][mask] - 2.*Q_stokes[mask]*U_stokes[mask]*Stokes_cov[1,2][mask])
    s_P[np.isnan(s_P)] = fmax
    s_PA[np.isnan(s_PA)] = fmax

    # Catch expected "OverflowWarning" as wrong pixel have an overflowing error
    with warnings.catch_warnings(record=True) as w:
        mask2 = P**2 >= s_P**2
    debiased_P = np.zeros(I_stokes.shape)
    debiased_P[mask2] = np.sqrt(P[mask2]**2 - s_P[mask2]**2)

    if (debiased_P>1.).any():
        print("WARNING : found {0:d} pixels for which debiased_P > 100%".format(debiased_P[debiased_P>1.].size))

    #Compute the total exposure time so that
    #I_stokes*exp_tot = N_tot the total number of events
    exp_tot = np.array([header['exptime'] for header in headers]).sum()
    #print("Total exposure time : {} sec".format(exp_tot))
    N_obs = I_stokes*exp_tot

    #Errors on P, PA supposing Poisson noise
    s_P_P = np.ones(I_stokes.shape)*fmax
    s_P_P[mask] = np.sqrt(2.)/np.sqrt(N_obs[mask])*100.
    s_PA_P = np.ones(I_stokes.shape)*fmax
    s_PA_P[mask2] = s_P_P[mask2]/(2.*P[mask2])*180./np.pi

    # Nan handling :
    P[np.isnan(P)] = 0.
    s_P[np.isnan(s_P)] = fmax
    s_PA[np.isnan(s_PA)] = fmax
    debiased_P[np.isnan(debiased_P)] = 0.
    s_P_P[np.isnan(s_P_P)] = fmax
    s_PA_P[np.isnan(s_PA_P)] = fmax

    return P, debiased_P, s_P, s_P_P, PA, s_PA, s_PA_P


def rotate_Stokes(I_stokes, Q_stokes, U_stokes, Stokes_cov, data_mask, headers,
                ang=None, SNRi_cut=None):
    """
    Use scipy.ndimage.rotate to rotate I_stokes to an angle, and a rotation
    matrix to rotate Q, U of a given angle in degrees and update header
    orientation keyword.
    ----------
    Inputs:
    I_stokes : numpy.ndarray
        Image (2D floats) containing the Stokes parameters accounting for
        total intensity
    Q_stokes : numpy.ndarray
        Image (2D floats) containing the Stokes parameters accounting for
        vertical/horizontal linear polarization intensity
    U_stokes : numpy.ndarray
        Image (2D floats) containing the Stokes parameters accounting for
        +45/-45deg linear polarization intensity
    Stokes_cov : numpy.ndarray
        Covariance matrix of the Stokes parameters I, Q, U.
    data_mask : numpy.ndarray
        2D boolean array delimiting the data to work on.
    headers : header list
        List of headers corresponding to the reduced images.
    ang : float, optional
        Rotation angle (in degrees) that should be applied to the Stokes
        parameters. If None, will rotate to have North up.
        Defaults to None.
    SNRi_cut : float, optional
        Cut that should be applied to the signal-to-noise ratio on I.
        Any SNR < SNRi_cut won't be displayed. If None, cut won't be applied.
        Defaults to None.
    ----------
    Returns:
    new_I_stokes : numpy.ndarray
        Rotated mage (2D floats) containing the rotated Stokes parameters
        accounting for total intensity
    new_Q_stokes : numpy.ndarray
        Rotated mage (2D floats) containing the rotated Stokes parameters
        accounting for vertical/horizontal linear polarization intensity
    new_U_stokes : numpy.ndarray
        Rotated image (2D floats) containing the rotated Stokes parameters
        accounting for +45/-45deg linear polarization intensity.
    new_Stokes_cov : numpy.ndarray
        Updated covariance matrix of the Stokes parameters I, Q, U.
    new_headers : header list
        Updated list of headers corresponding to the reduced images accounting
        for the new orientation angle.
    new_data_mask : numpy.ndarray
        Updated 2D boolean array delimiting the data to work on.
    """
    #Apply cuts
    if not(SNRi_cut is None):
        SNRi = I_stokes/np.sqrt(Stokes_cov[0,0])
        mask = SNRi < SNRi_cut
        eps = 1e-5
        for i in range(I_stokes.shape[0]):
            for j in range(I_stokes.shape[1]):
                if mask[i,j]:
                    I_stokes[i,j] = eps*np.sqrt(Stokes_cov[0,0][i,j])
                    Q_stokes[i,j] = eps*np.sqrt(Stokes_cov[1,1][i,j])
                    U_stokes[i,j] = eps*np.sqrt(Stokes_cov[2,2][i,j])

    #Rotate I_stokes, Q_stokes, U_stokes using rotation matrix
    if ang is None:
        ang = np.zeros((len(headers),))
        for i,head in enumerate(headers):
            ang[i] = -head['orientat']
        ang = ang.mean()
    alpha = ang*np.pi/180.
    mrot = np.array([[1., 0., 0.],
                    [0., np.cos(2.*alpha), np.sin(2.*alpha)],
                    [0, -np.sin(2.*alpha), np.cos(2.*alpha)]])

    old_center = np.array(I_stokes.shape)/2
    shape = np.fix(np.array(I_stokes.shape)*np.sqrt(2.5)).astype(int)
    new_center = np.array(shape)/2

    I_stokes = zeropad(I_stokes, shape)
    Q_stokes = zeropad(Q_stokes, shape)
    U_stokes = zeropad(U_stokes, shape)
    data_mask = zeropad(data_mask, shape)
    Stokes_cov = zeropad(Stokes_cov, [*Stokes_cov.shape[:-2],*shape])
    new_I_stokes = np.zeros(shape)
    new_Q_stokes = np.zeros(shape)
    new_U_stokes = np.zeros(shape)
    new_Stokes_cov = np.zeros((*Stokes_cov.shape[:-2],*shape))

    for i in range(shape[0]):
        for j in range(shape[1]):
            new_I_stokes[i,j], new_Q_stokes[i,j], new_U_stokes[i,j] = np.dot(mrot, np.array([I_stokes[i,j], Q_stokes[i,j], U_stokes[i,j]])).T
            new_Stokes_cov[:,:,i,j] = np.dot(mrot, np.dot(Stokes_cov[:,:,i,j], mrot.T))

    #Rotate original images using scipy.ndimage.rotate
    new_I_stokes = sc_rotate(new_I_stokes, ang, order=1, reshape=False, cval=0.)
    new_Q_stokes = sc_rotate(new_Q_stokes, ang, order=1, reshape=False, cval=0.)
    new_U_stokes = sc_rotate(new_U_stokes, ang, order=1, reshape=False, cval=0.)
    new_data_mask = sc_rotate(data_mask.astype(float)*10., ang, order=1, reshape=False, cval=0.)
    new_data_mask[new_data_mask < 2] = 0.
    new_data_mask = new_data_mask.astype(bool)
    for i in range(3):
        for j in range(3):
            new_Stokes_cov[i,j] = sc_rotate(new_Stokes_cov[i,j], ang, order=1,
                    reshape=False, cval=0.)
        new_Stokes_cov[i,i] = np.abs(new_Stokes_cov[i,i])

    #Update headers to new angle
    new_headers = []
    mrot = np.array([[np.cos(-alpha), -np.sin(-alpha)],
                    [np.sin(-alpha), np.cos(-alpha)]])
    for header in headers:
        new_header = deepcopy(header)
        new_header['orientat'] = header['orientat'] + ang
        new_wcs = WCS(header).deepcopy()

        new_wcs.wcs.pc = np.dot(mrot, new_wcs.wcs.pc)
        print(new_wcs.wcs.pc)
        new_wcs.wcs.crpix = np.dot(mrot, new_wcs.wcs.crpix - old_center[::-1]) + new_center[::-1]
        new_wcs.wcs.set()
        for key, val in new_wcs.to_header().items():
            new_header.set(key,val)
        if new_wcs.wcs.pc[0,0] == 1.:
            new_header.set('PC1_1',1.)
        if new_wcs.wcs.pc[1,1] == 1.:
            new_header.set('PC2_2',1.)

        new_headers.append(new_header)

    # Nan handling :
    fmax = np.finfo(np.float64).max

    new_I_stokes[np.isnan(new_I_stokes)] = 0.
    new_Q_stokes[new_I_stokes == 0.] = 0.
    new_U_stokes[new_I_stokes == 0.] = 0.
    new_Q_stokes[np.isnan(new_Q_stokes)] = 0.
    new_U_stokes[np.isnan(new_U_stokes)] = 0.
    new_Stokes_cov[np.isnan(new_Stokes_cov)] = fmax

    #Compute updated integrated values for P, PA
    mask = deepcopy(new_data_mask).astype(bool)
    n_pix = new_I_stokes[mask].size
    I_diluted = new_I_stokes[mask].sum()
    Q_diluted = new_Q_stokes[mask].sum()
    U_diluted = new_U_stokes[mask].sum()
    I_diluted_err = np.sqrt(np.sum(new_Stokes_cov[0,0][mask]))
    Q_diluted_err = np.sqrt(np.sum(new_Stokes_cov[1,1][mask]))
    U_diluted_err = np.sqrt(np.sum(new_Stokes_cov[2,2][mask]))
    IQ_diluted_err = np.sqrt(np.sum(new_Stokes_cov[0,1][mask]**2))
    IU_diluted_err = np.sqrt(np.sum(new_Stokes_cov[0,2][mask]**2))
    QU_diluted_err = np.sqrt(np.sum(new_Stokes_cov[1,2][mask]**2))

    P_diluted = np.sqrt(Q_diluted**2+U_diluted**2)/I_diluted
    P_diluted_err = (1./I_diluted)*np.sqrt((Q_diluted**2*Q_diluted_err**2 + U_diluted**2*U_diluted_err**2 + 2.*Q_diluted*U_diluted*QU_diluted_err)/(Q_diluted**2 + U_diluted**2) + ((Q_diluted/I_diluted)**2 + (U_diluted/I_diluted)**2)*I_diluted_err**2 - 2.*(Q_diluted/I_diluted)*IQ_diluted_err - 2.*(U_diluted/I_diluted)*IU_diluted_err)

    PA_diluted = princ_angle((90./np.pi)*np.arctan2(U_diluted,Q_diluted))
    PA_diluted_err = (90./(np.pi*(Q_diluted**2 + U_diluted**2)))*np.sqrt(U_diluted**2*Q_diluted_err**2 + Q_diluted**2*U_diluted_err**2 - 2.*Q_diluted*U_diluted*QU_diluted_err)

    for header in new_headers:
        header['P_int'] = (P_diluted, 'Integrated polarization degree')
        header['P_int_err'] = (P_diluted_err, 'Integrated polarization degree error')
        header['PA_int'] = (PA_diluted, 'Integrated polarization angle')
        header['PA_int_err'] = (PA_diluted_err, 'Integrated polarization angle error')


    return new_I_stokes, new_Q_stokes, new_U_stokes, new_Stokes_cov, new_data_mask, new_headers


def rotate_data(data_array, error_array, data_mask, headers, ang):
    """
    Use scipy.ndimage.rotate to rotate I_stokes to an angle, and a rotation
    matrix to rotate Q, U of a given angle in degrees and update header
    orientation keyword.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array of images (2D floats) to be rotated by angle ang.
    error_array : numpy.ndarray
        Array of error associated to images in data_array.
    data_mask : numpy.ndarray
        2D boolean array delimiting the data to work on.
    headers : header list
        List of headers corresponding to the reduced images.
    ang : float
        Rotation angle (in degrees) that should be applied to the Stokes
        parameters
    ----------
    Returns:
    new_data_array : numpy.ndarray
        Updated array containing the rotated images.
    new_error_array : numpy.ndarray
        Updated array containing the rotated errors.
    new_headers : header list
        Updated list of headers corresponding to the reduced images accounting
        for the new orientation angle.
    new_data_mask : numpy.ndarray
        Updated 2D boolean array delimiting the data to work on.
    """
    #Rotate I_stokes, Q_stokes, U_stokes using rotation matrix
    alpha = ang*np.pi/180.

    old_center = np.array(data_array[0].shape)/2
    shape = np.fix(np.array(data_array[0].shape)*np.sqrt(2.5)).astype(int)
    new_center = np.array(shape)/2

    data_array = zeropad(data_array, [data_array.shape[0],*shape])
    error_array = zeropad(error_array, [error_array.shape[0],*shape])
    data_mask = zeropad(data_mask, shape)
    #Rotate original images using scipy.ndimage.rotate
    new_data_array = []
    new_error_array = []
    for i in range(data_array.shape[0]):
        new_data_array.append(sc_rotate(data_array[i], ang, order=1, reshape=False,
            cval=0.))
        new_error_array.append(sc_rotate(error_array[i], ang, order=1, reshape=False,
            cval=0.))
    new_data_array = np.array(new_data_array)
    new_error_array = np.array(new_error_array)
    new_data_mask = sc_rotate(data_mask*10., ang, order=1, reshape=False, cval=0.)
    new_data_mask[new_data_mask < 2] = 0.
    new_data_mask = new_data_mask.astype(bool)

    for i in range(new_data_array.shape[0]):
        new_data_array[i][new_data_array[i] < 0.] = 0.

    #Update headers to new angle
    new_headers = []
    mrot = np.array([[np.cos(-alpha), -np.sin(-alpha)],
            [np.sin(-alpha), np.cos(-alpha)]])
    for header in headers:
        new_header = deepcopy(header)
        new_header['orientat'] = header['orientat'] + ang

        new_wcs = WCS(header).deepcopy()

        new_wcs.wcs.pc = np.dot(mrot, new_wcs.wcs.pc)
        new_wcs.wcs.crpix = np.dot(mrot, new_wcs.wcs.crpix - old_center[::-1]) + new_center[::-1]
        new_wcs.wcs.set()
        for key, val in new_wcs.to_header().items():
            new_header[key] = val

        new_headers.append(new_header)
    globals()['theta'] = theta - alpha

    return new_data_array, new_error_array, new_data_mask, new_headers
