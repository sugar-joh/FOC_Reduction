"""
Library function computing various steps of the reduction pipeline.

prototypes :
    - bin_ndarray(ndarray, new_shape, operation) -> ndarray
        Bins an ndarray to new_shape.

    - crop_array(data_array, error_array, step, null_val, inside) -> crop_data_array, crop_error_array
        Homogeneously crop out null edges off a data array.

    - deconvolve_array(data_array, psf, FWHM, iterations) -> deconvolved_data_array
        Homogeneously deconvolve a data array using Richardson-Lucy iterative algorithm

    - get_error(data_array, sub_shape, display, headers, savename, plots_folder) -> data_array, error_array
        Compute the error (noise) on each image of the input array.

    - rebin_array(data_array, error_array, headers, pxsize, scale, operation) -> rebinned_data, rebinned_error, rebinned_headers, Dxy
        Homegeneously rebin a data array given a target pixel size in scale units.

    - align_data(data_array, error_array, upsample_factor, ref_data, ref_center, return_shifts) -> data_array, error_array (, shifts, errors)
        Align data_array on ref_data by cross-correlation.

    - smooth_data(data_array, error_array, FWHM, scale, smoothing) -> smoothed_array
        Smooth data by convoluting with a gaussian or by combining weighted images

    - polarizer_avg(data_array, error_array, headers, FWHM, scale, smoothing) -> polarizer_array, pol_error_array
        Average images in data_array on each used polarizer filter.

    - compute_Stokes(data_array, error_array, headers, FWHM, scale, smoothing) -> I_stokes, Q_stokes, U_stokes, Stokes_cov
        Compute Stokes parameters I, Q and U and their respective errors from data_array.

    - compute_pol(I_stokes, Q_stokes, U_stokes, Stokes_cov, headers) -> P, debiased_P, s_P, s_P_P, PA, s_PA, s_PA_P
        Compute polarization degree (in %) and angle (in degree) and their
        respective errors

    - rotate_Stokes(I_stokes, Q_stokes, U_stokes, Stokes_cov, headers, ang) -> I_stokes, Q_stokes, U_stokes, Stokes_cov, headers
        Rotate I, Q, U given an angle in degrees using scipy functions.

    - rotate2_Stokes(I_stokes, Q_stokes, U_stokes, Stokes_cov, headers, ang) -> I_stokes, Q_stokes, U_stokes, Stokes_cov, P, debiased_P, s_P, s_P_P, PA, s_PA, s_PA_P, headers
        Rotate I, Q, U, P, PA and associated errors given an angle in degrees
        using scipy functions.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from scipy.ndimage import rotate as sc_rotate
from scipy.ndimage import shift as sc_shift
from astropy.wcs import WCS
from lib.deconvolve import deconvolve_im, gaussian_psf
from lib.convex_hull import image_hull
from lib.plots import plot_obs
from lib.cross_correlation import phase_cross_correlation


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


def crop_array(data_array, error_array=None, step=5, null_val=None, inside=False):
    """
    Homogeneously crop an array: all contained images will have the same shape.
    'inside' parameter will decide how much should be cropped.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array containing the observation data (2D float arrays) to
        homogeneously crop.
    error_array : numpy.ndarray, optional
        Array of images (2D floats, aligned and of the same shape) containing
        the error in each pixel of the observation images in data_array.
        If None, will be initialized to zeros.
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
    ----------
    Returns:
    cropped_array : numpy.ndarray
        Array containing the observationnal data homogeneously cropped.
    """
    if error_array is None:
        error_array = np.zeros(data_array.shape)
    if null_val is None:
        null_val = [1.00*error.mean() for error in error_array]
    elif type(null_val) is float:
        null_val = [null_val,]*len(error_array)

    vertex = np.zeros((data_array.shape[0],4),dtype=int)
    for i,image in enumerate(data_array):
        vertex[i] = image_hull(image,step=step,null_val=null_val[i],inside=inside)
    v_array = np.zeros(4,dtype=int)
    if inside:
        v_array[0] = np.max(vertex[:,0]).astype(int)
        v_array[1] = np.min(vertex[:,1]).astype(int)
        v_array[2] = np.max(vertex[:,2]).astype(int)
        v_array[3] = np.min(vertex[:,3]).astype(int)
    else:
        v_array[0] = np.min(vertex[:,0]).astype(int)
        v_array[1] = np.max(vertex[:,1]).astype(int)
        v_array[2] = np.min(vertex[:,2]).astype(int)
        v_array[3] = np.max(vertex[:,3]).astype(int)

    new_shape = np.array([v_array[1]-v_array[0],v_array[3]-v_array[2]])
    crop_array = np.zeros((data_array.shape[0],new_shape[0],new_shape[1]))
    crop_error_array = np.zeros((data_array.shape[0],new_shape[0],new_shape[1]))
    for i,image in enumerate(data_array):
        crop_array[i] = image[v_array[0]:v_array[1],v_array[2]:v_array[3]]
        crop_error_array[i] = error_array[i][v_array[0]:v_array[1],v_array[2]:v_array[3]]

    return crop_array, crop_error_array


def deconvolve_array(data_array, headers, psf='gaussian', FWHM=1., scale='px',
        shape=(9,9), iterations=20):
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
        Number of iterations of Richardson-Lucy deconvolution algorithm. Act as
        as a regulation of the process.
        Defaults to 20.
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
            if w.wcs.has_cd():
                del w.wcs.cd
                keys = list(w.to_header().keys())+['CD1_1','CD1_2','CD2_1','CD2_2']
                for key in keys:
                    header.remove(key, ignore_missing=True)
                w.wcs.cdelt = 3600.*np.sqrt(np.sum(w.wcs.get_pc()**2,axis=1))
            if (w.wcs.cdelt == np.array([1., 1.])).all() and \
                    (w.array_shape in [(512, 512),(1024,512),(512,1024),(1024,1024)]):
                # Update WCS with relevant information
                HST_aper = 2400.    # HST aperture in mm
                f_ratio = header['f_ratio']
                px_dim = np.array([25., 25.])   # Pixel dimension in µm
                if header['pxformt'].lower() == 'zoom':
                    px_dim[0] = 50.
                w.wcs.cdelt = 206.3*px_dim/(f_ratio*HST_aper)
            header.update(w.to_header())
            pxsize[i] = np.round(w.wcs.cdelt,5)
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
                clip=True, filter_epsilon=None)

    return deconv_array


def get_error(data_array, sub_shape=(15,15), display=False, headers=None,
        savename=None, plots_folder="", return_background=False):
    """
    Look for sub-image of shape sub_shape that have the smallest integrated
    flux (no source assumption) and define the background on the image by the
    standard deviation on this sub-image.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array containing the data to study (2D float arrays).
    sub_shape : tuple, optional
        Shape of the sub-image to look for. Must be odd.
        Defaults to (15,15).
    display : boolean, optional
        If True, data_array will be displayed with a rectangle around the
        sub-image selected for background computation.
        Defaults to False.
    headers : header list, optional
        Headers associated with the images in data_array. Will only be used if
        display is True.
        Defaults to None.
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
    error_array : numpy.ndarray
        Array containing the background values associated to the images in
        data_array.
    background : numpy.ndarray
        Array containing the pixel background value for each image in
        data_array.
        Only returned if return_background is True.
    """
    # Crop out any null edges
    data, error = crop_array(data_array, step=5, null_val=0., inside=False)

    sub_shape = np.array(sub_shape)
    # Make sub_shape of odd values
    if not(np.all(sub_shape%2)):
        sub_shape += 1-sub_shape%2

    shape = np.array(data.shape)
    diff = (sub_shape-1).astype(int)
    temp = np.zeros((shape[0],shape[1]-diff[0],shape[2]-diff[1]))
    error_array = np.ones(data_array.shape)
    rectangle = np.zeros((shape[0],4))
    background = np.zeros((shape[0]))

    for i,image in enumerate(data):
        # Find the sub-image of smallest integrated flux (suppose no source)
        #sub-image dominated by background
        for r in range(temp.shape[1]):
            for c in range(temp.shape[0]):
                temp[i][r,c] = image[r:r+diff[0],c:c+diff[1]].sum()

    minima = np.unravel_index(np.argmin(temp.sum(axis=0)),temp.shape[1:])

    for i, image in enumerate(data):
        rectangle[i] = minima[0], minima[1], sub_shape[0], sub_shape[1]
        # Compute error : root mean square of the background
        sub_image = image[minima[0]:minima[0]+sub_shape[0],minima[1]:minima[1]+sub_shape[1]]
        #error_array[i] *= np.std(sub_image)    # Previously computed using standard deviation over the background
        error_array[i] *= np.sqrt(np.sum(sub_image**2)/sub_image.size)
        background[i] = sub_image.sum()
        data_array[i] = np.abs(data_array[i] - sub_image.mean())

    if display:

        date_time = np.array([headers[i]['date-obs']+';'+headers[i]['time-obs']
            for i in range(len(headers))])
        date_time = np.array([datetime.strptime(d,'%Y-%m-%d;%H:%M:%S')
            for d in date_time])
        filt = np.array([headers[i]['filtnam1'] for i in range(len(headers))])
        dict_filt = {"POL0":'r', "POL60":'g', "POL120":'b'}
        c_filt = np.array([dict_filt[f] for f in filt])

        fig,ax = plt.subplots(figsize=(10,6),constrained_layout=True)
        for f in np.unique(filt):
            mask = [fil==f for fil in filt]
            ax.scatter(date_time[mask], background[mask], color=dict_filt[f],
                    label="Filter : {0:s}".format(f))
        ax.errorbar(date_time, background, yerr=error_array[:,0,0], fmt='+k',
                markersize=0, ecolor=c_filt)
        # Date handling
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlabel("Observation date and time")
        ax.set_ylabel(r"Flux [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")
        ax.set_title("Background flux and error computed for each image")
        plt.legend()

        if not(savename is None):
            fig.suptitle(savename+"_background_flux.png")
            fig.savefig(plots_folder+savename+"_background_flux.png",
                    bbox_inches='tight')
            vmin = np.min(np.log10(data[data > 0.]))
            vmax = np.max(np.log10(data[data > 0.]))
            plot_obs(np.log10(data), headers, vmin=vmin, vmax=vmax,
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
        return data_array, error_array, np.array([error_array[i][0,0] for i in range(error_array.shape[0])])
    else:
        return data_array, error_array


def rebin_array(data_array, error_array, headers, pxsize, scale,
        operation='sum'):
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
    ----------
    Returns:
    rebinned_data, rebinned_error : numpy.ndarray
        Rebinned arrays containing the images and associated errors.
    rebinned_headers : header list
        Updated headers corresponding to the images in rebinned_data.
    Dxy : numpy.ndarray
        Array containing the rebinning factor in each direction of the image.
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
            if w.wcs.has_cd():
                del w.wcs.cd
                keys = list(w.to_header().keys())+['CD1_1','CD1_2','CD2_1','CD2_2']
                for key in keys:
                    header.remove(key, ignore_missing=True)
                w.wcs.cdelt = 3600.*np.sqrt(np.sum(w.wcs.get_pc()**2,axis=1))
            if (w.wcs.cdelt == np.array([1., 1.])).all() and \
                    (w.array_shape in [(512, 512),(1024,512),(512,1024),(1024,1024)]):
                # Update WCS with relevant information
                f_ratio = header['f_ratio']
                px_dim = np.array([25., 25.])   # Pixel dimension in µm
                if header['pxformt'].lower() == 'zoom':
                    px_dim[0] = 50.
                w.wcs.cdelt = 206.3*px_dim/(f_ratio*HST_aper)
            header.update(w.to_header())

            # Compute binning ratio
            if scale.lower() in ['px', 'pixel']:
                Dxy = np.array([pxsize,]*2)
            elif scale.lower() in ['arcsec','arcseconds']:
                Dxy = np.floor(pxsize/w.wcs.cdelt).astype(int)
            else:
                raise ValueError("'{0:s}' invalid scale for binning.".format(scale))

            if (Dxy <= 1.).any():
                raise ValueError("Requested pixel size is below resolution.")
            new_shape = (image.shape//Dxy).astype(int)

            # Rebin data
            rebinned_data.append(bin_ndarray(image, new_shape=new_shape,
                operation=operation))

            # Propagate error
            rms_image = np.sqrt(bin_ndarray(image**2, new_shape=new_shape,
                operation='average'))
            #std_image = np.sqrt(bin_ndarray(image**2, new_shape=new_shape,
            #    operation='average') - bin_ndarray(image, new_shape=new_shape,
            #        operation='average')**2)
            new_error = np.sqrt(Dxy[0]*Dxy[1])*bin_ndarray(error,
                    new_shape=new_shape, operation='average')
            rebinned_error.append(np.sqrt(rms_image**2 + new_error**2))

            # Update header
            w = w.slice((np.s_[::Dxy[0]], np.s_[::Dxy[1]]))
            header['NAXIS1'],header['NAXIS2'] = w.array_shape
            header.update(w.to_header())
            rebinned_headers.append(header)


    rebinned_data = np.array(rebinned_data)
    rebinned_error = np.array(rebinned_error)

    return rebinned_data, rebinned_error, rebinned_headers, Dxy


def align_data(data_array, error_array=None, upsample_factor=1., ref_data=None,
        ref_center=None, return_shifts=True):
    """
    Align images in data_array using cross correlation, and rescale them to
    wider images able to contain any rotation of the reference image.
    All images in data_array must have the same shape.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array containing the data to align (2D float arrays).
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
        _, error_array, background = get_error(data_array, return_background=True)
    else:
        _, _, background = get_error(data_array, return_background=True)

    # Crop out any null edges
    #(ref_data must be cropped as well)
    full_array = np.concatenate((data_array,[ref_data]),axis=0)
    err_array = np.concatenate((error_array,[np.zeros(ref_data.shape)]),axis=0)

    full_array, err_array = crop_array(full_array, err_array, step=5,
            inside=False)

    data_array, ref_data = full_array[:-1], full_array[-1]
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
    res_shape = int(np.ceil(np.sqrt(2)*np.max(shape[1:])))
    rescaled_image = np.ones((shape[0],res_shape,res_shape))
    rescaled_error = np.ones((shape[0],res_shape,res_shape))
    res_center = (np.array(rescaled_image.shape[1:])/2).astype(int)

    shifts, errors = [], []
    for i,image in enumerate(data_array):
        # Initialize rescaled images to background values
        rescaled_image[i] *= 0.1*background[i]
        rescaled_error[i] *= background[i]
        # Get shifts and error by cross-correlation to ref_data
        shift, error, phase_diff = phase_cross_correlation(ref_data, image,
                upsample_factor=upsample_factor)
        # Rescale image to requested output
        center = np.fix(ref_center-shift).astype(int)
        res_shift = res_center-ref_center
        rescaled_image[i,res_shift[0]:res_shift[0]+shape[1],
                res_shift[1]:res_shift[1]+shape[2]] = copy.deepcopy(image)
        rescaled_error[i,res_shift[0]:res_shift[0]+shape[1],
                res_shift[1]:res_shift[1]+shape[2]] = copy.deepcopy(error_array[i])
        # Shift images to align
        rescaled_image[i] = sc_shift(rescaled_image[i], shift, cval=0.1*background[i])
        rescaled_error[i] = sc_shift(rescaled_error[i], shift, cval=background[i])

        shifts.append(shift)
        errors.append(error)

    shifts = np.array(shifts)
    errors = np.array(errors)

    if return_shifts:
        return rescaled_image, rescaled_error, shifts, errors
    else:
        return rescaled_image, rescaled_error


def smooth_data(data_array, error_array, headers, FWHM=1., scale='pixel',
        smoothing='gaussian'):
    """
    Smooth a data_array using selected function.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array containing the data to smooth (2D float arrays).
    error_array : numpy.ndarray
        Array of images (2D floats, aligned and of the same shape) containing
        the error in each pixel of the observation images in data_array.
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
            if w.wcs.has_cd():
                del w.wcs.cd
                keys = list(w.to_header().keys())+['CD1_1','CD1_2','CD2_1','CD2_2']
                for key in keys:
                    header.remove(key, ignore_missing=True)
                w.wcs.cdelt = 3600.*np.sqrt(np.sum(w.wcs.get_pc()**2,axis=1))
            if (w.wcs.cdelt == np.array([1., 1.])).all() and \
                    (w.array_shape in [(512, 512),(1024,512),(512,1024),(1024,1024)]):
                # Update WCS with relevant information
                HST_aper = 2400.    # HST aperture in mm
                f_ratio = header['f_ratio']
                px_dim = np.array([25., 25.])   # Pixel dimension in µm
                if header['pxformt'].lower() == 'zoom':
                    px_dim[0] = 50.
                w.wcs.cdelt = 206.3*px_dim/(f_ratio*HST_aper)
            header.update(w.to_header())
            pxsize[i] = np.round(w.wcs.cdelt,4)
        if (pxsize != pxsize[0]).any():
            raise ValueError("Not all images in array have same pixel size")
        FWHM /= pxsize[0].min()

    #Define gaussian stdev
    stdev = FWHM/(2.*np.sqrt(2.*np.log(2)))

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
                dist_rc = np.sqrt((r-xx)**2+(c-yy)**2)
                g_rc = np.array([np.exp(-0.5*(dist_rc/stdev)**2),]*len(data_array))
                # Apply weighted combination
                smoothed[r,c] = np.sum(data_array*weight*g_rc)/np.sum(weight*g_rc)
                error[r,c] = np.sqrt(np.sum(weight*g_rc**2))/np.sum(weight*g_rc)

    elif smoothing.lower() in ['gauss','gaussian']:
        #Convolution with gaussian function
        smoothed = np.zeros(data_array.shape)
        error = np.zeros(error_array.shape)
        for i,image in enumerate(data_array):
            xx, yy = np.indices(image.shape)
            for r in range(image.shape[0]):
                for c in range(image.shape[1]):
                    dist_rc = np.sqrt((r-xx)**2+(c-yy)**2)
                    g_rc = np.exp(-0.5*(dist_rc/stdev)**2)/(2.*np.pi*stdev**2)
                    smoothed[i][r,c] = np.sum(image*g_rc)
                    error[i][r,c] = np.sum(error_array*g_rc**2)

    else:
        raise ValueError("{} is not a valid smoothing option".format(smoothing))

    return smoothed, error


def polarizer_avg(data_array, error_array, headers, FWHM=None, scale='pixel',
        smoothing='gaussian'):
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
            pol0, err0 = smooth_data(pol0_array, err0_array, headers0,
                    FWHM=FWHM, scale=scale, smoothing=smoothing)
            pol60, err60 = smooth_data(pol60_array, err60_array, headers60,
                    FWHM=FWHM, scale=scale, smoothing=smoothing)
            pol120, err120 = smooth_data(pol120_array, err120_array, headers120,
                    FWHM=FWHM, scale=scale, smoothing=smoothing)

        else:
            # Average on each polarization filter.
            pol0 = pol0_array.mean(axis=0)
            pol60 = pol60_array.mean(axis=0)
            pol120 = pol120_array.mean(axis=0)
            pol_array = np.array([pol0, pol60, pol120])

            # Propagate uncertainties quadratically
            err0 = np.mean(err0_array,axis=0)/np.sqrt(err0_array.shape[0])
            err60 = np.mean(err60_array,axis=0)/np.sqrt(err60_array.shape[0])
            err120 = np.mean(err120_array,axis=0)/np.sqrt(err120_array.shape[0])
            polerr_array = np.array([err0, err60, err120])

            # Update headers
            for header in headers:
                if header['filtnam1']=='POL0':
                    list_head = headers0
                elif header['filtnam1']=='POL60':
                    list_head = headers60
                else:
                    list_head = headers120
                header['exptime'] = np.mean([head['exptime'] for head in list_head])/len(list_head)
            headers_array = [headers0[0], headers60[0], headers120[0]]
            if not(FWHM is None) and (smoothing.lower() in ['gaussian','gauss']):
                # Smooth by convoluting with a gaussian each polX image.
                pol_array, polerr_array = smooth_data(pol_array, polerr_array,
                        headers_array, FWHM=FWHM, scale=scale)
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


def compute_Stokes(data_array, error_array, headers, FWHM=None,
        scale='pixel', smoothing='gaussian_after'):
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
        pol_array, pol_cov = polarizer_avg(data_array, error_array, headers,
                FWHM=FWHM, scale=scale, smoothing=smoothing)
        pol0, pol60, pol120 = pol_array

        #Stokes parameters
        I_stokes = (2./3.)*(pol0 + pol60 + pol120)
        Q_stokes = (2./3.)*(2*pol0 - pol60 - pol120)
        U_stokes = (2./np.sqrt(3.))*(pol60 - pol120)

        #Stokes covariance matrix
        Stokes_cov = np.zeros((3,3,I_stokes.shape[0],I_stokes.shape[1]))
        Stokes_cov[0,0] = (4./9.)*(pol_cov[0,0]+pol_cov[1,1]+pol_cov[2,2]) + (8./9.)*(pol_cov[0,1]+pol_cov[0,2]+pol_cov[1,2])
        Stokes_cov[1,1] = (4./3.)*(pol_cov[1,1]+pol_cov[2,2]) - (8./3.)*pol_cov[1,2]
        Stokes_cov[2,2] = (4./9.)*(4.*pol_cov[0,0]+pol_cov[1,1]+pol_cov[2,2]) - (8./3.)*(2.*pol_cov[0,1]+2.*pol_cov[0,2]-pol_cov[1,2])
        Stokes_cov[0,1] = Stokes_cov[1,0] = (4./(3.*np.sqrt(3.)))*(pol_cov[1,1]-pol_cov[2,2]+pol_cov[0,1]-pol_cov[0,2])
        Stokes_cov[0,2] = Stokes_cov[2,0] = (4./9.)*(2.*pol_cov[0,0]-pol_cov[1,1]-pol_cov[2,2]+pol_cov[0,1]+pol_cov[0,2]-2.*pol_cov[1,2])
        Stokes_cov[1,2] = Stokes_cov[2,1] = (4./(3.*np.sqrt(3.)))*(-pol_cov[1,1]+pol_cov[2,2]+2.*pol_cov[0,1]-2.*pol_cov[0,2])

        #Remove nan
        I_stokes[np.isnan(I_stokes)]=0.
        Q_stokes[np.isnan(Q_stokes)]=0.
        U_stokes[np.isnan(U_stokes)]=0.

        if not(FWHM is None) and (smoothing.lower() in ['gaussian_after','gauss_after']):
            Stokes_array = np.array([I_stokes, Q_stokes, U_stokes])
            Stokes_error = np.array([np.sqrt(Stokes_cov[i,i]) for i in range(3)])
            Stokes_headers = headers[0:3]

            Stokes_array, Stokes_error = smooth_data(Stokes_array, Stokes_error,
                    headers=Stokes_headers, FWHM=FWHM, scale=scale)

            I_stokes, Q_stokes, U_stokes = Stokes_array
            Stokes_cov[0,0], Stokes_cov[1,1], Stokes_cov[2,2] = Stokes_error**2

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
    I_pol = np.sqrt(Q_stokes**2 + U_stokes**2)
    P = I_pol/I_stokes*100.
    PA = (90./np.pi)*np.arctan2(U_stokes,Q_stokes)+90

    if (np.isfinite(P)>100.).any():
        print("WARNING : found pixels for which P > 100%")

    #Associated errors
    s_P = (100./I_stokes)*np.sqrt((Q_stokes**2*Stokes_cov[1,1] + U_stokes**2*Stokes_cov[2,2] + 2.*Q_stokes*U_stokes*Stokes_cov[1,2])/(Q_stokes**2 + U_stokes**2) + ((Q_stokes/I_stokes)**2 + (U_stokes/I_stokes)**2)*Stokes_cov[0,0] - 2.*(Q_stokes/I_stokes)*Stokes_cov[0,1] - 2.*(U_stokes/I_stokes)*Stokes_cov[0,2])

    s_PA = (90./(np.pi*(Q_stokes**2 + U_stokes**2)))*np.sqrt(U_stokes**2*Stokes_cov[1,1] + Q_stokes**2*Stokes_cov[2,2] - 2.*Q_stokes*U_stokes*Stokes_cov[1,2])

    debiased_P = np.sqrt(P**2 - s_P**2)

    #Compute the total exposure time so that
    #I_stokes*exp_tot = N_tot the total number of events
    exp_tot = np.array([header['exptime'] for header in headers]).sum()
    N_obs = I_stokes/np.array([header['photflam'] for header in headers]).mean()*exp_tot

    #Errors on P, PA supposing Poisson noise
    s_P_P = np.sqrt(2.)*(N_obs)**(-0.5)*100.
    s_PA_P = s_P_P/(2.*P/100.)*180./np.pi

    return P, debiased_P, s_P, s_P_P, PA, s_PA, s_PA_P


def rotate_Stokes(I_stokes, Q_stokes, U_stokes, Stokes_cov, headers, ang):
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
    headers : header list
        List of headers corresponding to the reduced images.
    ang : float
        Rotation angle (in degrees) that should be applied to the Stokes
        parameters
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
    """
    #Rotate I_stokes, Q_stokes, U_stokes using rotation matrix
    alpha = ang*np.pi/180.
    new_I_stokes = 1.*I_stokes
    new_Q_stokes = np.cos(2*alpha)*Q_stokes + np.sin(2*alpha)*U_stokes
    new_U_stokes = -np.sin(2*alpha)*Q_stokes + np.cos(2*alpha)*U_stokes

    #Compute new covariance matrix on rotated parameters
    new_Stokes_cov = copy.deepcopy(Stokes_cov)
    new_Stokes_cov[1,1] = np.cos(2.*alpha)**2*Stokes_cov[1,1] + np.sin(2.*alpha)**2*Stokes_cov[2,2] + 2.*np.cos(2.*alpha)*np.sin(2.*alpha)*Stokes_cov[1,2]
    new_Stokes_cov[2,2] = np.sin(2.*alpha)**2*Stokes_cov[1,1] + np.cos(2.*alpha)**2*Stokes_cov[2,2] - 2.*np.cos(2.*alpha)*np.sin(2.*alpha)*Stokes_cov[1,2]
    new_Stokes_cov[0,1] = new_Stokes_cov[1,0] = np.cos(2.*alpha)*Stokes_cov[0,1] + np.sin(2.*alpha)*Stokes_cov[0,2]
    new_Stokes_cov[0,2] = new_Stokes_cov[2,0] = -np.sin(2.*alpha)*Stokes_cov[0,1] + np.cos(2.*alpha)*Stokes_cov[0,2]
    new_Stokes_cov[1,2] = new_Stokes_cov[2,1] = np.cos(2.*alpha)*np.sin(2.*alpha)*(Stokes_cov[2,2] - Stokes_cov[1,1]) + (np.cos(2.*alpha)**2 - np.sin(2.*alpha)**2)*Stokes_cov[1,2]

    #Rotate original images using scipy.ndimage.rotate
    new_I_stokes = sc_rotate(new_I_stokes, ang, reshape=False,
            cval=0.10*np.sqrt(new_Stokes_cov[0,0][0,0]))
    new_Q_stokes = sc_rotate(new_Q_stokes, ang, reshape=False,
            cval=0.10*np.sqrt(new_Stokes_cov[1,1][0,0]))
    new_U_stokes = sc_rotate(new_U_stokes, ang, reshape=False,
            cval=0.10*np.sqrt(new_Stokes_cov[2,2][0,0]))
    for i in range(3):
        for j in range(3):
            new_Stokes_cov[i,j] = sc_rotate(new_Stokes_cov[i,j], ang, reshape=False,
                    cval=0.10*new_Stokes_cov[i,j].mean())

    #Update headers to new angle
    new_headers = []
    mrot = np.array([[np.cos(-alpha), -np.sin(-alpha)],
        [np.sin(-alpha), np.cos(-alpha)]])
    for header in headers:
        new_header = copy.deepcopy(header)
        new_header['orientat'] = header['orientat'] + ang

        new_wcs = WCS(header).deepcopy()
        if new_wcs.wcs.has_cd():    # CD matrix
            del w.wcs.cd
            keys = ['CD1_1','CD1_2','CD2_1','CD2_2']
            for key in keys:
                new_header.remove(key, ignore_missing=True)
            w.wcs.cdelt = 3600.*np.sqrt(np.sum(w.wcs.get_pc()**2,axis=1))
        elif new_wcs.wcs.has_pc():      # PC matrix + CDELT
            newpc = np.dot(mrot, new_wcs.wcs.get_pc())
            new_wcs.wcs.pc = newpc
        new_wcs.wcs.set()
        new_header.update(new_wcs.to_header())

        new_headers.append(new_header)

    return new_I_stokes, new_Q_stokes, new_U_stokes, new_Stokes_cov, new_headers


def rotate2_Stokes(I_stokes, Q_stokes, U_stokes, Stokes_cov, headers, ang):
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
    headers : header list
        List of headers corresponding to the reduced images.
    ang : float
        Rotation angle (in degrees) that should be applied to the Stokes
        parameters
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
    P : numpy.ndarray
        Image (2D floats) containing the polarization degree (in %).
    s_P : numpy.ndarray
        Image (2D floats) containing the error on the polarization degree.
    PA : numpy.ndarray
        Image (2D floats) containing the polarization angle.
    s_PA : numpy.ndarray
        Image (2D floats) containing the error on the polarization angle.
    debiased_P : numpy.ndarray
        Image (2D floats) containing the debiased polarization degree (in %).
    s_P_P : numpy.ndarray
        Image (2D floats) containing the Poisson noise error on the
        polarization degree.
    s_PA_P : numpy.ndarray
        Image (2D floats) containing the Poisson noise error on the
        polarization angle.
    """
    # Rotate I_stokes, Q_stokes, U_stokes using rotation matrix
    alpha = ang*np.pi/180.
    new_I_stokes = 1.*I_stokes
    new_Q_stokes = np.cos(2*alpha)*Q_stokes + np.sin(2*alpha)*U_stokes
    new_U_stokes = -np.sin(2*alpha)*Q_stokes + np.cos(2*alpha)*U_stokes

    # Compute new covariance matrix on rotated parameters
    new_Stokes_cov = copy.deepcopy(Stokes_cov)
    new_Stokes_cov[1,1] = np.cos(2.*alpha)**2*Stokes_cov[1,1] + np.sin(2.*alpha)**2*Stokes_cov[2,2] + 2.*np.cos(2.*alpha)*np.sin(2.*alpha)*Stokes_cov[1,2]
    new_Stokes_cov[2,2] = np.sin(2.*alpha)**2*Stokes_cov[1,1] + np.cos(2.*alpha)**2*Stokes_cov[2,2] - 2.*np.cos(2.*alpha)*np.sin(2.*alpha)*Stokes_cov[1,2]
    new_Stokes_cov[0,1] = new_Stokes_cov[1,0] = np.cos(2.*alpha)*Stokes_cov[0,1] + np.sin(2.*alpha)*Stokes_cov[0,2]
    new_Stokes_cov[0,2] = new_Stokes_cov[2,0] = -np.sin(2.*alpha)*Stokes_cov[0,1] + np.cos(2.*alpha)*Stokes_cov[0,2]
    new_Stokes_cov[1,2] = new_Stokes_cov[2,1] = np.cos(2.*alpha)*np.sin(2.*alpha)*(Stokes_cov[2,2] - Stokes_cov[1,1]) + (np.cos(2.*alpha)**2 - np.sin(2.*alpha)**2)*Stokes_cov[1,2]

    # Compute new polarization parameters
    P, debiased_P, s_P, s_P_P, PA, s_PA, s_PA_P = compute_pol(new_I_stokes,
            new_Q_stokes, new_U_stokes, new_Stokes_cov, headers)

    # Rotate original images using scipy.ndimage.rotate
    new_I_stokes = sc_rotate(new_I_stokes, ang, reshape=False,
            cval=np.sqrt(new_Stokes_cov[0,0][0,0]))
    new_Q_stokes = sc_rotate(new_Q_stokes, ang, reshape=False,
            cval=np.sqrt(new_Stokes_cov[1,1][0,0]))
    new_U_stokes = sc_rotate(new_U_stokes, ang, reshape=False,
            cval=np.sqrt(new_Stokes_cov[2,2][0,0]))
    P = sc_rotate(P, ang, reshape=False, cval=P.mean())
    debiased_P = sc_rotate(debiased_P, ang, reshape=False,
            cval=debiased_P.mean())
    s_P = sc_rotate(s_P, ang, reshape=False, cval=s_P.mean())
    s_P_P = sc_rotate(s_P_P, ang, reshape=False, cval=s_P_P.mean())
    PA = sc_rotate(PA, ang, reshape=False, cval=PA.mean())
    s_PA = sc_rotate(s_PA, ang, reshape=False, cval=s_PA.mean())
    s_PA_P = sc_rotate(s_PA_P, ang, reshape=False, cval=s_PA_P.mean())
    for i in range(3):
        for j in range(3):
            new_Stokes_cov[i,j] = sc_rotate(new_Stokes_cov[i,j], ang,
                    reshape=False, cval=new_Stokes_cov[i,j].mean())

    #Update headers to new angle
    new_headers = []
    mrot = np.array([[np.cos(-alpha), -np.sin(-alpha)],
        [np.sin(-alpha), np.cos(-alpha)]])
    for header in headers:
        new_header = copy.deepcopy(header)
        new_header['orientat'] = header['orientat'] + ang

        new_wcs = WCS(header).deepcopy()
        if new_wcs.wcs.has_cd():    # CD matrix
            del w.wcs.cd
            keys = ['CD1_1','CD1_2','CD2_1','CD2_2']
            for key in keys:
                new_header.remove(key, ignore_missing=True)
            w.wcs.cdelt = 3600.*np.sqrt(np.sum(w.wcs.get_pc()**2,axis=1))
        elif new_wcs.wcs.has_pc():      # PC matrix + CDELT
            newpc = np.dot(mrot, new_wcs.wcs.get_pc())
            new_wcs.wcs.pc = newpc
        new_wcs.wcs.set()
        new_header.update(new_wcs.to_header())

        new_headers.append(new_header)

    return new_I_stokes, new_Q_stokes, new_U_stokes, new_Stokes_cov, P, debiased_P, s_P, s_P_P, PA, s_PA, s_PA_P, new_headers
