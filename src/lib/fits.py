"""
Library function for simplified fits handling.

prototypes :
    - get_obs_data(infiles, data_folder) -> data_array, headers
        Extract the observationnal data from fits files

    - save_Stokes(I, Q, U, Stokes_cov, P, debiased_P, s_P, s_P_P, PA, s_PA, s_PA_P, ref_header, filename, data_folder, return_hdul) -> ( HDUL_data )
        Save computed polarimetry parameters to a single fits file (and return HDUList)
"""

import numpy as np
from astropy.io import fits
from astropy import wcs


def get_obs_data(infiles, data_folder="", compute_flux=False):
    """
    Extract the observationnal data from the given fits files.
    ----------
    Inputs:
    infiles : strlist
        List of the fits file names to be added to the observation set.
    data_folder : str, optional
        Relative or absolute path to the folder containing the data.
    compute_flux : boolean, optional
        If True, return data_array will contain flux information, assuming
        raw data are counts and header have keywork EXPTIME and PHOTFLAM.
        Default to False.
    ----------
    Returns:
    data_array : numpy.ndarray
        Array of images (2D floats) containing the observation data.
    headers : header list
        List of headers objects corresponding to each image in data_array.
    """
    data_array, headers = [], []
    for i in range(len(infiles)):
        with fits.open(data_folder+infiles[i]) as f:
            headers.append(f[0].header)
            data_array.append(f[0].data)
    data_array = np.array(data_array)

    # Prevent negative count value in imported data
    for i in range(len(data_array)):
        data_array[i][data_array[i] < 0.] = 0.

    if compute_flux:
        for i in range(len(infiles)):
            # Compute the flux in counts/sec
            data_array[i] /= headers[i]['EXPTIME']

    return data_array, headers


def save_Stokes(I_stokes, Q_stokes, U_stokes, Stokes_cov, P, debiased_P, s_P,
        s_P_P, PA, s_PA, s_PA_P, headers, filename, data_folder="",
        return_hdul=False):
    """
    Save computed polarimetry parameters to a single fits file,
    updating header accordingly.
    ----------
    Inputs:
    I_stokes, Q_stokes, U_stokes, P, debiased_P, s_P, s_P_P, PA, s_PA, s_PA_P : numpy.ndarray
        Images (2D float arrays) containing the computed polarimetric data :
        Stokes parameters I, Q, U, Polarization degree and debieased,
        its error propagated and assuming Poisson noise, Polarization angle,
        its error propagated and assuming Poisson noise.
    Stokes_cov : numpy.ndarray
        Covariance matrix of the Stokes parameters I, Q, U.
    headers : header list
        Header of reference some keywords will be copied from (CRVAL, CDELT,
        INSTRUME, PROPOSID, TARGNAME, ORIENTAT, EXPTOT).
    filename : str
        Name that will be given to the file on writing (will appear in header).
    data_folder : str, optional
        Relative or absolute path to the folder the fits file will be saved to.
        Defaults to current folder.
    return_hdul : boolean, optional
        If True, the function will return the created HDUList from the
        input arrays.
        Defaults to False.
    ----------
    Return:
    hdul : astropy.io.fits.hdu.hdulist.HDUList
        HDUList containing I_stokes in the PrimaryHDU, then Q_stokes, U_stokes,
        P, s_P, PA, s_PA in this order. Headers have been updated to relevant
        informations (WCS, orientation, data_type).
        Only returned if return_hdul is True.
    """
    #Create new WCS object given the modified images
    ref_header = headers[0]
    exp_tot = np.array([header['exptime'] for header in headers]).sum()
    new_wcs = wcs.WCS(ref_header).deepcopy()
    if new_wcs.wcs.has_cd():
        del new_wcs.wcs.cd
        keys = list(new_wcs.to_header().keys())+['CD1_1','CD1_2','CD2_1','CD2_2']
        for key in keys:
            ref_header.remove(key, ignore_missing=True)
        new_wcs.wcs.cdelt = 3600.*np.sqrt(np.sum(new_wcs.wcs.get_pc()**2,axis=1))
    if (new_wcs.wcs.cdelt == np.array([1., 1.])).all() and \
            (new_wcs.array_shape in [(512, 512),(1024,512),(512,1024),(1024,1024)]):
        # Update WCS with relevant information
        HST_aper = 2400.    # HST aperture in mm
        f_ratio = ref_header['f_ratio']
        px_dim = np.array([25., 25.])   # Pixel dimension in µm
        if ref_header['pxformt'].lower() == 'zoom':
            px_dim[0] = 50.
        new_wcs.wcs.cdelt = 206.3*px_dim/(f_ratio*HST_aper)
    new_wcs.wcs.crpix = [I_stokes.shape[0]/2, I_stokes.shape[1]/2]

    header = new_wcs.to_header()
    header['instrume'] = (ref_header['instrume'], 'Instrument from which data is reduced')
    header['photplam'] = (ref_header['photplam'], 'Pivot Wavelength')
    header['photflam'] = (ref_header['photflam'], 'Inverse Sensitivity in DN/sec/cm**2/Angst')
    header['exptot'] = (exp_tot, 'Total exposure time in sec')
    header['proposid'] = (ref_header['proposid'], 'PEP proposal identifier for observation')
    header['targname'] = (ref_header['targname'], 'Target name')
    header['orientat'] = (ref_header['orientat'], 'Angle between North and the y-axis of the image')
    header['filename'] = (filename, 'Original filename')

    #Create HDUList object
    hdul = fits.HDUList([])

    #Add I_stokes as PrimaryHDU
    header['datatype'] = ('I_stokes', 'type of data stored in the HDU')
    primary_hdu = fits.PrimaryHDU(data=I_stokes, header=header)
    hdul.append(primary_hdu)

    #Add Q, U, Stokes_cov, P, s_P, PA, s_PA to the HDUList
    for data, name in [[Q_stokes,'Q_stokes'],[U_stokes,'U_stokes'],
            [Stokes_cov,'IQU_cov_matrix'],[P, 'Pol_deg'],
            [debiased_P, 'Pol_deg_debiased'],[s_P, 'Pol_deg_err'],
            [s_P_P, 'Pol_deg_err_Poisson_noise'],[PA, 'Pol_ang'],
            [s_PA, 'Pol_ang_err'],[s_PA_P, 'Pol_ang_err_Poisson_noise']]:
        hdu_header = header.copy()
        hdu_header['datatype'] = name
        hdu = fits.ImageHDU(data=data,header=hdu_header)
        hdul.append(hdu)

    #Save fits file to designated filepath
    hdul.writeto(data_folder+filename+".fits", overwrite=True)

    if return_hdul:
        return hdul
    else:
        return 0
