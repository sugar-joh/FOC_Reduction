"""
Library functions for displaying  informations using matplotlib

prototypes :
    - plot_obs(data_array, headers, shape, vmin, vmax, savename, plots_folder)
        Plots whole observation raw data in given display shape

    - polarization_map(Stokes_hdul, SNRp_cut, SNRi_cut, step_vec, savename, plots_folder, display)
        Plots polarization map of polarimetric parameters saved in an HDUList
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from astropy.wcs import WCS


def plot_obs(data_array, headers, shape=None, vmin=0., vmax=6., rectangle=None,
        savename=None, plots_folder=""):
    """
    Plots raw observation imagery with some information on the instrument and
    filters.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array of images (2D floats, aligned and of the same shape) of a
        single observation with multiple polarizers of an instrument
    headers : header list
        List of headers corresponding to the images in data_array
    shape : array-like of length 2, optional
        Shape of the display, with shape = [#row, #columns]. If None, defaults
        to the optimal square.
        Defaults to None.
    vmin : float, optional
        Min pixel value that should be displayed.
        Defaults to 0.
    vmax : float, optional
        Max pixel value that should be displayed.
        Defaults to 6.
    rectangle : numpy.ndarray, optional
        Array of parameters for matplotlib.patches.Rectangle objects that will
        be displayed on each output image. If None, no rectangle displayed.
        Defaults to None.
    savename : str, optional
        Name of the figure the map should be saved to. If None, the map won't
        be saved (only displayed).
        Defaults to None.
    plots_folder : str, optional
        Relative (or absolute) filepath to the folder in wich the map will
        be saved. Not used if savename is None.
        Defaults to current folder.
    """
    if shape is None:
        shape = np.array([np.ceil(np.sqrt(data_array.shape[0])).astype(int),]*2)
    fig, ax = plt.subplots(shape[0], shape[1], figsize=(10,10), dpi=200,
            sharex=True, sharey=True)

    for i, enum in enumerate(list(zip(ax.flatten(),data_array))):
        ax = enum[0]
        data = enum[1]
        instr = headers[i]['instrume']
        rootname = headers[i]['rootname']
        exptime = headers[i]['exptime']
        filt = headers[i]['filtnam1']
        #plots
        im = ax.imshow(data, vmin=vmin, vmax=vmax, origin='lower')
        if not(rectangle is None):
            x, y, width, height = rectangle[i]
            ax.add_patch(Rectangle((x, y), width, height, edgecolor='r', fill=False))
        #position of centroid
        ax.plot([data.shape[1]/2, data.shape[1]/2], [0,data.shape[0]-1], lw=1,
                color='black')
        ax.plot([0,data.shape[1]-1], [data.shape[1]/2, data.shape[1]/2], lw=1,
                color='black')
        ax.annotate(instr+":"+rootname, color='white', fontsize=5, xy=(0.02, 0.95), xycoords='axes fraction')
        ax.annotate(filt, color='white', fontsize=10, xy=(0.02, 0.02), xycoords='axes fraction')
        ax.annotate(exptime, color='white', fontsize=5, xy=(0.80, 0.02), xycoords='axes fraction')

    fig.subplots_adjust(hspace=0, wspace=0, right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.12, 0.02, 0.75])
    fig.colorbar(im, cax=cbar_ax)

    if not (savename is None):
        fig.suptitle(savename)
        fig.savefig(plots_folder+savename+".png",bbox_inches='tight')
    plt.show()
    return 0


def plot_Stokes(Stokes, savename=None, plots_folder=""):
    """
    Plots I/Q/U maps.
    ----------
    Inputs:
    Stokes : astropy.io.fits.hdu.hdulist.HDUList
        HDUList containing I, Q, U, P, s_P, PA, s_PA (in this particular order)
        for one observation.
    savename : str, optional
        Name of the figure the map should be saved to. If None, the map won't
        be saved (only displayed).
        Defaults to None.
    plots_folder : str, optional
        Relative (or absolute) filepath to the folder in wich the map will
        be saved. Not used if savename is None.
        Defaults to current folder.
    """
    # Get data
    stkI = Stokes[np.argmax([Stokes[i].header['datatype']=='I_stokes' for i in range(len(Stokes))])].data
    stkQ = Stokes[np.argmax([Stokes[i].header['datatype']=='Q_stokes' for i in range(len(Stokes))])].data
    stkU = Stokes[np.argmax([Stokes[i].header['datatype']=='U_stokes' for i in range(len(Stokes))])].data

    wcs = WCS(Stokes[0]).deepcopy()

    # Plot figure
    fig = plt.figure(figsize=(30,10))

    ax = fig.add_subplot(131, projection=wcs)
    im = ax.imshow(stkI, origin='lower')
    plt.colorbar(im)
    ax.set(xlabel="RA", ylabel="DEC", title=r"$I_{stokes}$")

    ax = fig.add_subplot(132, projection=wcs)
    im = ax.imshow(stkQ, origin='lower')
    plt.colorbar(im)
    ax.set(xlabel="RA", ylabel="DEC", title=r"$Q_{stokes}$")

    ax = fig.add_subplot(133, projection=wcs)
    im = ax.imshow(stkU, origin='lower')
    plt.colorbar(im)
    ax.set(xlabel="RA", ylabel="DEC", title=r"$U_{stokes}$")

    if not (savename is None):
        fig.suptitle(savename+"_IQU")
        fig.savefig(plots_folder+savename+"_IQU.png",bbox_inches='tight')
    plt.show()
    return 0


def polarization_map(Stokes, SNRp_cut=3., SNRi_cut=30., step_vec=1,
        savename=None, plots_folder="", display=None):
    """
    Plots polarization map from Stokes HDUList.
    ----------
    Inputs:
    Stokes : astropy.io.fits.hdu.hdulist.HDUList
        HDUList containing I, Q, U, P, s_P, PA, s_PA (in this particular order)
        for one observation.
    SNRp_cut : float, optional
        Cut that should be applied to the signal-to-noise ratio on P.
        Any SNR < SNRp_cut won't be displayed.
        Defaults to 3.
    SNRi_cut : float, optional
        Cut that should be applied to the signal-to-noise ratio on I.
        Any SNR < SNRi_cut won't be displayed.
        Defaults to 30. This value implies an uncertainty in P of 4.7%
    step_vec : int, optional
        Number of steps between each displayed polarization vector.
        If step_vec = 2, every other vector will be displayed.
        Defaults to 1
    savename : str, optional
        Name of the figure the map should be saved to. If None, the map won't
        be saved (only displayed).
        Defaults to None.
    plots_folder : str, optional
        Relative (or absolute) filepath to the folder in wich the map will
        be saved. Not used if savename is None.
        Defaults to current folder.
    display : str, optional
        Choose the map to display between intensity (default), polarization
        degree ('p','pol','pol_deg') or polarization degree error ('s_p',
        'pol_err','pol_deg_err').
        Defaults to None (intensity).
    """
    #Get data
    stkI = Stokes[np.argmax([Stokes[i].header['datatype']=='I_stokes' for i in range(len(Stokes))])]
    stkQ = Stokes[np.argmax([Stokes[i].header['datatype']=='Q_stokes' for i in range(len(Stokes))])]
    stkU = Stokes[np.argmax([Stokes[i].header['datatype']=='U_stokes' for i in range(len(Stokes))])]
    stk_cov = Stokes[np.argmax([Stokes[i].header['datatype']=='IQU_cov_matrix' for i in range(len(Stokes))])]
    pol = Stokes[np.argmax([Stokes[i].header['datatype']=='Pol_deg' for i in range(len(Stokes))])]
    pol_err = Stokes[np.argmax([Stokes[i].header['datatype']=='Pol_deg_err' for i in range(len(Stokes))])]
    #pol_err_Poisson = Stokes[np.argmax([Stokes[i].header['datatype']=='Pol_deg_err_Poisson_noise' for i in range(len(Stokes))])]
    pang = Stokes[np.argmax([Stokes[i].header['datatype']=='Pol_ang' for i in range(len(Stokes))])]
    pang_err = Stokes[np.argmax([Stokes[i].header['datatype']=='Pol_ang_err' for i in range(len(Stokes))])]

    pivot_wav = Stokes[0].header['photplam']
    convert_flux = Stokes[0].header['photflam']
    wcs = WCS(Stokes[0]).deepcopy()

    #Plot Stokes parameters map
    if display is None:
        plot_Stokes(Stokes, savename=savename, plots_folder=plots_folder)

    #Compute SNR and apply cuts
    pol.data[pol.data == 0.] = np.nan
    SNRp = pol.data/pol_err.data
    SNRp[np.isnan(SNRp)] = 0.
    pol.data[SNRp < SNRp_cut] = np.nan
    SNRi = stkI.data/np.sqrt(stk_cov.data[0,0])
    SNRi[np.isnan(SNRi)] = 0.
    pol.data[SNRi < SNRi_cut] = np.nan

    mask = (SNRp > SNRp_cut) * (SNRi > SNRi_cut)

    # Look for pixel of max polarization
    if np.isfinite(pol.data).any():
        p_max = np.max(pol.data[np.isfinite(pol.data)])
        x_max, y_max = np.unravel_index(np.argmax(pol.data==p_max),pol.data.shape)
    else:
        print("No pixel with polarization information above requested SNR.")

    #Plot the map
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection=wcs)
    ax.set_facecolor('k')
    fig.subplots_adjust(hspace=0, wspace=0, right=0.95)
    cbar_ax = fig.add_axes([0.98, 0.12, 0.01, 0.75])

    if display is None:
        # If no display selected, show intensity map
        vmin, vmax = 0., np.max(stkI.data[stkI.data > 0.]*convert_flux)
        im = ax.imshow(stkI.data*convert_flux,extent=[-stkI.data.shape[1]/2.,stkI.data.shape[1]/2.,-stkI.data.shape[0]/2.,stkI.data.shape[0]/2.], vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")
        levelsI = np.linspace(SNRi_cut, np.max(SNRi[SNRi > 0.]), 10)
        cont = ax.contour(SNRi, extent=[-SNRi.shape[1]/2.,SNRi.shape[1]/2.,-SNRi.shape[0]/2.,SNRi.shape[0]/2.], levels=levelsI, colors='grey', linewidths=0.5)
    elif display.lower() in ['p','pol','pol_deg']:
        # Display polarization degree map
        vmin, vmax = 0., 100.
        im = ax.imshow(pol.data,extent=[-pol.data.shape[1]/2.,pol.data.shape[1]/2.,-pol.data.shape[0]/2.,pol.data.shape[0]/2.], vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$P$ [%]")
    elif display.lower() in ['s_p','pol_err','pol_deg_err']:
        # Display polarization degree error map
        vmin, vmax = 0., 5.
        im = ax.imshow(pol_err.data,extent=[-pol_err.data.shape[1]/2.,pol_err.data.shape[1]/2.,-pol_err.data.shape[0]/2.,pol_err.data.shape[0]/2.], vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$\sigma_P$ [%]")
    elif display.lower() in ['snr','snri']:
        # Display I_stokes signal-to-noise map
        vmin, vmax = 0., np.max(SNRi[SNRi > 0.])
        im = ax.imshow(SNRi, extent=[-SNRi.shape[1]/2.,SNRi.shape[1]/2.,-SNRi.shape[0]/2.,SNRi.shape[0]/2.], vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$I_{Stokes}/\sigma_{I}$")
        levelsI = np.linspace(SNRi_cut, np.max(SNRi[SNRi > 0.]), 10)
        cont = ax.contour(SNRi, extent=[-SNRi.shape[1]/2.,SNRi.shape[1]/2.,-SNRi.shape[0]/2.,SNRi.shape[0]/2.], levels=levelsI, colors='grey', linewidths=0.5)
    elif display.lower() in ['snrp']:
        # Display polarization degree signal-to-noise map
        vmin, vmax = SNRp_cut, np.max(SNRp[SNRp > 0.])
        im = ax.imshow(SNRp, extent=[-SNRp.shape[1]/2.,SNRp.shape[1]/2.,-SNRp.shape[0]/2.,SNRp.shape[0]/2.], vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$P/\sigma_{P}$")
        levelsP = np.linspace(SNRp_cut, np.max(SNRp[SNRp > 0.]), 10)
        cont = ax.contour(SNRp, extent=[-SNRp.shape[1]/2.,SNRp.shape[1]/2.,-SNRp.shape[0]/2.,SNRp.shape[0]/2.], levels=levelsP, colors='grey', linewidths=0.5)
    else:
        # Defaults to intensity map
        vmin, vmax = 0., np.max(stkI.data[stkI.data > 0.]*convert_flux)
        im = ax.imshow(stkI.data*convert_flux,extent=[-stkI.data.shape[1]/2.,stkI.data.shape[1]/2.,-stkI.data.shape[0]/2.,stkI.data.shape[0]/2.], vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA$]")
        levelsI = np.linspace(SNRi_cut, SNRi.max(), 10)
        cont = ax.contour(SNRi, extent=[-SNRi.shape[1]/2.,SNRi.shape[1]/2.,-SNRi.shape[0]/2.,SNRi.shape[0]/2.], levels=levelsI, colors='grey', linewidths=0.5)

    px_size = wcs.wcs.get_cdelt()[0]
    px_sc = AnchoredSizeBar(ax.transData, 1./px_size, '1 arcsec', 3, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color='w')
    ax.add_artist(px_sc)

    X, Y = np.meshgrid(np.linspace(-stkI.data.shape[0]/2.,stkI.data.shape[0]/2.,stkI.data.shape[0]), np.linspace(-stkI.data.shape[1]/2.,stkI.data.shape[1]/2.,stkI.data.shape[1]))
    U, V = pol.data*np.cos(np.pi/2.+pang.data*np.pi/180.), pol.data*np.sin(np.pi/2.+pang.data*np.pi/180.)
    Q = ax.quiver(X[::step_vec,::step_vec],Y[::step_vec,::step_vec],U[::step_vec,::step_vec],V[::step_vec,::step_vec],units='xy',angles='uv',scale=50.,scale_units='xy',pivot='mid',headwidth=0.,headlength=0.,headaxislength=0.,width=0.1,color='w')
    pol_sc = AnchoredSizeBar(ax.transData, 2., r"$P$= 100 %", 4, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color='w')
    ax.add_artist(pol_sc)

    # Compute integrated parameters and associated errors for pixels in the cut
    I_int = stkI.data[mask].sum()
    Q_int = stkQ.data[mask].sum()
    U_int = stkU.data[mask].sum()
    I_int_err = np.sqrt(np.sum(stk_cov.data[0,0][mask]))
    Q_int_err = np.sqrt(np.sum(stk_cov.data[1,1][mask]))
    U_int_err = np.sqrt(np.sum(stk_cov.data[2,2][mask]))
    IQ_int_err = np.sqrt(np.sum(stk_cov.data[0,1][mask]**2))
    IU_int_err = np.sqrt(np.sum(stk_cov.data[0,2][mask]**2))
    QU_int_err = np.sqrt(np.sum(stk_cov.data[1,2][mask]**2))

    P_int = np.sqrt(Q_int**2+U_int**2)/I_int*100.
    P_int_err = (100./I_int)*np.sqrt((Q_int**2*Q_int_err**2 + U_int**2*U_int_err**2 + 2.*Q_int*U_int*QU_int_err)/(Q_int**2 + U_int**2) + ((Q_int/I_int)**2 + (U_int/I_int)**2)*I_int_err**2 - 2.*(Q_int/I_int)*IQ_int_err - 2.*(U_int/I_int)*IU_int_err)

    PA_int = (90./np.pi)*np.arctan2(U_int,Q_int)+90.
    PA_int_err = (90./(np.pi*(Q_int**2 + U_int**2)))*np.sqrt(U_int**2*Q_int_err**2 + Q_int**2*U_int_err**2 - 2.*Q_int*U_int*QU_int_err)

    # Compute integrated parameters and associated errors for all pixels
    I_diluted = stkI.data.sum()
    Q_diluted = stkQ.data.sum()
    U_diluted = stkU.data.sum()
    I_diluted_err = np.sqrt(np.sum(stk_cov.data[0,0]))
    Q_diluted_err = np.sqrt(np.sum(stk_cov.data[1,1]))
    U_diluted_err = np.sqrt(np.sum(stk_cov.data[2,2]))
    IQ_diluted_err = np.sqrt(np.sum(stk_cov.data[0,1]**2))
    IU_diluted_err = np.sqrt(np.sum(stk_cov.data[0,2]**2))
    QU_diluted_err = np.sqrt(np.sum(stk_cov.data[1,2]**2))

    P_diluted = np.sqrt(Q_diluted**2+U_diluted**2)/I_diluted*100.
    P_diluted_err = (100./I_diluted)*np.sqrt((Q_diluted**2*Q_diluted_err**2 + U_diluted**2*U_diluted_err**2 + 2.*Q_diluted*U_diluted*QU_diluted_err)/(Q_diluted**2 + U_diluted**2) + ((Q_diluted/I_diluted)**2 + (U_diluted/I_diluted)**2)*I_diluted_err**2 - 2.*(Q_diluted/I_diluted)*IQ_diluted_err - 2.*(U_diluted/I_diluted)*IU_diluted_err)

    PA_diluted = (90./np.pi)*np.arctan2(U_diluted,Q_diluted)+90.
    PA_diluted_err = (90./(np.pi*(Q_diluted**2 + U_diluted**2)))*np.sqrt(U_diluted**2*Q_diluted_err**2 + Q_diluted**2*U_diluted_err**2 - 2.*Q_diluted*U_diluted*QU_diluted_err)

    ax.annotate(r"$F_{{\lambda}}^{{int}}$({0:.0f} $\AA$) = {1:.1e} $\pm$ {2:.1e} $ergs \cdot cm^{{-2}} \cdot s^{{-1}} \cdot \AA^{{-1}}$".format(pivot_wav,I_int*convert_flux,I_int_err*convert_flux)+"\n"+r"$P^{{int}}$ = {0:.2f} $\pm$ {1:.2f} %".format(P_int,P_int_err)+"\n"+r"$\theta_{{P}}^{{int}}$ = {0:.2f} $\pm$ {1:.2f} °".format(PA_int,PA_int_err)+"\n"+r"$P^{{diluted}}$ = {0:.2f} $\pm$ {1:.2f} %".format(P_diluted,P_diluted_err)+"\n"+r"$\theta_{{P}}^{{diluted}}$ = {0:.2f} $\pm$ {1:.2f} °".format(PA_diluted,PA_diluted_err), color='white', fontsize=11, xy=(0.01, 0.90), xycoords='axes fraction')

    ax.coords.grid(True, color='white', ls='dotted', alpha=0.5)
    ax.coords[0].set_axislabel('Right Ascension (J2000)')
    ax.coords[0].set_axislabel_position('t')
    ax.coords[0].set_ticklabel_position('t')
    ax.coords[1].set_axislabel('Declination (J2000)')
    ax.coords[1].set_axislabel_position('l')
    ax.coords[1].set_ticklabel_position('l')
    ax.axis('equal')

    if not savename is None:
        fig.suptitle(savename)
        fig.savefig(plots_folder+savename+".png",bbox_inches='tight',dpi=200)

    plt.show()
    return 0
