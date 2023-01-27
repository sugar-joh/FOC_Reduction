"""
Library functions for displaying  informations using matplotlib

prototypes :
    - plot_obs(data_array, headers, shape, vmin, vmax, rectangle, savename, plots_folder)
        Plots whole observation raw data in given display shape.
    
    - plot_Stokes(Stokes, savename, plots_folder)
        Plot the I/Q/U maps from the Stokes HDUList.

    - polarization_map(Stokes, data_mask, rectangle, SNRp_cut, SNRi_cut, step_vec, savename, plots_folder, display) -> fig, ax
        Plots polarization map of polarimetric parameters saved in an HDUList.
    
    class align_maps(map, other_map, **kwargs)
        Class to interactively align maps with different WCS.
    
    class overplot_radio(align_maps)
        Class inherited from align_maps to overplot radio data as contours.
    
    class overplot_pol(align_maps)
        Class inherited from align_maps to overplot UV polarization vectors on other maps.
    
    class crop_map(hdul, fig, ax)
        Class to interactively crop a region of interest of a HDUList.
    
    class crop_Stokes(crop_map)
        Class inherited from crop_map to work on polarization maps.
    
    class image_lasso_selector(img, fig, ax)
        Class to interactively select part of a map to work on.
    
    class aperture(img, cdelt, radius, fig, ax)
        Class to interactively simulate aperture integration.
    
    class pol_map(Stokes, SNRp_cut, SNRi_cut, selection)
        Class to interactively study polarization maps making use of the cropping and selecting tools.
"""

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.path import Path
from matplotlib.widgets import RectangleSelector, LassoSelector, Button, Slider, TextBox
from matplotlib.colors import LogNorm
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar, AnchoredDirectionArrows
from astropy.wcs import WCS
from astropy.io import fits


def princ_angle(ang):
    """
    Return the principal angle in the 0° to 360° quadrant.
    """
    if type(ang) != np.ndarray:
        A = np.array([ang])
    else:
        A = np.array(ang)
    while np.any(A < 0.):
        A[A<0.] = A[A<0.]+360.
    while np.any(A >= 180.):
        A[A>=180.] = A[A>=180.]-180.
    if type(ang) == type(A):
        return A
    else:
        return A[0]


def sci_not(v,err,rnd=1):
    """
    Return the scientifque error notation as a string.
    """
    power = - int(('%E' % v)[-3:])+1
    output = r"({0}".format(round(v*10**power,rnd))
    if type(err) == list:
        for error in err:
            output += r" $\pm$ {0}".format(round(error*10**power,rnd))
    else:
        output += r" $\pm$ {0}".format(round(err*10**power,rnd))
    return output+r")e{0}".format(-power)


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
    plt.rcParams.update({'font.size': 10})
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
        #im = ax.imshow(data, vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
        im = ax.imshow(data, norm=LogNorm(data[data>0.].min()/10.,data.max()), origin='lower', cmap='gray')
        if not(rectangle is None):
            x, y, width, height, angle, color = rectangle[i]
            ax.add_patch(Rectangle((x, y), width, height, angle=angle,
                edgecolor=color, fill=False))
        #position of centroid
        ax.plot([data.shape[1]/2, data.shape[1]/2], [0,data.shape[0]-1], '--', lw=1,
                color='grey', alpha=0.5)
        ax.plot([0,data.shape[1]-1], [data.shape[1]/2, data.shape[1]/2], '--', lw=1,
                color='grey', alpha=0.5)
        ax.annotate(instr+":"+rootname,color='white',fontsize=5,xy=(0.02, 0.95),
                xycoords='axes fraction')
        ax.annotate(filt,color='white',fontsize=10,xy=(0.02, 0.02),
                xycoords='axes fraction')
        ax.annotate(exptime,color='white',fontsize=5,xy=(0.80, 0.02),
                xycoords='axes fraction')

    fig.subplots_adjust(hspace=0.01, wspace=0.01, right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.12, 0.02, 0.75])
    fig.colorbar(im, cax=cbar_ax, label=r'$Counts \cdot s^{-1}$')

    if not (savename is None):
        #fig.suptitle(savename)
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
    plt.rcParams.update({'font.size': 10})
    fig = plt.figure(figsize=(15,5))

    ax = fig.add_subplot(131, projection=wcs)
    im = ax.imshow(stkI, origin='lower', cmap='inferno')
    plt.colorbar(im)
    ax.set(xlabel="RA", ylabel="DEC", title=r"$I_{stokes}$")

    ax = fig.add_subplot(132, projection=wcs)
    im = ax.imshow(stkQ, origin='lower', cmap='inferno')
    plt.colorbar(im)
    ax.set(xlabel="RA", ylabel="DEC", title=r"$Q_{stokes}$")

    ax = fig.add_subplot(133, projection=wcs)
    im = ax.imshow(stkU, origin='lower', cmap='inferno')
    plt.colorbar(im)
    ax.set(xlabel="RA", ylabel="DEC", title=r"$U_{stokes}$")

    if not (savename is None):
        #fig.suptitle(savename+"_IQU")
        fig.savefig(plots_folder+savename+"_IQU.png",bbox_inches='tight')
    plt.show()
    return 0


def polarization_map(Stokes, data_mask=None, rectangle=None, SNRp_cut=3., SNRi_cut=30.,
        step_vec=1, savename=None, plots_folder="", display="default"):
    """
    Plots polarization map from Stokes HDUList.
    ----------
    Inputs:
    Stokes : astropy.io.fits.hdu.hdulist.HDUList
        HDUList containing I, Q, U, P, s_P, PA, s_PA (in this particular order)
        for one observation.
    rectangle : numpy.ndarray, optional
        Array of parameters for matplotlib.patches.Rectangle objects that will
        be displayed on each output image. If None, no rectangle displayed.
        Defaults to None.
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
    ----------
    Returns:
    fig, ax : matplotlib.pyplot object
        The figure and ax created for interactive contour maps.
    """
    #Get data
    stkI = Stokes[np.argmax([Stokes[i].header['datatype']=='I_stokes' for i in range(len(Stokes))])]
    stkQ = Stokes[np.argmax([Stokes[i].header['datatype']=='Q_stokes' for i in range(len(Stokes))])]
    stkU = Stokes[np.argmax([Stokes[i].header['datatype']=='U_stokes' for i in range(len(Stokes))])]
    stk_cov = Stokes[np.argmax([Stokes[i].header['datatype']=='IQU_cov_matrix' for i in range(len(Stokes))])]
    pol = Stokes[np.argmax([Stokes[i].header['datatype']=='Pol_deg_debiased' for i in range(len(Stokes))])]
    pol_err = Stokes[np.argmax([Stokes[i].header['datatype']=='Pol_deg_err' for i in range(len(Stokes))])]
    pang = Stokes[np.argmax([Stokes[i].header['datatype']=='Pol_ang' for i in range(len(Stokes))])]
    try:
        if data_mask is None:
            data_mask = Stokes[np.argmax([Stokes[i].header['datatype']=='Data_mask' for i in range(len(Stokes))])].data.astype(bool)
    except KeyError:
        data_mask = np.ones(stkI.shape).astype(bool)

    pivot_wav = Stokes[0].header['photplam']
    convert_flux = Stokes[0].header['photflam']
    wcs = WCS(Stokes[0]).deepcopy()

    #Plot Stokes parameters map
    if display is None or display.lower() in ['default']:
        plot_Stokes(Stokes, savename=savename, plots_folder=plots_folder)

    #Compute SNR and apply cuts
    pol.data[pol.data == 0.] = np.nan
    pol_err.data[pol_err.data == 0.] = np.nan
    SNRp = pol.data/pol_err.data
    SNRp[np.isnan(SNRp)] = 0.
    pol.data[SNRp < SNRp_cut] = np.nan
    pang.data[SNRp < SNRp_cut] = np.nan

    maskI = stk_cov.data[0,0] > 0
    SNRi = np.zeros(stkI.data.shape)
    SNRi[maskI] = stkI.data[maskI]/np.sqrt(stk_cov.data[0,0][maskI])
    pol.data[SNRi < SNRi_cut] = np.nan
    pang.data[SNRi < SNRi_cut] = np.nan

    mask = (SNRp > SNRp_cut) * (SNRi > SNRi_cut)

    # Look for pixel of max polarization
    if np.isfinite(pol.data).any():
        p_max = np.max(pol.data[np.isfinite(pol.data)])
        x_max, y_max = np.unravel_index(np.argmax(pol.data==p_max),pol.data.shape)
    else:
        print("No pixel with polarization information above requested SNR.")

    #Plot the map
    plt.rcParams.update({'font.size': 10})
    plt.rcdefaults()
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection=wcs)
    ax.set_facecolor('k')
    fig.subplots_adjust(hspace=0, wspace=0, right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.12, 0.01, 0.75])

    if display.lower() in ['intensity']:
        # If no display selected, show intensity map
        display='i'
        vmin, vmax = np.max(np.sqrt(stk_cov.data[0,0][mask])*convert_flux), np.max(stkI.data[stkI.data > 0.]*convert_flux)
        im = ax.imshow(stkI.data*convert_flux, norm=LogNorm(vmin,vmax), aspect='equal', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")
        levelsI = np.linspace(vmax*0.01, vmax*0.99, 10)
        print("Total flux contour levels : ", levelsI)
        cont = ax.contour(stkI.data*convert_flux, levels=levelsI, colors='grey', linewidths=0.5)
        #ax.clabel(cont,inline=True,fontsize=6)
    elif display.lower() in ['pol_flux']:
        # Display polarisation flux
        display='pf'
        pf_mask = (stkI.data > 0.) * (pol.data > 0.)
        vmin, vmax = np.max(np.sqrt(stk_cov.data[0,0][mask])*convert_flux), np.max(stkI.data[stkI.data > 0.]*convert_flux)
        im = ax.imshow(stkI.data*convert_flux*pol.data, norm=LogNorm(vmin,vmax), aspect='equal', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$F_{\lambda} \cdot P$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")
        levelsPf = np.linspace(vmax*0.01, vmax*0.99, 10)
        print("Polarized flux contour levels : ", levelsPf)
        cont = ax.contour(stkI.data*convert_flux*pol.data, levels=levelsPf, colors='grey', linewidths=0.5)
        #ax.clabel(cont,inline=True,fontsize=6)
    elif display.lower() in ['p','pol','pol_deg']:
        # Display polarization degree map
        display='p'
        vmin, vmax = 0., 100.
        im = ax.imshow(pol.data*100., vmin=vmin, vmax=vmax, aspect='equal', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$P$ [%]")
    elif display.lower() in ['pa','pang','pol_ang']:
        # Display polarization degree map
        display='pa'
        vmin, vmax = 0., 180.
        im = ax.imshow(princ_angle(pang.data), vmin=vmin, vmax=vmax, aspect='equal', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$\theta_P$ [°]")
    elif display.lower() in ['s_p','pol_err','pol_deg_err']:
        # Display polarization degree error map
        display='s_p'
        vmin, vmax = 0., np.max(pol_err.data[SNRp > SNRp_cut])*100.
        p_err = deepcopy(pol_err.data)
        p_err[p_err > vmax/100.] = np.nan
        im = ax.imshow(p_err*100., vmin=vmin, vmax=vmax, aspect='equal', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$\sigma_P$ [%]")
    elif display.lower() in ['s_i','i_err']:
        # Display intensity error map
        display='s_i'
        vmin, vmax = 0., np.max(np.sqrt(stk_cov.data[0,0][stk_cov.data[0,0] > 0.])*convert_flux)
        im = ax.imshow(np.sqrt(stk_cov.data[0,0])*convert_flux, vmin=vmin, vmax=vmax, aspect='equal', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$\sigma_I$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")
    elif display.lower() in ['snr','snri']:
        # Display I_stokes signal-to-noise map
        display='snri'
        vmin, vmax = 0., np.max(SNRi[SNRi > 0.])
        im = ax.imshow(SNRi, vmin=vmin, vmax=vmax, aspect='equal', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$I_{Stokes}/\sigma_{I}$")
        levelsSNRi = np.linspace(SNRi_cut, vmax*0.99, 10)
        print("SNRi contour levels : ", levelsSNRi)
        cont = ax.contour(SNRi, levels=levelsSNRi, colors='grey', linewidths=0.5)
        #ax.clabel(cont,inline=True,fontsize=6)
    elif display.lower() in ['snrp']:
        # Display polarization degree signal-to-noise map
        display='snrp'
        vmin, vmax = SNRp_cut, np.max(SNRp[SNRp > 0.])
        im = ax.imshow(SNRp, vmin=vmin, vmax=vmax, aspect='equal', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$P/\sigma_{P}$")
        levelsSNRp = np.linspace(SNRp_cut, vmax*0.99, 10)
        print("SNRp contour levels : ", levelsSNRp)
        cont = ax.contour(SNRp, levels=levelsSNRp, colors='grey', linewidths=0.5)
        #ax.clabel(cont,inline=True,fontsize=6)
    else:
        # Defaults to intensity map
        vmin, vmax = np.min(stkI.data[SNRi > SNRi_cut]*convert_flux)/5., np.max(stkI.data[SNRi > SNRi_cut]*convert_flux)
        #im = ax.imshow(stkI.data*convert_flux, vmin=vmin, vmax=vmax, aspect='equal', cmap='inferno', alpha=1.)
        #cbar = plt.colorbar(im, cax=cbar_ax, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA$]")
        im = ax.imshow(stkI.data*convert_flux, norm=LogNorm(vmin,vmax), aspect='equal', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA$]")

    #Get integrated values from header
    n_pix = stkI.data[data_mask].size
    I_diluted = stkI.data[data_mask].sum()
    I_diluted_err = np.sqrt(n_pix)*np.sqrt(np.sum(stk_cov.data[0,0][data_mask]))

    P_diluted = Stokes[0].header['P_int']
    P_diluted_err = Stokes[0].header['P_int_err']
    PA_diluted = Stokes[0].header['PA_int']
    PA_diluted_err = Stokes[0].header['PA_int_err']

    px_size = wcs.wcs.get_cdelt()[0]*3600.
    px_sc = AnchoredSizeBar(ax.transData, 1./px_size, '1 arcsec', 3, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color='w')
    north_dir = AnchoredDirectionArrows(ax.transAxes, "E", "N", length=-0.08, fontsize=0.025, loc=1, aspect_ratio=-1, sep_y=0.01, sep_x=0.01, back_length=0., head_length=10., head_width=10., angle=-Stokes[0].header['orientat'], color='white', text_props={'ec': None, 'fc': 'w', 'alpha': 1, 'lw': 0.4}, arrow_props={'ec': None,'fc':'w','alpha': 1,'lw': 1})

    if display.lower() in ['i','s_i','snri','pf','p','pa','s_p','snrp']:
        if step_vec == 0:
            pol.data[np.isfinite(pol.data)] = 1./2.
            step_vec = 1
        X, Y = np.meshgrid(np.arange(stkI.data.shape[1]), np.arange(stkI.data.shape[0]))
        U, V = pol.data*np.cos(np.pi/2.+pang.data*np.pi/180.), pol.data*np.sin(np.pi/2.+pang.data*np.pi/180.)
        Q = ax.quiver(X[::step_vec,::step_vec],Y[::step_vec,::step_vec],U[::step_vec,::step_vec],V[::step_vec,::step_vec],units='xy',angles='uv',scale=0.5,scale_units='xy',pivot='mid',headwidth=0.,headlength=0.,headaxislength=0.,width=0.1,color='w')
        pol_sc = AnchoredSizeBar(ax.transData, 2., r"$P$= 100 %", 4, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color='w')
        
        ax.add_artist(pol_sc)
        ax.add_artist(px_sc)
        ax.add_artist(north_dir)
        
        ax.annotate(r"$F_{{\lambda}}^{{int}}$({0:.0f} $\AA$) = {1} $ergs \cdot cm^{{-2}} \cdot s^{{-1}} \cdot \AA^{{-1}}$".format(pivot_wav,sci_not(I_diluted*convert_flux,I_diluted_err*convert_flux,2))+"\n"+r"$P^{{int}}$ = {0:.1f} $\pm$ {1:.1f} %".format(P_diluted*100.,P_diluted_err*100.)+"\n"+r"$\theta_{{P}}^{{int}}$ = {0:.1f} $\pm$ {1:.1f} °".format(PA_diluted,PA_diluted_err), color='white', xy=(0.01, 0.92), xycoords='axes fraction')
    else:
        if display.lower() == 'default':
            ax.add_artist(px_sc)
            ax.add_artist(north_dir)
        ax.annotate(r"$F_{{\lambda}}^{{int}}$({0:.0f} $\AA$) = {1} $ergs \cdot cm^{{-2}} \cdot s^{{-1}} \cdot \AA^{{-1}}$".format(pivot_wav,sci_not(I_diluted*convert_flux,I_diluted_err*convert_flux,2)), color='white', xy=(0.01, 0.97), xycoords='axes fraction')

    # Display instrument FOV
    if not(rectangle is None):
        x, y, width, height, angle, color = rectangle
        x, y = np.array([x, y])- np.array(stkI.data.shape)/2.
        ax.add_patch(Rectangle((x, y), width, height, angle=angle,
            edgecolor=color, fill=False))


    #ax.coords.grid(True, color='white', ls='dotted', alpha=0.5)
    ax.coords[0].set_axislabel('Right Ascension (J2000)')
    ax.coords[0].set_axislabel_position('t')
    ax.coords[0].set_ticklabel_position('t')
    ax.coords[1].set_axislabel('Declination (J2000)')
    ax.coords[1].set_axislabel_position('l')
    ax.coords[1].set_ticklabel_position('l')
    ax.axis('equal')

    if not savename is None:
        #fig.suptitle(savename)
        fig.savefig(plots_folder+savename+".png",bbox_inches='tight',dpi=300)

    plt.show()
    return fig, ax


class align_maps(object):
    """
    Class to interactively align maps with different WCS.
    """
    def __init__(self, map1, other_map, **kwargs):
        self.aligned = False
        self.map = map1
        self.other_map = other_map

        self.wcs_map = deepcopy(WCS(self.map[0])).celestial
        if len(self.map[0].data.shape) == 4:
            self.map[0].data = self.map[0].data[0,0]
        elif len(self.map[0].data.shape) == 3:
            self.map[0].data = self.map[0].data[1]

        self.wcs_other = deepcopy(WCS(self.other_map[0])).celestial
        if len(self.other_map[0].data.shape) == 4:
            self.other_map[0].data = self.other_map[0].data[0,0]
        elif len(self.other_map[0].data.shape) == 3:
            self.other_map[0].data = self.other_map[0].data[1]

        try:
            convert_flux = self.map[0].header['photflam']
        except KeyError:
            convert_flux = 1.
        try:
            other_convert = self.other_map[0].header['photflam']
        except KeyError:
            other_convert = 1.

        #Get data
        data = self.map[0].data
        other_data = self.other_map[0].data

        plt.rcParams.update({'font.size': 10})
        self.fig = plt.figure(figsize=(10,10))
        #Plot the UV map
        self.ax1 = self.fig.add_subplot(121, projection=self.wcs_map)
        self.ax1.set_facecolor('k')

        vmin, vmax = 0., np.max(data[data > 0.]*convert_flux)
        for key, value in [["cmap",[["cmap","inferno"]]], ["norm",[["vmin",vmin],["vmax",vmax]]]]:
            try:
                test = kwargs[key]
            except KeyError:
                for key_i, val_i in value:
                    kwargs[key_i] = val_i
        im1 = self.ax1.imshow(data*convert_flux, aspect='equal', **kwargs)

        px_size = self.wcs_map.wcs.get_cdelt()[0]*3600.
        px_sc = AnchoredSizeBar(self.ax1.transData, 1./px_size, '1 arcsec', 3, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color='w')
        self.ax1.add_artist(px_sc)

        try:
            north_dir1 = AnchoredDirectionArrows(self.ax1.transAxes, "E", "N", length=-0.08, fontsize=0.025, loc=1, aspect_ratio=-1, sep_y=0.01, sep_x=0.01, back_length=0., head_length=10., head_width=10., angle=-self.map[0].header['orientat'], color='white', text_props={'ec': None, 'fc': 'w', 'alpha': 1, 'lw': 0.4}, arrow_props={'ec': None,'fc':'w','alpha': 1,'lw': 1})
            self.ax1.add_artist(north_dir1)
        except KeyError:
            pass

        self.cr_map, = self.ax1.plot(*self.wcs_map.wcs.crpix, 'r+')

        self.ax1.set(xlabel="Right Ascension (J2000)", ylabel="Declination (J2000)", title="Click on selected point of reference.")

        #Plot the other map
        self.ax2 = self.fig.add_subplot(122, projection=self.wcs_other)
        self.ax2.set_facecolor('k')

        vmin, vmax = 0., np.max(other_data[other_data > 0.]*other_convert)
        for key, value in [["cmap",[["cmap","inferno"]]], ["norm",[["vmin",vmin],["vmax",vmax]]]]:
            try:
                test = kwargs[key]
            except KeyError:
                for key_i, val_i in value:
                    kwargs[key_i] = val_i
        im2 = self.ax2.imshow(other_data*other_convert, aspect='equal', **kwargs)

        fontprops = fm.FontProperties(size=16)
        px_size = self.wcs_other.wcs.get_cdelt()[0]*3600.
        px_sc = AnchoredSizeBar(self.ax2.transData, 1./px_size, '1 arcsec', 3, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color='w', fontproperties=fontprops)
        self.ax2.add_artist(px_sc)

        try:
            north_dir2 = AnchoredDirectionArrows(self.ax2.transAxes, "E", "N", length=-0.08, fontsize=0.03, loc=1, aspect_ratio=-1, sep_y=0.01, sep_x=0.01, angle=-self.other_map[0].header['orientat'], color='w', arrow_props={'ec': None, 'fc': 'w', 'alpha': 1,'lw': 2})
            self.ax2.add_artist(north_dir2)
        except KeyError:
            pass

        self.cr_other, = self.ax2.plot(*self.wcs_other.wcs.crpix, 'r+')

        self.ax2.set(xlabel="Right Ascension (J2000)", ylabel="Declination (J2000)", title="Click on selected point of reference.")

        #Selection button
        self.axapply = self.fig.add_axes([0.80, 0.01, 0.1, 0.04])
        self.bapply = Button(self.axapply, 'Apply reference')
        self.bapply.label.set_fontsize(8)
        self.axreset = self.fig.add_axes([0.60, 0.01, 0.1, 0.04])
        self.breset = Button(self.axreset, 'Leave as is')
        self.breset.label.set_fontsize(8)
        self.enter = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_key(self, event):
        if event.key.lower() == "enter":
            self.on_close_align(event)

    def get_aligned_wcs(self):
        return self.wcs_map, self.wcs_other

    def onclick_ref(self, event) -> None:
        if self.fig.canvas.manager.toolbar.mode == '':
            if (event.inaxes is not None) and (event.inaxes == self.ax1):
                x = event.xdata
                y = event.ydata

                self.cr_map.set(data=[x,y])
                self.fig.canvas.draw_idle()

            if (event.inaxes is not None) and (event.inaxes == self.ax2):
                x = event.xdata
                y = event.ydata

                self.cr_other.set(data=[x,y])
                self.fig.canvas.draw_idle()

    def reset_align(self, event):
        self.wcs_map.wcs.crpix = WCS(self.map[0].header).wcs.crpix[:2]
        self.wcs_other.wcs.crpix = WCS(self.other_map[0].header).wcs.crpix[:2]
        self.fig.canvas.draw_idle()

        if self.aligned:
            plt.close()

        self.aligned = True

    def apply_align(self, event=None):
        if np.array(self.cr_map.get_data()).shape == (2,1):
            self.wcs_map.wcs.crpix = np.array(self.cr_map.get_data())[:,0]
        else:
            self.wcs_map.wcs.crpix = np.array(self.cr_map.get_data())
        if np.array(self.cr_other.get_data()).shape == (2,1):
            self.wcs_other.wcs.crpix = np.array(self.cr_other.get_data())[:,0]
        else:
            self.wcs_other.wcs.crpix = np.array(self.cr_other.get_data())
        self.wcs_map.wcs.crval = np.array(self.wcs_map.pixel_to_world_values(*self.wcs_map.wcs.crpix))
        self.wcs_other.wcs.crval = self.wcs_map.wcs.crval
        self.fig.canvas.draw_idle()

        if self.aligned:
            plt.close()

        self.aligned = True

    def on_close_align(self, event):
        if not self.aligned:
            self.aligned = True
            self.apply_align()

    def align(self):
        self.fig.canvas.draw()
        self.fig.canvas.mpl_connect('button_press_event', self.onclick_ref)
        self.bapply.on_clicked(self.apply_align)
        self.breset.on_clicked(self.reset_align)
        self.fig.canvas.mpl_connect('close_event', self.on_close_align)
        plt.show(block=True)
        return self.get_aligned_wcs()


class overplot_radio(align_maps):
    """
    Class to overplot maps from different observations.
    Inherit from class align_maps in order to get the same WCS on both maps.
    """
    def overplot(self, other_levels, SNRp_cut=3., SNRi_cut=30., savename=None):
        self.Stokes_UV = self.map
        self.wcs_UV = self.wcs_map
        #Get Data
        obj = self.Stokes_UV[0].header['targname']
        stkI = self.Stokes_UV[np.argmax([self.Stokes_UV[i].header['datatype']=='I_stokes' for i in range(len(self.Stokes_UV))])]
        stk_cov = self.Stokes_UV[np.argmax([self.Stokes_UV[i].header['datatype']=='IQU_cov_matrix' for i in range(len(self.Stokes_UV))])]
        pol = self.Stokes_UV[np.argmax([self.Stokes_UV[i].header['datatype']=='Pol_deg_debiased' for i in range(len(self.Stokes_UV))])]
        pol_err = self.Stokes_UV[np.argmax([self.Stokes_UV[i].header['datatype']=='Pol_deg_err' for i in range(len(self.Stokes_UV))])]
        pang = self.Stokes_UV[np.argmax([self.Stokes_UV[i].header['datatype']=='Pol_ang' for i in range(len(self.Stokes_UV))])]

        other_data = self.other_map[0].data
        other_convert = 1.
        other_unit = self.other_map[0].header['bunit']
        if other_unit.lower() == 'jy/beam':
            other_unit = r"mJy/Beam"
            other_convert = 1e3
        other_freq = self.other_map[0].header['crval3']

        convert_flux = self.Stokes_UV[0].header['photflam']

        #Compute SNR and apply cuts
        pol.data[pol.data == 0.] = np.nan
        SNRp = pol.data/pol_err.data
        SNRp[np.isnan(SNRp)] = 0.
        pol.data[SNRp < SNRp_cut] = np.nan
        SNRi = stkI.data/np.sqrt(stk_cov.data[0,0])
        SNRi[np.isnan(SNRi)] = 0.
        pol.data[SNRi < SNRi_cut] = np.nan

        plt.rcParams.update({'font.size': 16})
        self.fig2 = plt.figure(figsize=(15,15))
        self.ax = self.fig2.add_subplot(111, projection=self.wcs_UV)
        self.ax.set_facecolor('k')
        self.fig2.subplots_adjust(hspace=0, wspace=0, right=0.9)

        #Display UV intensity map with polarization vectors
        vmin, vmax = 0., np.max(stkI.data[stkI.data > 0.]*convert_flux)
        im = self.ax.imshow(stkI.data*convert_flux, vmin=vmin, vmax=vmax, aspect='equal', cmap='inferno', alpha=1.)
        cbar_ax = self.fig2.add_axes([0.95, 0.12, 0.01, 0.75])
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")

        pol.data[np.isfinite(pol.data)] = 1./2.
        step_vec = 1
        X, Y = np.meshgrid(np.arange(stkI.data.shape[1]), np.arange(stkI.data.shape[0]))
        U, V = pol.data*np.cos(np.pi/2.+pang.data*np.pi/180.), pol.data*np.sin(np.pi/2.+pang.data*np.pi/180.)
        Q = self.ax.quiver(X[::step_vec,::step_vec],Y[::step_vec,::step_vec],U[::step_vec,::step_vec],V[::step_vec,::step_vec],units='xy',angles='uv',scale=0.5,scale_units='xy',pivot='mid',headwidth=0.,headlength=0.,headaxislength=0.,width=0.1,color='w')
        self.ax.autoscale(False)

        #Display other map as contours
        other_cont = self.ax.contour(other_data*other_convert, transform=self.ax.get_transform(self.wcs_other), levels=other_levels*other_convert, colors='grey')
        self.ax.clabel(other_cont, inline=True, fontsize=8)

        self.ax.set(xlabel="Right Ascension (J2000)", ylabel="Declination (J2000)", title="HST/FOC UV polarization map of {0:s} overplotted with {1:.2f}GHz map in {2:s}.".format(obj, other_freq*1e-9, other_unit))

        #Display pixel scale and North direction
        fontprops = fm.FontProperties(size=16)
        px_size = self.wcs_UV.wcs.get_cdelt()[0]*3600.
        px_sc = AnchoredSizeBar(self.ax.transData, 1./px_size, '1 arcsec', 3, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color='w', fontproperties=fontprops)
        self.ax.add_artist(px_sc)
        north_dir = AnchoredDirectionArrows(self.ax.transAxes, "E", "N", length=-0.08, fontsize=0.03, loc=1, aspect_ratio=-1, sep_y=0.01, sep_x=0.01, angle=-self.Stokes_UV[0].header['orientat'], color='w', arrow_props={'ec': None, 'fc': 'w', 'alpha': 1,'lw': 2})
        self.ax.add_artist(north_dir)

        self.cr_map, = self.ax.plot(*self.wcs_map.wcs.crpix, 'r+')
        crpix_other = self.wcs_map.world_to_pixel(self.wcs_other.pixel_to_world(*self.wcs_other.wcs.crpix))
        self.cr_other, = self.ax.plot(*crpix_other, 'g+')

        if not(savename is None):
            self.fig2.savefig(savename,bbox_inches='tight',dpi=200)

        self.fig2.canvas.draw()

    def plot(self, levels, SNRp_cut=3., SNRi_cut=30., savename=None) -> None:
        while not self.aligned:
            self.align()
        self.overplot(other_levels=levels, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, savename=savename)
        plt.show(block=True)


class overplot_pol(align_maps):
    """
    Class to overplot maps from different observations.
    Inherit from class align_maps in order to get the same WCS on both maps.
    """
    def overplot(self, SNRp_cut=3., SNRi_cut=30., savename=None, **kwargs):
        self.Stokes_UV = self.map
        self.wcs_UV = self.wcs_map
        #Get Data
        obj = self.Stokes_UV[0].header['targname']
        stkI = self.Stokes_UV[np.argmax([self.Stokes_UV[i].header['datatype']=='I_stokes' for i in range(len(self.Stokes_UV))])]
        stk_cov = self.Stokes_UV[np.argmax([self.Stokes_UV[i].header['datatype']=='IQU_cov_matrix' for i in range(len(self.Stokes_UV))])]
        pol = self.Stokes_UV[np.argmax([self.Stokes_UV[i].header['datatype']=='Pol_deg_debiased' for i in range(len(self.Stokes_UV))])]
        pol_err = self.Stokes_UV[np.argmax([self.Stokes_UV[i].header['datatype']=='Pol_deg_err' for i in range(len(self.Stokes_UV))])]
        pang = self.Stokes_UV[np.argmax([self.Stokes_UV[i].header['datatype']=='Pol_ang' for i in range(len(self.Stokes_UV))])]

        convert_flux = self.Stokes_UV[0].header['photflam']

        other_data = self.other_map[0].data
        try:
            other_convert = self.other_map[0].header['photflam']
        except KeyError:
            other_convert = 1.

        #Compute SNR and apply cuts
        pol.data[pol.data == 0.] = np.nan
        SNRp = pol.data/pol_err.data
        SNRp[np.isnan(SNRp)] = 0.
        pol.data[SNRp < SNRp_cut] = np.nan
        SNRi = stkI.data/np.sqrt(stk_cov.data[0,0])
        SNRi[np.isnan(SNRi)] = 0.
        pol.data[SNRi < SNRi_cut] = np.nan

        plt.rcParams.update({'font.size': 16})
        self.fig2 = plt.figure(figsize=(15,15))
        self.ax = self.fig2.add_subplot(111, projection=self.wcs_UV)
        self.ax.set_facecolor('k')
        self.fig2.subplots_adjust(hspace=0, wspace=0, right=0.9)

        #Display Stokes I as contours
        levels_stkI = np.rint(np.linspace(10,99,10))/100.*np.max(stkI.data[stkI.data > 0.]*convert_flux)
        cont_stkI = self.ax.contour(stkI.data*convert_flux, transform=self.ax.get_transform(self.wcs_UV), levels=levels_stkI, colors='grey', alpha=0.5)
        #self.ax.clabel(cont_stkI, inline=True, fontsize=8)

        self.ax.autoscale(False)

        #Display full size polarization vectors
        pol.data[np.isfinite(pol.data)] = 1./2.
        step_vec = 1
        X, Y = np.meshgrid(np.arange(stkI.data.shape[1]), np.arange(stkI.data.shape[0]))
        U, V = pol.data*np.cos(np.pi/2.+pang.data*np.pi/180.), pol.data*np.sin(np.pi/2.+pang.data*np.pi/180.)
        Q = self.ax.quiver(X[::step_vec,::step_vec],Y[::step_vec,::step_vec],U[::step_vec,::step_vec],V[::step_vec,::step_vec],units='xy',angles='uv',scale=0.5,scale_units='xy',pivot='mid',headwidth=0.,headlength=0.,headaxislength=0.,width=0.1,color='w')

        #Display "other" intensity map
        vmin, vmax = 0., np.max(other_data[other_data > 0.]*other_convert)
        for key, value in [["cmap",[["cmap","inferno"]]], ["norm",[["vmin",vmin],["vmax",vmax]]]]:
            try:
                test = kwargs[key]
            except KeyError:
                for key_i, val_i in value:
                    kwargs[key_i] = val_i
        im = self.ax.imshow(other_data*other_convert, transform=self.ax.get_transform(self.wcs_other), alpha=1., **kwargs)
        cbar_ax = self.fig2.add_axes([0.95, 0.12, 0.01, 0.75])
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")

        self.ax.set(xlabel="Right Ascension (J2000)", ylabel="Declination (J2000)", title="{0:s} overplotted with polarization vectors and Stokes I contours from HST/FOC".format(obj))

        #Display pixel scale and North direction
        fontprops = fm.FontProperties(size=16)
        px_size = self.wcs_UV.wcs.get_cdelt()[0]*3600.
        px_sc = AnchoredSizeBar(self.ax.transData, 1./px_size, '1 arcsec', 3, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color='w', fontproperties=fontprops)
        self.ax.add_artist(px_sc)
        north_dir = AnchoredDirectionArrows(self.ax.transAxes, "E", "N", length=-0.08, fontsize=0.03, loc=1, aspect_ratio=-1, sep_y=0.01, sep_x=0.01, angle=-self.Stokes_UV[0].header['orientat'], color='w', arrow_props={'ec': None, 'fc': 'w', 'alpha': 1,'lw': 2})
        self.ax.add_artist(north_dir)

        self.cr_map, = self.ax.plot(*self.wcs_map.wcs.crpix, 'r+')
        crpix_other = self.wcs_map.world_to_pixel(self.wcs_other.pixel_to_world(*self.wcs_other.wcs.crpix))
        self.cr_other, = self.ax.plot(*crpix_other, 'g+')

        if not(savename is None):
            self.fig2.savefig(savename,bbox_inches='tight',dpi=200)

        self.fig2.canvas.draw()

    def plot(self, SNRp_cut=3., SNRi_cut=30., savename=None, **kwargs) -> None:
        while not self.aligned:
            self.align()
        self.overplot(SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, savename=savename, **kwargs)
        plt.show(block=True)


class align_pol(object):
    def __init__(self, maps, **kwargs):
        order = np.argsort(np.array([curr[0].header['mjd-obs'] for curr in maps]))
        maps = np.array(maps)[order]
        self.ref_map, self.other_maps = maps[0], maps[1:]

        self.wcs = WCS(self.ref_map[0].header)
        self.wcs_other = np.array([WCS(map[0].header) for map in self.other_maps])

        self.aligned = np.zeros(self.other_maps.shape[0], dtype=bool)

        self.kwargs = kwargs
    
    def single_plot(self, curr_map, wcs, v_lim=None, ax_lim=None, SNRp_cut=3., SNRi_cut=30., savename=None, **kwargs):
        #Get data
        stkI = curr_map[np.argmax([curr_map[i].header['datatype']=='I_stokes' for i in range(len(curr_map))])]
        stkQ = curr_map[np.argmax([curr_map[i].header['datatype']=='Q_stokes' for i in range(len(curr_map))])]
        stkU = curr_map[np.argmax([curr_map[i].header['datatype']=='U_stokes' for i in range(len(curr_map))])]
        stk_cov = curr_map[np.argmax([curr_map[i].header['datatype']=='IQU_cov_matrix' for i in range(len(curr_map))])]
        pol = curr_map[np.argmax([curr_map[i].header['datatype']=='Pol_deg_debiased' for i in range(len(curr_map))])]
        pol_err = curr_map[np.argmax([curr_map[i].header['datatype']=='Pol_deg_err' for i in range(len(curr_map))])]
        pang = curr_map[np.argmax([curr_map[i].header['datatype']=='Pol_ang' for i in range(len(curr_map))])]
        try:
            data_mask = curr_map[np.argmax([curr_map[i].header['datatype']=='Data_mask' for i in range(len(curr_map))])].data.astype(bool)
        except KeyError:
            data_mask = np.ones(stkI.shape).astype(bool)

        pivot_wav = curr_map[0].header['photplam']
        convert_flux = curr_map[0].header['photflam']

        #Compute SNR and apply cuts
        pol.data[pol.data == 0.] = np.nan
        pol_err.data[pol_err.data == 0.] = np.nan
        SNRp = pol.data/pol_err.data
        SNRp[np.isnan(SNRp)] = 0.
        pol.data[SNRp < SNRp_cut] = np.nan

        maskI = stk_cov.data[0,0] > 0
        SNRi = np.zeros(stkI.data.shape)
        SNRi[maskI] = stkI.data[maskI]/np.sqrt(stk_cov.data[0,0][maskI])
        pol.data[SNRi < SNRi_cut] = np.nan

        mask = (SNRp > SNRp_cut) * (SNRi > SNRi_cut)

        #Plot the map
        plt.rcParams.update({'font.size': 10})
        plt.rcdefaults()
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection=wcs)
        ax.set(xlabel="Right Ascension (J2000)", ylabel="Declination (J2000)", facecolor='k',
            title="target {0:s} observed on {1:s}".format(curr_map[0].header['targname'], curr_map[0].header['date-obs']))
        fig.subplots_adjust(hspace=0, wspace=0, right=0.9)
        cbar_ax = fig.add_axes([0.95, 0.12, 0.01, 0.75])

        if not ax_lim is None:
            lim = np.concatenate([wcs.world_to_pixel(ax_lim[i]) for i in range(len(ax_lim))])
            x_lim, y_lim = lim[0::2], lim[1::2]
            ax.set(xlim=x_lim,ylim=y_lim)

        if v_lim is None:
            vmin, vmax = 0., np.max(stkI.data[stkI.data > 0.]*convert_flux)
        else:
            vmin, vmax = v_lim*convert_flux
        
        for key, value in [["cmap",[["cmap","inferno"]]], ["norm",[["vmin",vmin],["vmax",vmax]]]]:
            try:
                test = kwargs[key]
                if str(type(test)) == "<class 'matplotlib.colors.LogNorm'>":
                    kwargs[key] = LogNorm(vmin, vmax)
            except KeyError:
                for key_i, val_i in value:
                    kwargs[key_i] = val_i

        im = ax.imshow(stkI.data*convert_flux, aspect='equal', **kwargs)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")

        px_size = wcs.wcs.get_cdelt()[0]*3600.
        px_sc = AnchoredSizeBar(ax.transData, 1./px_size, '1 arcsec', 3, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color='w')
        ax.add_artist(px_sc)

        north_dir = AnchoredDirectionArrows(ax.transAxes, "E", "N", length=-0.08, fontsize=0.025, loc=1, aspect_ratio=-1, sep_y=0.01, sep_x=0.01, back_length=0., head_length=10., head_width=10., angle=curr_map[0].header['orientat'], color='white', text_props={'ec': None, 'fc': 'w', 'alpha': 1, 'lw': 0.4}, arrow_props={'ec': None,'fc':'w','alpha': 1,'lw': 1})
        ax.add_artist(north_dir)
        
        step_vec = 1
        X, Y = np.meshgrid(np.arange(stkI.data.shape[1]), np.arange(stkI.data.shape[0]))
        U, V = pol.data*np.cos(np.pi/2.+pang.data*np.pi/180.), pol.data*np.sin(np.pi/2.+pang.data*np.pi/180.)
        Q = ax.quiver(X[::step_vec,::step_vec],Y[::step_vec,::step_vec],U[::step_vec,::step_vec],V[::step_vec,::step_vec],units='xy',angles='uv',scale=0.5,scale_units='xy',pivot='mid',headwidth=0.,headlength=0.,headaxislength=0.,width=0.1,color='w')
        pol_sc = AnchoredSizeBar(ax.transData, 2., r"$P$= 100 %", 4, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color='w')
        ax.add_artist(pol_sc)

        if not savename is None:
            fig.savefig(savename+".png",bbox_inches='tight',dpi=300)

        plt.show(block=True)
        return fig, ax

    def align(self):
        for i, curr_map in enumerate(self.other_maps):
            curr_align = align_maps(self.ref_map, curr_map, **self.kwargs)
            self.wcs, self.wcs_other[i] = curr_align.align()
            self.aligned[i] = curr_align.aligned
    
    def plot(self, SNRp_cut=3., SNRi_cut=30., savename=None, **kwargs):
        while not self.aligned.all():
            self.align()
        eps = 1e-35
        vmin = np.min([np.min(curr_map[0].data[curr_map[0].data > SNRi_cut*np.max([eps*np.ones(curr_map[0].data.shape),np.sqrt(curr_map[3].data[0,0])],axis=0)]) for curr_map in self.other_maps])/2.5
        vmax = np.max([np.max(curr_map[0].data[curr_map[0].data > SNRi_cut*np.max([eps*np.ones(curr_map[0].data.shape),np.sqrt(curr_map[3].data[0,0])],axis=0)]) for curr_map in self.other_maps])
        vmin = np.min([vmin, np.min(self.ref_map[0].data[self.ref_map[0].data > SNRi_cut*np.max([eps*np.ones(self.ref_map[0].data.shape),np.sqrt(self.ref_map[3].data[0,0])],axis=0)])])/2.5
        vmax = np.max([vmax, np.max(self.ref_map[0].data[self.ref_map[0].data > SNRi_cut*np.max([eps*np.ones(self.ref_map[0].data.shape),np.sqrt(self.ref_map[3].data[0,0])],axis=0)])])
        v_lim = np.array([vmin, vmax])

        fig, ax = self.single_plot(self.ref_map, self.wcs, v_lim = v_lim, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, savename=savename+'_0', **kwargs)
        x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
        ax_lim = np.array([self.wcs.pixel_to_world(x_lim[i], y_lim[i]) for i in range(len(x_lim))])
        
        for i, curr_map in enumerate(self.other_maps):
            self.single_plot(curr_map, self.wcs_other[i], v_lim=v_lim, ax_lim=ax_lim, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, savename=savename+'_'+str(i+1), **kwargs)


class crop_map(object):
    """
    Class to interactively crop a map to desired Region of Interest
    """
    def __init__(self, hdul, fig=None, ax=None):
        #Get data
        self.cropped=False
        self.hdul = hdul
        self.header = deepcopy(self.hdul[0].header)
        self.wcs = WCS(self.header).deepcopy()

        self.data = deepcopy(self.hdul[0].data)
        try:
            self.convert_flux = self.header['photflam']
        except KeyError:
            self.convert_flux = 1.

        #Plot the map
        plt.rcParams.update({'font.size': 12})
        plt.ioff()
        if fig is None:
            self.fig = plt.figure(figsize=(15,15))
            self.fig.suptitle("Click and drag to crop to desired Region of Interest.")
        else:
            self.fig = fig
        if ax is None:
            self.ax = self.fig.add_subplot(111, projection=self.wcs)
            self.mask_alpha=1.
            #Selection button
            self.axapply = self.fig.add_axes([0.80, 0.01, 0.1, 0.04])
            self.bapply = Button(self.axapply, 'Apply')
            self.axreset = self.fig.add_axes([0.60, 0.01, 0.1, 0.04])
            self.breset = Button(self.axreset, 'Reset')
            self.embedded = False
        else:
            self.ax = ax
            self.mask_alpha = 0.75
            self.rect_selector = RectangleSelector(self.ax, self.onselect_crop,
                    drawtype='box', button=[1], interactive=True)
            self.embedded = True
        self.display()
        plt.ion()

        self.extent = np.array([0.,self.data.shape[0],0., self.data.shape[1]])
        self.center = np.array(self.data.shape)/2
        self.RSextent = deepcopy(self.extent)
        self.RScenter = deepcopy(self.center)

        plt.show()

    def display(self, data=None, wcs=None, convert_flux=None):
        if data is None:
            data = self.data
        if wcs is None:
            wcs = self.wcs
        if convert_flux is None:
            convert_flux = self.convert_flux

        vmin, vmax = 0., np.max(data[data > 0.]*convert_flux)
        if hasattr(self, 'im'):
            self.im.remove()
        self.im = self.ax.imshow(data*convert_flux, vmin=vmin, vmax=vmax, aspect='equal', cmap='inferno', alpha=self.mask_alpha, origin='lower')
        if hasattr(self, 'cr'):
            self.cr[0].set_data(*wcs.wcs.crpix)
        else:
            self.cr = self.ax.plot(*wcs.wcs.crpix, 'r+')
        self.fig.canvas.draw_idle()
        return self.im

    @property
    def crpix_in_RS(self):
        crpix = self.wcs.wcs.crpix
        x_lim, y_lim = self.RSextent[:2], self.RSextent[2:]
        if (crpix[0] > x_lim[0] and crpix[0] < x_lim[1]):
            if (crpix[1] > y_lim[0] and crpix[1] < y_lim[1]):
                return True
        return False

    def reset_crop(self, event):
        self.ax.reset_wcs(self.wcs)
        if hasattr(self, 'hdul_crop'):
            del self.hdul_crop, self.data_crop
        self.display()

        if self.fig.canvas.manager.toolbar.mode == '':
            self.rect_selector = RectangleSelector(self.ax, self.onselect_crop,
                    drawtype='box', button=[1], interactive=True)

        self.RSextent = deepcopy(self.extent)
        self.RScenter = deepcopy(self.center)
        self.ax.set_xlim(*self.extent[:2])
        self.ax.set_ylim(*self.extent[2:])
        self.fig.canvas.draw_idle()

    def onselect_crop(self, eclick, erelease) -> None:
        # Obtain (xmin, xmax, ymin, ymax) values
        self.RSextent = np.array(self.rect_selector.extents)
        self.RScenter = np.array(self.rect_selector.center)
        if self.embedded:
            self.apply_crop(erelease)

    def apply_crop(self, event):
        if hasattr(self, 'hdul_crop'):
            header = self.header_crop
            data = self.data_crop
            wcs = self.wcs_crop
        else:
            header = self.header
            data = self.data
            wcs = self.wcs

        vertex = self.RSextent.astype(int)
        shape = vertex[1::2] - vertex[0::2]

        extent = np.array(self.im.get_extent())
        shape_im = extent[1::2] - extent[0::2]
        if (shape_im.astype(int) != shape).any() and (self.RSextent != self.extent).any():
            #Update WCS and header in new cropped image
            crpix = np.array(wcs.wcs.crpix)
            self.wcs_crop = wcs.deepcopy()
            self.wcs_crop.array_shape = shape
            if self.crpix_in_RS:
                self.wcs_crop.wcs.crpix = np.array(self.wcs_crop.wcs.crpix) - self.RSextent[::2]
            else:
                self.wcs_crop.wcs.crval = wcs.wcs_pix2world([self.RScenter],1)[0]
                self.wcs_crop.wcs.crpix = self.RScenter-self.RSextent[::2]

            # Crop dataset
            self.data_crop = deepcopy(data[vertex[2]:vertex[3], vertex[0]:vertex[1]])

            #Write cropped map to new HDUList
            self.header_crop = deepcopy(header)
            self.header_crop.update(self.wcs_crop.to_header())
            self.hdul_crop = fits.HDUList([fits.PrimaryHDU(self.data_crop,self.header_crop)])

            try:
                convert_flux = self.header_crop['photflam']
            except KeyError:
                convert_flux = 1.

            self.rect_selector.clear()
            self.ax.reset_wcs(self.wcs_crop)
            self.display(data=self.data_crop, wcs=self.wcs_crop, convert_flux=convert_flux)

            xlim, ylim = self.RSextent[1::2]-self.RSextent[0::2]
            self.ax.set_xlim(0,xlim)
            self.ax.set_ylim(0,ylim)

            if self.fig.canvas.manager.toolbar.mode == '':
                self.rect_selector = RectangleSelector(self.ax, self.onselect_crop,
                        drawtype='box', button=[1], interactive=True)
        self.fig.canvas.draw_idle()

    def on_close(self, event) -> None:
        if not hasattr(self, 'hdul_crop'):
            self.hdul_crop = self.hdul
        self.rect_selector.disconnect_events()
        self.cropped = True

    def crop(self) -> None:
        if self.fig.canvas.manager.toolbar.mode == '':
            self.rect_selector = RectangleSelector(self.ax, self.onselect_crop,
                    drawtype='box', button=[1], interactive=True)
        self.bapply.on_clicked(self.apply_crop)
        self.breset.on_clicked(self.reset_crop)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        plt.show()

    def writeto(self, filename):
        self.hdul_crop.writeto(filename,overwrite=True)


class crop_Stokes(crop_map):
    """
    Class to interactively crop a polarization map to desired Region of Interest.
    Inherit from crop_map.
    """
    def apply_crop(self,event):
        """
        Redefine apply_crop method for the Stokes HDUList.
        """
        if hasattr(self, 'hdul_crop'):
            hdul = self.hdul_crop
            data = self.data_crop
            wcs = self.wcs_crop
        else:
            hdul = self.hdul
            data = self.data
            wcs = self.wcs

        vertex = self.RSextent.astype(int)
        shape = vertex[1::2] - vertex[0::2]

        extent = np.array(self.im.get_extent())
        shape_im = extent[1::2] - extent[0::2]
        if (shape_im.astype(int) != shape).any() and (self.RSextent != self.extent).any():
            #Update WCS and header in new cropped image
            self.hdul_crop = deepcopy(hdul)
            crpix = np.array(wcs.wcs.crpix)
            self.wcs_crop = wcs.deepcopy()
            self.wcs_crop.array_shape = shape
            if self.crpix_in_RS:
                self.wcs_crop.wcs.crpix = np.array(self.wcs_crop.wcs.crpix) - self.RSextent[::2]
            else:
                self.wcs_crop.wcs.crval = wcs.wcs_pix2world([self.RScenter],1)[0]
                self.wcs_crop.wcs.crpix = self.RScenter-self.RSextent[::2]

            # Crop dataset
            for dataset in self.hdul_crop:
                if dataset.header['datatype']=='IQU_cov_matrix':
                    stokes_cov = np.zeros((3,3,shape[1],shape[0]))
                    for i in range(3):
                        for j in range(3):
                            stokes_cov[i,j] = deepcopy(dataset.data[i,j][vertex[2]:vertex[3], vertex[0]:vertex[1]])
                    dataset.data = stokes_cov
                else:
                    dataset.data = deepcopy(dataset.data[vertex[2]:vertex[3], vertex[0]:vertex[1]])
                dataset.header.update(self.wcs_crop.to_header())

            try:
                convert_flux = self.hdul_crop[0].header['photflam']
            except KeyError:
                convert_flux = 1.

            self.data_crop = self.hdul_crop[0].data
            self.rect_selector.clear()
            if not self.embedded:
                self.ax.reset_wcs(self.wcs_crop)
                self.display(data=self.data_crop, wcs=self.wcs_crop, convert_flux=convert_flux)

                xlim, ylim = self.RSextent[1::2]-self.RSextent[0::2]
                self.ax.set_xlim(0,xlim)
                self.ax.set_ylim(0,ylim)
            else:
                self.on_close(event)

            if self.fig.canvas.manager.toolbar.mode == '':
                self.rect_selector = RectangleSelector(self.ax, self.onselect_crop,
                        drawtype='box', button=[1], interactive=True)
        self.fig.canvas.draw_idle()

    @property
    def data_mask(self):
        return self.hdul_crop[-1].data


class image_lasso_selector(object):
    def __init__(self, img, fig=None, ax=None):
        """
        img must have shape (X, Y)
        """
        self.selected = False
        self.img = img
        self.vmin, self.vmax = 0., np.max(self.img[self.img>0.])
        plt.ioff() # see https://github.com/matplotlib/matplotlib/issues/17013
        if fig is None:
            self.fig = plt.figure(figsize=(15,15))
        else:
            self.fig = fig
        if ax is None:
            self.ax = self.fig.gca()
            self.mask_alpha = 1.
            self.embedded = False
        else:
            self.ax = ax
            self.mask_alpha = 0.1
            self.embedded = True
        self.displayed = self.ax.imshow(self.img, vmin=self.vmin, vmax=self.vmax, aspect='equal', cmap='inferno',alpha=self.mask_alpha)
        plt.ion()

        lineprops = {'color': 'grey', 'linewidth': 1, 'alpha': 0.8}
        self.lasso = LassoSelector(self.ax, self.onselect, props=lineprops, useblit=False)
        self.lasso.set_visible(True)

        pix_x = np.arange(self.img.shape[0])
        pix_y = np.arange(self.img.shape[1])
        xv, yv = np.meshgrid(pix_y,pix_x)
        self.pix = np.vstack( (xv.flatten(), yv.flatten()) ).T

        self.fig.canvas.mpl_connect('close_event', self.on_close)
        plt.show()

    def on_close(self, event=None) -> None:
        if not hasattr(self, 'mask'):
            self.mask = np.zeros(self.img.shape[:2],dtype=bool)
        self.lasso.disconnect_events()
        self.selected = True

    def onselect(self, verts):
        self.verts = verts
        p = Path(verts)
        self.indices = p.contains_points(self.pix, radius=0).reshape(self.img.shape[:2])
        self.update_mask()

    def update_mask(self):
        self.displayed.remove()
        self.displayed = self.ax.imshow(self.img, vmin=self.vmin, vmax=self.vmax, aspect='equal', cmap='inferno',alpha=self.mask_alpha)
        array = self.displayed.get_array().data

        self.mask = np.zeros(self.img.shape[:2],dtype=bool)
        self.mask[self.indices] = True
        if hasattr(self, 'cont'):
            for coll in self.cont.collections:
                coll.remove()
        self.cont = self.ax.contour(self.mask.astype(float),levels=[0.5], colors='white', linewidths=1)
        if not self.embedded:
            self.displayed.set_data(array)
            self.fig.canvas.draw_idle()
        else:
            self.on_close()


class aperture(object):
    def __init__(self, img, cdelt=np.array([1.,1.]), radius=1., fig=None, ax=None):
        """
        img must have shape (X, Y)
        """
        self.selected = False
        self.img = img
        self.vmin, self.vmax = 0., np.max(self.img[self.img>0.])
        plt.ioff() # see https://github.com/matplotlib/matplotlib/issues/17013
        if fig is None:
            self.fig = plt.figure(figsize=(15,15))
        else:
            self.fig = fig
        if ax is None:
            self.ax = self.fig.gca()
            self.mask_alpha = 1.
            self.embedded = False
        else:
            self.ax = ax
            self.mask_alpha = 0.1
            self.embedded = True
        
        self.displayed = self.ax.imshow(self.img, vmin=self.vmin, vmax=self.vmax, aspect='equal', cmap='inferno',alpha=self.mask_alpha)
        plt.ion()

        xx, yy = np.indices(self.img.shape)
        self.pix = np.vstack( (xx.flatten(), yy.flatten()) ).T

        self.x0, self.y0 = np.array(self.img.shape)/2.
        if np.abs(cdelt).max() != 1.:
            self.cdelt = cdelt
            self.radius = radius/np.abs(self.cdelt).max()/3600.

        self.circ = Circle((self.x0, self.y0), self.radius, alpha=0.8, ec='grey',fc='none')
        self.ax.add_patch(self.circ)       
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.x0, self.y0 = self.circ.center
        self.pressevent = None
        plt.show()

    def on_close(self, event=None) -> None:
        if not hasattr(self, 'mask'):
            self.mask = np.zeros(self.img.shape[:2],dtype=bool)
        self.selected = True

    def on_press(self, event):
        if event.inaxes != self.ax:
            return

        if not self.circ.contains(event)[0]:
            return

        self.pressevent = event

    def on_release(self, event):
        self.pressevent = None
        self.x0, self.y0 = self.circ.center
        self.update_mask()

    def on_move(self, event):
        if self.pressevent is None or event.inaxes != self.pressevent.inaxes:
            return

        dx = event.xdata - self.pressevent.xdata
        dy = event.ydata - self.pressevent.ydata
        self.circ.center = self.x0 + dx, self.y0 + dy
        self.fig.canvas.draw_idle()
    
    def update_radius(self, radius):
        self.radius = radius/np.abs(self.cdelt).max()/3600
        self.circ.set_radius(self.radius)
        self.fig.canvas.draw_idle()

    def update_mask(self):
        if hasattr(self, 'displayed'):
            try:
                self.displayed.remove()
            except:
                return
        self.displayed = self.ax.imshow(self.img, vmin=self.vmin, vmax=self.vmax, aspect='equal', cmap='inferno',alpha=self.mask_alpha)
        array = self.displayed.get_array().data

        yy, xx = np.indices(self.img.shape[:2])
        x0, y0 = self.circ.center
        self.mask = np.sqrt((xx-x0)**2+(yy-y0)**2) < self.radius
        if hasattr(self, 'cont'):
            for coll in self.cont.collections:
                try:
                    coll.remove()
                except:
                    return
        self.cont = self.ax.contour(self.mask.astype(float),levels=[0.5], colors='white', linewidths=1)
        if not self.embedded:
            self.displayed.set_data(array)
            self.fig.canvas.draw_idle()
        else:
            self.on_close()


class pol_map(object):
    """
    Class to interactively study polarization maps.
    """
    def __init__(self, Stokes, SNRp_cut=3., SNRi_cut=30., selection=None):

        if type(Stokes) == str:
            Stokes = fits.open(Stokes)
        self.Stokes = deepcopy(Stokes)
        self.SNRp_cut = SNRp_cut
        self.SNRi_cut = SNRi_cut
        self.SNRi = deepcopy(self.SNRi_cut)
        self.SNRp = deepcopy(self.SNRp_cut)
        self.region = None
        self.data = None
        self.display_selection = selection

        #Get data
        self.targ = self.Stokes[0].header['targname']
        self.pivot_wav = self.Stokes[0].header['photplam']
        self.convert_flux = self.Stokes[0].header['photflam']

        #Create figure
        plt.rcParams.update({'font.size': 10})
        self.fig = plt.figure(figsize=(10,10))
        self.fig.subplots_adjust(hspace=0, wspace=0, right=0.88)
        self.ax = self.fig.add_subplot(111,projection=self.wcs)
        self.ax_cosmetics()
        self.cbar_ax = self.fig.add_axes([0.925, 0.13, 0.01, 0.74])

        #Display selected data (Default to total flux)
        self.display()
        #Display polarization vectors in SNR_cut
        self.pol_vector()
        #Display integrated values in ROI
        self.pol_int()

        #Set axes for sliders (SNRp_cut, SNRi_cut)
        ax_I_cut = self.fig.add_axes([0.125, 0.080, 0.35, 0.01])
        ax_P_cut = self.fig.add_axes([0.125, 0.055, 0.35, 0.01])
        ax_snr_reset = self.fig.add_axes([0.125, 0.020, 0.05, 0.02])
        SNRi_max = np.max(self.I[self.IQU_cov[0,0]>0.]/np.sqrt(self.IQU_cov[0,0][self.IQU_cov[0,0]>0.]))
        SNRp_max = np.max(self.P[self.s_P>0.]/self.s_P[self.s_P > 0.])
        s_I_cut = Slider(ax_I_cut,r"$SNR^{I}_{cut}$",1.,int(SNRi_max*0.95),valstep=1,valinit=self.SNRi_cut)
        s_P_cut = Slider(ax_P_cut,r"$SNR^{P}_{cut}$",1.,int(SNRp_max*0.95),valstep=1,valinit=self.SNRp_cut)
        b_snr_reset = Button(ax_snr_reset,"Reset")
        b_snr_reset.label.set_fontsize(8)

        def update_snri(val):
            self.SNRi = val
            self.pol_vector()
            self.pol_int()
            self.fig.canvas.draw_idle()

        def update_snrp(val):
            self.SNRp = val
            self.pol_vector()
            self.pol_int()
            self.fig.canvas.draw_idle()

        def reset_snr(event):
            s_I_cut.reset()
            s_P_cut.reset()

        s_I_cut.on_changed(update_snri)
        s_P_cut.on_changed(update_snrp)
        b_snr_reset.on_clicked(reset_snr)

        #Set axe for Aperture selection
        ax_aper = self.fig.add_axes([0.55, 0.040, 0.05, 0.02])
        ax_aper_reset = self.fig.add_axes([0.605, 0.040, 0.05, 0.02])
        ax_aper_radius = self.fig.add_axes([0.55, 0.020, 0.10, 0.01])
        self.selected = False
        b_aper = Button(ax_aper,"Aperture")
        b_aper.label.set_fontsize(8)
        b_aper_reset = Button(ax_aper_reset,"Reset")
        b_aper_reset.label.set_fontsize(8)
        s_aper_radius = Slider(ax_aper_radius, r"$R_{aper}$", 0.5, 3.5, valstep=0.1, valinit=1)

        def select_aperture(event):
            if self.data is None:
                self.data = self.Stokes[0].data
            if self.selected:
                self.selected = False
                self.select_instance.update_mask()
                self.region = deepcopy(self.select_instance.mask.astype(bool))
                self.select_instance.displayed.remove()
                for coll in self.select_instance.cont.collections[:]:
                    coll.remove()
                self.select_instance.circ.set_visible(False)
                self.set_data_mask(deepcopy(self.region))
                self.pol_int()
            else:
                self.selected = True
                self.region = None
                self.select_instance = aperture(self.data, fig=self.fig, ax=self.ax, cdelt=self.wcs.wcs.cdelt, radius=s_aper_radius.val)
                self.select_instance.circ.set_visible(True)

            self.fig.canvas.draw_idle()
        
        def update_aperture(val):
            if hasattr(self, 'select_instance'):
                if hasattr(self.select_instance, 'radius'):
                    self.select_instance.update_radius(val)
                else:
                    self.selected = True
                    self.select_instance = aperture(self.data, fig=self.fig, ax=self.ax, cdelt=self.wcs.wcs.cdelt, radius=val)
            else:
                self.selected = True
                self.select_instance = aperture(self.data, fig=self.fig, ax=self.ax, cdelt=self.wcs.wcs.cdelt, radius=val)
            self.fig.canvas.draw_idle()
                    

        def reset_aperture(event):
            self.region = None
            self.pol_int()
            self.fig.canvas.draw_idle()

        b_aper.on_clicked(select_aperture)
        b_aper_reset.on_clicked(reset_aperture)
        s_aper_radius.on_changed(update_aperture)

        #Set axe for ROI selection
        ax_select = self.fig.add_axes([0.55, 0.070, 0.05, 0.02])
        ax_roi_reset = self.fig.add_axes([0.605, 0.070, 0.05, 0.02])
        b_select = Button(ax_select,"Select")
        b_select.label.set_fontsize(8)
        self.selected = False
        b_roi_reset = Button(ax_roi_reset,"Reset")
        b_roi_reset.label.set_fontsize(8)

        def select_roi(event):
            if self.data is None:
                self.data = self.Stokes[0].data
            if self.selected:
                self.selected = False
                self.region = deepcopy(self.select_instance.mask.astype(bool))
                self.select_instance.displayed.remove()
                for coll in self.select_instance.cont.collections:
                    coll.remove()
                self.select_instance.lasso.set_active(False)
                self.set_data_mask(deepcopy(self.region))
                self.pol_int()
            else:
                self.selected = True
                self.region = None
                self.select_instance = image_lasso_selector(self.data, fig=self.fig, ax=self.ax)
                self.select_instance.lasso.set_active(True)
                k = 0
                while not self.select_instance.selected and k<60:
                    self.fig.canvas.start_event_loop(timeout=1)
                    k+=1
                select_roi(event)
            self.fig.canvas.draw_idle()

        def reset_roi(event):
            self.region = None
            self.pol_int()
            self.fig.canvas.draw_idle()

        b_select.on_clicked(select_roi)
        b_roi_reset.on_clicked(reset_roi)

        #Set axe for crop Stokes
        ax_crop = self.fig.add_axes([0.70, 0.070, 0.05, 0.02])
        ax_crop_reset = self.fig.add_axes([0.755, 0.070, 0.05, 0.02])
        b_crop = Button(ax_crop,"Crop")
        b_crop.label.set_fontsize(8)
        self.cropped = False
        b_crop_reset = Button(ax_crop_reset,"Reset")
        b_crop_reset.label.set_fontsize(8)

        def crop(event):
            if self.cropped:
                self.cropped = False
                self.crop_instance.im.remove()
                self.crop_instance.cr.pop(0).remove()
                self.crop_instance.rect_selector.set_active(False)
                self.Stokes = self.crop_instance.hdul_crop
                self.region = deepcopy(self.data_mask.astype(bool))
                self.pol_int()
                self.ax.reset_wcs(self.wcs)
                self.ax_cosmetics()
                self.display()
                self.ax.set_xlim(0,self.I.shape[1])
                self.ax.set_ylim(0,self.I.shape[0])
                self.pol_vector()
            else:
                self.cropped = True
                self.crop_instance = crop_Stokes(self.Stokes, fig=self.fig, ax=self.ax)
                self.crop_instance.rect_selector.set_active(True)
                k = 0
                while not self.crop_instance.cropped and k<60:
                    self.fig.canvas.start_event_loop(timeout=1)
                    k+=1
                crop(event)
            self.fig.canvas.draw_idle()

        def reset_crop(event):
            self.Stokes = deepcopy(Stokes)
            self.region = None
            self.pol_int()
            self.ax.reset_wcs(self.wcs)
            self.ax_cosmetics()
            self.display()
            self.pol_vector()

        b_crop.on_clicked(crop)
        b_crop_reset.on_clicked(reset_crop)

        #Set axe for saving plot
        ax_save = self.fig.add_axes([0.850, 0.070, 0.05, 0.02])
        b_save = Button(ax_save, "Save")
        b_save.label.set_fontsize(8)
        ax_text_save = self.fig.add_axes([0.3, 0.020, 0.5, 0.025],visible=False)
        text_save = TextBox(ax_text_save, "Save to:", initial='')

        def saveplot(event):
            ax_text_save.set(visible=True)
            ax_snr_reset.set(visible=False)
            ax_save.set(visible=False)
            ax_dump.set(visible=False)
            self.fig.canvas.draw_idle()

        b_save.on_clicked(saveplot)

        def submit_save(expression):
            ax_text_save.set(visible=False)
            if expression != '':
                save_fig = plt.figure(figsize=(15,15))
                save_ax = save_fig.add_subplot(111, projection=self.wcs)
                self.ax_cosmetics(ax=save_ax)
                self.display(fig=save_fig,ax=save_ax)
                self.pol_vector(fig=save_fig,ax=save_ax)
                self.pol_int(fig=save_fig,ax=save_ax)
                save_fig.suptitle(r"{0:s} with $SNR_{{p}} \geq$ {1:d} and $SNR_{{I}} \geq$ {2:d}".format(self.targ, int(self.SNRp), int(self.SNRi)))
                if not expression[-4:] in ['.png', '.jpg']:
                    expression += '.png'
                save_fig.savefig(expression, bbox_inches='tight', dpi=200)
                plt.close(save_fig)
                text_save.set_val('')
            ax_snr_reset.set(visible=True)
            ax_save.set(visible=True)
            ax_dump.set(visible=True)
            self.fig.canvas.draw_idle()

        text_save.on_submit(submit_save)

        #Set axe for data dump
        ax_dump = self.fig.add_axes([0.850, 0.045, 0.05, 0.02])
        b_dump = Button(ax_dump, "Dump")
        b_dump.label.set_fontsize(8)
        ax_text_dump = self.fig.add_axes([0.3, 0.020, 0.5, 0.025],visible=False)
        text_dump = TextBox(ax_text_dump, "Dump to:", initial='')

        def dump(event):
            ax_text_dump.set(visible=True)
            ax_snr_reset.set(visible=False)
            ax_save.set(visible=False)
            ax_dump.set(visible=False)
            self.fig.canvas.draw_idle()

            shape = np.array(self.I.shape)
            center = (shape/2).astype(int)
            cdelt_arcsec = self.wcs.wcs.cdelt*3600
            xx, yy = np.indices(shape)
            x, y = (xx-center[0])*cdelt_arcsec[0], (yy-center[1])*cdelt_arcsec[1]

            P, PA = np.zeros(shape), np.zeros(shape)
            P[self.cut] = self.P[self.cut]
            PA[self.cut] = self.PA[self.cut]
            dump_list = []
            for i in range(shape[0]):
                for j in range(shape[1]):
                    dump_list.append([x[i,j], y[i,j], self.I[i,j]*self.convert_flux, self.Q[i,j]*self.convert_flux, self.U[i,j]*self.convert_flux, P[i,j], PA[i,j]])
            self.data_dump = np.array(dump_list)

        b_dump.on_clicked(dump)

        def submit_dump(expression):
            ax_text_dump.set(visible=False)
            if expression != '':
                if not expression[-4:] in ['.txt', '.dat']:
                    expression += '.txt'
                np.savetxt(expression, self.data_dump)
                text_dump.set_val('')
            ax_snr_reset.set(visible=True)
            ax_save.set(visible=True)
            ax_dump.set(visible=True)
            self.fig.canvas.draw_idle()

        text_dump.on_submit(submit_dump)

        #Set axes for display buttons
        ax_tf = self.fig.add_axes([0.925, 0.105, 0.05, 0.02])
        ax_pf = self.fig.add_axes([0.925, 0.085, 0.05, 0.02])
        ax_p = self.fig.add_axes([0.925, 0.065, 0.05, 0.02])
        ax_pa = self.fig.add_axes([0.925, 0.045, 0.05, 0.02])
        ax_snri = self.fig.add_axes([0.925, 0.025, 0.05, 0.02])
        ax_snrp = self.fig.add_axes([0.925, 0.005, 0.05, 0.02])
        b_tf = Button(ax_tf,r"$F_{\lambda}$")
        b_pf = Button(ax_pf,r"$F_{\lambda} \cdot P$")
        b_p = Button(ax_p,r"$P$")
        b_pa = Button(ax_pa,r"$\theta_{P}$")
        b_snri = Button(ax_snri,r"$I / \sigma_{I}$")
        b_snrp = Button(ax_snrp,r"$P / \sigma_{P}$")

        def d_tf(event):
            self.display_selection = 'total_flux'
            self.display()
            self.pol_int()
        b_tf.on_clicked(d_tf)

        def d_pf(event):
            self.display_selection = 'pol_flux'
            self.display()
            self.pol_int()
        b_pf.on_clicked(d_pf)

        def d_p(event):
            self.display_selection = 'pol_deg'
            self.display()
            self.pol_int()
        b_p.on_clicked(d_p)

        def d_pa(event):
            self.display_selection = 'pol_ang'
            self.display()
            self.pol_int()
        b_pa.on_clicked(d_pa)

        def d_snri(event):
            self.display_selection = 'snri'
            self.display()
            self.pol_int()
        b_snri.on_clicked(d_snri)

        def d_snrp(event):
            self.display_selection = 'snrp'
            self.display()
            self.pol_int()
        b_snrp.on_clicked(d_snrp)

        plt.show()

    @property
    def wcs(self):
        return deepcopy(WCS(self.Stokes[0].header))
    @property
    def I(self):
        return self.Stokes[np.argmax([self.Stokes[i].header['datatype']=='I_stokes' for i in range(len(self.Stokes))])].data
    @property
    def Q(self):
        return self.Stokes[np.argmax([self.Stokes[i].header['datatype']=='Q_stokes' for i in range(len(self.Stokes))])].data
    @property
    def U(self):
        return self.Stokes[np.argmax([self.Stokes[i].header['datatype']=='U_stokes' for i in range(len(self.Stokes))])].data
    @property
    def IQU_cov(self):
        return self.Stokes[np.argmax([self.Stokes[i].header['datatype']=='IQU_cov_matrix' for i in range(len(self.Stokes))])].data
    @property
    def P(self):
        return self.Stokes[np.argmax([self.Stokes[i].header['datatype']=='Pol_deg_debiased' for i in range(len(self.Stokes))])].data
    @property
    def s_P(self):
        return self.Stokes[np.argmax([self.Stokes[i].header['datatype']=='Pol_deg_err' for i in range(len(self.Stokes))])].data
    @property
    def PA(self):
        return self.Stokes[np.argmax([self.Stokes[i].header['datatype']=='Pol_ang' for i in range(len(self.Stokes))])].data
    @property
    def data_mask(self):
        return self.Stokes[np.argmax([self.Stokes[i].header['datatype']=='Data_mask' for i in range(len(self.Stokes))])].data

    def set_data_mask(self, mask):
        self.Stokes[np.argmax([self.Stokes[i].header['datatype']=='Data_mask' for i in range(len(self.Stokes))])].data = mask.astype(float)

    @property
    def cut(self):
        s_I = np.sqrt(self.IQU_cov[0,0])
        SNRp_mask, SNRi_mask = np.zeros(self.P.shape).astype(bool), np.zeros(self.I.shape).astype(bool)
        SNRp_mask[self.s_P > 0.] = self.P[self.s_P > 0.] / self.s_P[self.s_P > 0.] > self.SNRp
        SNRi_mask[s_I > 0.] = self.I[s_I > 0.] / s_I[s_I > 0.] > self.SNRi
        return np.logical_and(SNRi_mask,SNRp_mask)

    def ax_cosmetics(self, ax=None):
        if ax is None:
            ax = self.ax
        ax.set_facecolor('black')

        ax.coords.grid(True, color='white', ls='dotted', alpha=0.5)
        ax.coords[0].set_axislabel('Right Ascension (J2000)')
        ax.coords[0].set_axislabel_position('t')
        ax.coords[0].set_ticklabel_position('t')
        ax.coords[1].set_axislabel('Declination (J2000)')
        ax.coords[1].set_axislabel_position('l')
        ax.coords[1].set_ticklabel_position('l')
        ax.axis('equal')

        #Display scales and orientation
        fontprops = fm.FontProperties(size=14)
        px_size = self.wcs.wcs.cdelt[0]*3600.
        if hasattr(self,'px_sc'):
            self.px_sc.remove()
        self.px_sc = AnchoredSizeBar(ax.transData, 1./px_size, '1 arcsec', 3, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color='white', fontproperties=fontprops)
        ax.add_artist(self.px_sc)
        if hasattr(self,'pol_sc'):
            self.pol_sc.remove()
        self.pol_sc = AnchoredSizeBar(ax.transData, 2., r"$P$= 100%", 4, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color='white', fontproperties=fontprops)
        ax.add_artist(self.pol_sc)
        if hasattr(self,'north_dir'):
            self.north_dir.remove()
        self.north_dir = AnchoredDirectionArrows(ax.transAxes, "E", "N", length=-0.08, fontsize=0.025, loc=1, aspect_ratio=-1, sep_y=0.01, sep_x=0.01, back_length=0., head_length=10., head_width=10., angle=-self.Stokes[0].header['orientat'], color='white', text_props={'ec': None, 'fc': 'w', 'alpha': 1, 'lw': 0.4}, arrow_props={'ec': None,'fc':'w','alpha': 1,'lw': 1})
        ax.add_artist(self.north_dir)

    def display(self, fig=None, ax=None):
        norm = None
        if self.display_selection is None:
            self.display_selection = "total_flux"
        if self.display_selection.lower() in ['total_flux']:
            self.data = self.I*self.convert_flux
            vmin, vmax = np.max(np.sqrt(self.IQU_cov[0,0][self.cut])*self.convert_flux), np.max(self.data[self.data > 0.])
            norm = LogNorm(vmin, vmax)
            label = r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]"
        elif self.display_selection.lower() in ['pol_flux']:
            self.data = self.I*self.convert_flux*self.P
            vmin, vmax = np.max(np.sqrt(self.IQU_cov[0,0][self.cut])*self.convert_flux), np.max(self.I[self.data > 0.]*self.convert_flux)
            norm = LogNorm(vmin, vmax)
            label = r"$F_{\lambda} \cdot P$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]"
        elif self.display_selection.lower() in ['pol_deg']:
            self.data = self.P*100.
            vmin, vmax = 0., 100. #np.max(self.data[self.data > 0.])
            label = r"$P$ [%]"
        elif self.display_selection.lower() in ['pol_ang']:
            self.data = princ_angle(self.PA)
            vmin, vmax = 0, 180.
            label = r"$\theta_{P}$ [°]"
        elif self.display_selection.lower() in ['snri']:
            s_I = np.sqrt(self.IQU_cov[0,0])
            SNRi = np.zeros(self.I.shape)
            SNRi[s_I > 0.] = self.I[s_I > 0.]/s_I[s_I > 0.]
            self.data = SNRi
            vmin, vmax = 0., np.max(self.data[self.data > 0.])
            label = r"$I_{Stokes}/\sigma_{I}$"
        elif self.display_selection.lower() in ['snrp']:
            SNRp = np.zeros(self.P.shape)
            SNRp[self.s_P > 0.] = self.P[self.s_P > 0.]/self.s_P[self.s_P > 0.]
            self.data = SNRp
            vmin, vmax = 0., np.max(self.data[self.data > 0.])
            label = r"$P/\sigma_{P}$"

        if fig is None:
            fig = self.fig
            if ax is None:
                ax = self.ax
            if hasattr(self, 'im'):
                self.im.remove()
            if not norm is None:
                self.im = ax.imshow(self.data, norm=norm, aspect='equal', cmap='inferno')
            else:
                self.im = ax.imshow(self.data, vmin=vmin, vmax=vmax, aspect='equal', cmap='inferno')
            self.cbar = plt.colorbar(self.im, cax=self.cbar_ax, label=label)
            fig.canvas.draw_idle()
            return self.im
        else:
            if not norm is None:
                im = ax.imshow(self.data, norm=norm, aspect='equal', cmap='inferno')
            else:
                im = ax.imshow(self.data, vmin=vmin, vmax=vmax, aspect='equal', cmap='inferno')
            ax.set_xlim(0,self.data.shape[1])
            ax.set_ylim(0,self.data.shape[0])
            plt.colorbar(im, pad=0.025, aspect=80, label=label)
            fig.canvas.draw_idle()

    def pol_vector(self, fig=None, ax=None):
        P_cut = np.ones(self.P.shape)*np.nan
        P_cut[self.cut] = self.P[self.cut]
        X, Y = np.meshgrid(np.arange(self.I.shape[1]),np.arange(self.I.shape[0]))
        XY_U, XY_V = P_cut*np.cos(np.pi/2. + self.PA*np.pi/180.), P_cut*np.sin(np.pi/2. + self.PA*np.pi/180.)

        if fig is None:
            fig = self.fig
            if ax is None:
                ax = self.ax
            if hasattr(self, 'quiver'):
                self.quiver.remove()
            self.quiver = ax.quiver(X, Y, XY_U, XY_V, units='xy', scale=0.5, scale_units='xy', pivot='mid', headwidth=0., headlength=0., headaxislength=0., width=0.1, color='white')
            fig.canvas.draw_idle()
            return self.quiver
        else:
            ax.quiver(X, Y, XY_U, XY_V, units='xy', scale=0.5, scale_units='xy', pivot='mid', headwidth=0., headlength=0., headaxislength=0., width=0.1, color='white')
            fig.canvas.draw_idle()

    def pol_int(self, fig=None, ax=None):
        if self.region is None:
            n_pix = self.I.size
            s_I = np.sqrt(self.IQU_cov[0,0])
            I_reg = self.I.sum()
            I_reg_err = np.sqrt(n_pix)*np.sqrt(np.sum(s_I**2))
            P_reg = self.Stokes[0].header['P_int']
            P_reg_err = self.Stokes[0].header['P_int_err']
            PA_reg = self.Stokes[0].header['PA_int']
            PA_reg_err = self.Stokes[0].header['PA_int_err']

            s_I = np.sqrt(self.IQU_cov[0,0])
            s_Q = np.sqrt(self.IQU_cov[1,1])
            s_U = np.sqrt(self.IQU_cov[2,2])
            s_IQ = self.IQU_cov[0,1]
            s_IU = self.IQU_cov[0,2]
            s_QU = self.IQU_cov[1,2]

            I_cut = self.I[self.cut].sum()
            Q_cut = self.Q[self.cut].sum()
            U_cut = self.U[self.cut].sum()
            I_cut_err = np.sqrt(np.sum(s_I[self.cut]**2))
            Q_cut_err = np.sqrt(np.sum(s_Q[self.cut]**2))
            U_cut_err = np.sqrt(np.sum(s_U[self.cut]**2))
            IQ_cut_err = np.sqrt(np.sum(s_IQ[self.cut]**2))
            IU_cut_err = np.sqrt(np.sum(s_IU[self.cut]**2))
            QU_cut_err = np.sqrt(np.sum(s_QU[self.cut]**2))

            P_cut = np.sqrt(Q_cut**2+U_cut**2)/I_cut
            P_cut_err = np.sqrt((Q_cut**2*Q_cut_err**2 + U_cut**2*U_cut_err**2 + 2.*Q_cut*U_cut*QU_cut_err)/(Q_cut**2 + U_cut**2) + ((Q_cut/I_cut)**2 + (U_cut/I_cut)**2)*I_cut_err**2 - 2.*(Q_cut/I_cut)*IQ_cut_err - 2.*(U_cut/I_cut)*IU_cut_err)/I_cut

            PA_cut = princ_angle(np.degrees((1./2.)*np.arctan2(U_cut,Q_cut)))
            PA_cut_err = princ_angle(np.degrees((1./(2.*(Q_cut**2+U_cut**2)))*np.sqrt(U_cut**2*Q_cut_err**2 + Q_cut**2*U_cut_err**2 - 2.*Q_cut*U_cut*QU_cut_err)))

        else:
            n_pix = self.I[self.region].size
            s_I = np.sqrt(self.IQU_cov[0,0])
            s_Q = np.sqrt(self.IQU_cov[1,1])
            s_U = np.sqrt(self.IQU_cov[2,2])
            s_IQ = self.IQU_cov[0,1]
            s_IU = self.IQU_cov[0,2]
            s_QU = self.IQU_cov[1,2]

            I_reg = self.I[self.region].sum()
            Q_reg = self.Q[self.region].sum()
            U_reg = self.U[self.region].sum()
            I_reg_err = np.sqrt(np.sum(s_I[self.region]**2))
            Q_reg_err = np.sqrt(np.sum(s_Q[self.region]**2))
            U_reg_err = np.sqrt(np.sum(s_U[self.region]**2))
            IQ_reg_err = np.sqrt(np.sum(s_IQ[self.region]**2))
            IU_reg_err = np.sqrt(np.sum(s_IU[self.region]**2))
            QU_reg_err = np.sqrt(np.sum(s_QU[self.region]**2))

            P_reg = np.sqrt(Q_reg**2+U_reg**2)/I_reg
            P_reg_err = np.sqrt((Q_reg**2*Q_reg_err**2 + U_reg**2*U_reg_err**2 + 2.*Q_reg*U_reg*QU_reg_err)/(Q_reg**2 + U_reg**2) + ((Q_reg/I_reg)**2 + (U_reg/I_reg)**2)*I_reg_err**2 - 2.*(Q_reg/I_reg)*IQ_reg_err - 2.*(U_reg/I_reg)*IU_reg_err)/I_reg

            PA_reg = princ_angle((90./np.pi)*np.arctan2(U_reg,Q_reg))
            PA_reg_err = (90./(np.pi*(Q_reg**2+U_reg**2)))*np.sqrt(U_reg**2*Q_reg_err**2 + Q_reg**2*U_reg_err**2 - 2.*Q_reg*U_reg*QU_reg_err)

            new_cut = np.logical_and(self.region, self.cut)
            I_cut = self.I[new_cut].sum()
            Q_cut = self.Q[new_cut].sum()
            U_cut = self.U[new_cut].sum()
            I_cut_err = np.sqrt(np.sum(s_I[new_cut]**2))
            Q_cut_err = np.sqrt(np.sum(s_Q[new_cut]**2))
            U_cut_err = np.sqrt(np.sum(s_U[new_cut]**2))
            IQ_cut_err = np.sqrt(np.sum(s_IQ[new_cut]**2))
            IU_cut_err = np.sqrt(np.sum(s_IU[new_cut]**2))
            QU_cut_err = np.sqrt(np.sum(s_QU[new_cut]**2))

            P_cut = np.sqrt(Q_cut**2+U_cut**2)/I_cut
            P_cut_err = np.sqrt((Q_cut**2*Q_cut_err**2 + U_cut**2*U_cut_err**2 + 2.*Q_cut*U_cut*QU_cut_err)/(Q_cut**2 + U_cut**2) + ((Q_cut/I_cut)**2 + (U_cut/I_cut)**2)*I_cut_err**2 - 2.*(Q_cut/I_cut)*IQ_cut_err - 2.*(U_cut/I_cut)*IU_cut_err)/I_cut

            PA_cut = 360.-princ_angle((90./np.pi)*np.arctan2(U_cut,Q_cut))
            PA_cut_err = (90./(np.pi*(Q_cut**2+U_cut**2)))*np.sqrt(U_cut**2*Q_cut_err**2 + Q_cut**2*U_cut_err**2 - 2.*Q_cut*U_cut*QU_cut_err)

        if hasattr(self, 'cont'):
            for coll in self.cont.collections:
                try:
                    coll.remove()
                except:
                    return
            del self.cont
        if fig is None:
            fig = self.fig
            if ax is None:
                ax = self.ax
            if hasattr(self, 'an_int'):
                self.an_int.remove()
            self.an_int = ax.annotate(r"$F_{{\lambda}}^{{int}}$({0:.0f} $\AA$) = {1} $ergs \cdot cm^{{-2}} \cdot s^{{-1}} \cdot \AA^{{-1}}$".format(self.pivot_wav,sci_not(I_reg*self.convert_flux,I_reg_err*self.convert_flux,2))+"\n"+r"$P^{{int}}$ = {0:.1f} $\pm$ {1:.1f} %".format(P_reg*100.,np.ceil(P_reg_err*1000.)/10.)+"\n"+r"$\theta_{{P}}^{{int}}$ = {0:.1f} $\pm$ {1:.1f} °".format(PA_reg,np.ceil(PA_reg_err*10.)/10.), color='white', fontsize=12, xy=(0.01, 0.93), xycoords='axes fraction')
            #self.an_int = ax.annotate(r"$F_{{\lambda}}^{{int}}$({0:.0f} $\AA$) = {1} $ergs \cdot cm^{{-2}} \cdot s^{{-1}} \cdot \AA^{{-1}}$".format(self.pivot_wav,sci_not(I_reg*self.convert_flux,I_reg_err*self.convert_flux,2))+"\n"+r"$P^{{int}}$ = {0:.1f} $\pm$ {1:.1f} %".format(P_reg*100.,np.ceil(P_reg_err*1000.)/10.)+"\n"+r"$\theta_{{P}}^{{int}}$ = {0:.1f} $\pm$ {1:.1f} °".format(PA_reg,np.ceil(PA_reg_err*10.)/10.)+"\n"+r"$P^{{cut}}$ = {0:.1f} $\pm$ {1:.1f} %".format(P_cut*100.,np.ceil(P_cut_err*1000.)/10.)+"\n"+r"$\theta_{{P}}^{{cut}}$ = {0:.1f} $\pm$ {1:.1f} °".format(PA_cut,np.ceil(PA_cut_err*10.)/10.), color='white', fontsize=12, xy=(0.01, 0.85), xycoords='axes fraction')
            if not self.region is None:
                self.cont = ax.contour(self.region.astype(float),levels=[0.5], colors='white', linewidths=0.8)
            fig.canvas.draw_idle()
            return self.an_int
        else:
            ax.annotate(r"$F_{{\lambda}}^{{int}}$({0:.0f} $\AA$) = {1} $ergs \cdot cm^{{-2}} \cdot s^{{-1}} \cdot \AA^{{-1}}$".format(self.pivot_wav,sci_not(I_reg*self.convert_flux,I_reg_err*self.convert_flux,2))+"\n"+r"$P^{{int}}$ = {0:.1f} $\pm$ {1:.1f} %".format(P_reg*100.,np.ceil(P_reg_err*1000.)/10.)+"\n"+r"$\theta_{{P}}^{{int}}$ = {0:.1f} $\pm$ {1:.1f} °".format(PA_reg,np.ceil(PA_reg_err*10.)/10.), color='white', fontsize=12, xy=(0.01, 0.94), xycoords='axes fraction')
            #ax.annotate(r"$F_{{\lambda}}^{{int}}$({0:.0f} $\AA$) = {1} $ergs \cdot cm^{{-2}} \cdot s^{{-1}} \cdot \AA^{{-1}}$".format(self.pivot_wav,sci_not(I_reg*self.convert_flux,I_reg_err*self.convert_flux,2))+"\n"+r"$P^{{int}}$ = {0:.1f} $\pm$ {1:.1f} %".format(P_reg*100.,np.ceil(P_reg_err*1000.)/10.)+"\n"+r"$\theta_{{P}}^{{int}}$ = {0:.1f} $\pm$ {1:.1f} °".format(PA_reg,np.ceil(PA_reg_err*10.)/10.)+"\n"+r"$P^{{cut}}$ = {0:.1f} $\pm$ {1:.1f} %".format(P_cut*100.,np.ceil(P_cut_err*1000.)/10.)+"\n"+r"$\theta_{{P}}^{{cut}}$ = {0:.1f} $\pm$ {1:.1f} °".format(PA_cut,np.ceil(PA_cut_err*10.)/10.), color='white', fontsize=12, xy=(0.01, 0.90), xycoords='axes fraction')
            if not self.region is None:
                ax.contour(self.region.astype(float),levels=[0.5], colors='white', linewidths=0.8)
            fig.canvas.draw_idle()
